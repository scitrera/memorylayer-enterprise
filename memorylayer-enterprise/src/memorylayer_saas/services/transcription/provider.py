"""One OCR endpoint, spoken to over OpenAI ``/v1/chat/completions``.

A provider is a contract (:mod:`.contracts`) plus a URL, auth, and timing. It
owns no process lifecycle: the model is served by a sparkrun recipe on a managed
box, or by an on-demand platform that spins itself up. That separation is the
point — moving a rung from Thunder to Modal is a URL and an auth mode, not code.
"""

from __future__ import annotations

import asyncio
import base64
import random
import time
from dataclasses import dataclass, field
from logging import Logger

import httpx

from .contracts import OcrModelContract, clean_transcription_output
from .regions import PageRegion

AUTH_NONE = "none"
AUTH_BEARER = "bearer"
AUTH_MODAL_PROXY = "modal_proxy"
KNOWN_AUTH_KINDS = (AUTH_NONE, AUTH_BEARER, AUTH_MODAL_PROXY)

DEFAULT_MAX_TOKENS = 16384
DEFAULT_TIMEOUT_SEC = 120.0
#: Pages (and figure captions) in flight at once, PER WORKER PROCESS. Measured
#: against the live lane over 8 text-dense pages:
#:
#:     concurrency=1   56.3s   7.0s/page   --
#:     concurrency=4   23.8s   3.0s/page   2.37x
#:     concurrency=8   17.3s   2.2s/page   3.25x
#:
#: 8 is genuinely faster, so 4 is a deliberate choice rather than the best
#: number: the lane shares one GPU with the chat lane, and fleet load is
#: (worker replicas x this) against the proxy's ~50 ceiling -- 4 leaves room for
#: ~12 workers, 8 for ~6. Raise it via MEMORYLAYER_TRANSCRIBE_CONCURRENCY when
#: the replica count and the ceiling are known to accommodate it.
DEFAULT_TRANSCRIBE_CONCURRENCY = 4

#: Attempts (not retries) per page against ONE provider when the upstream says
#: it is overloaded. The proxy in front of a lane enforces a fleet-wide
#: concurrency ceiling and rejects past its queue with 503 -- a fail-fast bounded
#: queue, not a blocking one. Without a retry that rejection becomes a DROPPED
#: PAGE, so backing off here is what turns the ceiling into latency instead of
#: data loss.
DEFAULT_OVERLOAD_ATTEMPTS = 3
DEFAULT_OVERLOAD_BACKOFF_SEC = 1.0

#: How long a pooled connection may sit idle before we retire it. Chosen BELOW
#: the shortest keep-alive window on the path (uvicorn's default is 5s) so the
#: client always drops a connection before the server does, rather than
#: discovering a closed socket on the next request.
KEEPALIVE_EXPIRY_SEC = 4.0

#: Statuses that mean "try again shortly", not "this request is bad".
#: 503 is Envoy circuit-breaker overflow (response flag UO, usually with
#: x-envoy-overloaded); 429 is a rate limiter; 502/504 are transient upstream
#: blips. A read timeout is deliberately NOT retried: on a multi-minute OCR
#: budget it means real trouble, and retrying doubles the load that caused it.
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})

#: Transport failures worth another attempt, i.e. "the connection broke", not
#: "the server took too long".
#:
#: ``NetworkError`` is httpx's family for a socket that died (ConnectError,
#: ReadError, WriteError, CloseError) and deliberately EXCLUDES the timeout
#: family, so the read-timeout reasoning above still holds. ``RemoteProtocolError``
#: covers a peer that closed mid-response, and ``ConnectTimeout`` is cheap to
#: retry because nothing was ever sent.
#:
#: These must be retried alongside the statuses above: a pooled keep-alive
#: connection closed by the far end fails on the NEXT request, before a single
#: response byte arrives, so the request never reached the model at all.
#: Retrying picks up a fresh connection. Without this the whole page fails on
#: an error that had nothing to do with the page.
_RETRYABLE_TRANSPORT = (
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
)

# Finish reasons that mean "the content is unusable", not "here is your text".
_REJECTED_FINISH_REASONS = frozenset({"recitation", "content_filter", "safety"})
_LENGTH_FINISH_REASONS = frozenset({"length", "max_length", "max_tokens"})


@dataclass
class TranscriptionAttempt:
    """Diagnostics for one provider's shot at one page."""

    provider: str
    model: str
    success: bool = False
    content: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "unknown"
    error: str | None = None
    #: Layout segmentation, when the contract produced one. Carries the figure
    #: boxes, which are the only handle on illustrations the model emits no
    #: text for.
    regions: list[PageRegion] = field(default_factory=list)


@dataclass
class ProviderConfig:
    """Everything that distinguishes one cascade rung from another."""

    name: str
    base_url: str
    contract: OcrModelContract
    model: str = ""
    auth_kind: str = AUTH_NONE
    auth_token: str = ""
    auth_key: str = ""
    auth_secret: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    #: Sampling temperature, OMITTED from the request when None so the model's
    #: own serving default applies. Previously pinned to 0.0, which is greedy
    #: decoding -- the most repetition-prone setting there is, and this model
    #: demonstrably falls into emitting a ladder of zero-width regions on dense
    #: pages. An OCR model ships a default its authors chose; overriding it with
    #: the loopiest possible value was not a decision anyone made deliberately.
    temperature: float | None = None
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    #: Budget for this provider's FIRST request. An on-demand endpoint (Modal)
    #: boots a GPU on demand, so the first call can take minutes while steady
    #: state is seconds. 0 means "same as timeout_sec".
    cold_start_timeout_sec: float = 0.0
    overload_attempts: int = DEFAULT_OVERLOAD_ATTEMPTS
    overload_backoff_sec: float = DEFAULT_OVERLOAD_BACKOFF_SEC

    def auth_headers(self) -> dict[str, str]:
        if self.auth_kind == AUTH_BEARER:
            return {"Authorization": "Bearer %s" % self.auth_token}
        if self.auth_kind == AUTH_MODAL_PROXY:
            return {"Modal-Key": self.auth_key, "Modal-Secret": self.auth_secret}
        return {}


def image_to_data_url(image_b64: str) -> str:
    """Wrap already-base64 page bytes as a data URL, tolerating a full URL."""
    if image_b64.startswith("data:"):
        return image_b64
    return "data:image/png;base64,%s" % image_b64


def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


class OpenAIChatTranscriptionProvider:
    """Transcribe one page via chat-completions against a single endpoint."""

    def __init__(self, config: ProviderConfig, logger: Logger, client: httpx.AsyncClient | None = None):
        self.config = config
        self.logger = logger
        self._client = client
        self._owns_client = client is None
        self._first_request_done = False

    @property
    def name(self) -> str:
        return self.config.name

    async def connect(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                timeout=self.config.timeout_sec,
                headers=self.config.auth_headers(),
                # Retire pooled connections well before the far end does.
                # Transcription is bursty -- a batch, then a gap while pages
                # render -- so pooled sockets sit idle for long stretches. The
                # server (uvicorn defaults to a 5s keep-alive) closes them, and
                # the next request goes out on a socket that is already gone,
                # failing before any response byte arrives. Expiring first means
                # we open a fresh connection instead of inheriting a dead one.
                limits=httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY_SEC),
            )
            self._owns_client = True

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _request_timeout(self) -> float:
        """Cold-start budget until a request has completed, steady-state after.

        Unsynchronized on purpose: with several requests in flight the whole
        first burst sees the flag unset and every one of them gets the cold-start
        budget. That is the wanted behavior — if the endpoint really is cold,
        they are all waiting on the same boot.
        """
        cold = self.config.cold_start_timeout_sec
        if not self._first_request_done and cold > 0:
            return cold
        return self.config.timeout_sec

    def _backoff_delay(self, attempt_number: int, response: httpx.Response | None) -> float:
        """Exponential backoff with jitter, honoring Retry-After when sent.

        Jitter matters more than usual here: a batch rejected by the proxy's
        queue was submitted concurrently, so a fixed delay would march the whole
        batch back into the ceiling in lockstep.
        """
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        base = self.config.overload_backoff_sec * (2 ** (attempt_number - 1))
        return base * (0.5 + random.random())

    async def _post_with_overload_retry(self, payload: dict) -> httpx.Response:
        """POST the completion, retrying only while the upstream says it is busy."""
        attempts = max(1, int(self.config.overload_attempts))
        last: httpx.Response | None = None
        for attempt_number in range(1, attempts + 1):
            try:
                response = await self._client.post(
                    "/chat/completions", json=payload, timeout=self._request_timeout(),
                )
            except _RETRYABLE_TRANSPORT as exc:
                # A dropped connection, not a rejected request. Mark the cold
                # start done anyway: the endpoint answered the socket even if it
                # then went away, and a second cold-start budget would stall the
                # whole batch.
                self._first_request_done = True
                if attempt_number == attempts:
                    raise
                delay = self._backoff_delay(attempt_number, None)
                self.logger.info(
                    "%s connection dropped (%s: %s); retry %d/%d in %.1fs",
                    self.config.name,
                    type(exc).__name__,
                    exc or "no detail",
                    attempt_number,
                    attempts - 1,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            self._first_request_done = True
            if response.status_code not in _RETRYABLE_STATUS:
                return response
            last = response
            if attempt_number == attempts:
                break
            delay = self._backoff_delay(attempt_number, response)
            self.logger.info(
                "%s upstream busy (HTTP %d%s); retry %d/%d in %.1fs",
                self.config.name,
                response.status_code,
                " overloaded" if response.headers.get("x-envoy-overloaded") else "",
                attempt_number,
                attempts - 1,
                delay,
            )
            await asyncio.sleep(delay)
        return last

    async def transcribe_page(self, image_b64: str, system_prompt: str | None = None) -> TranscriptionAttempt:
        """One page, one endpoint. Never raises — the cascade reads the attempt."""
        cfg = self.config
        attempt = TranscriptionAttempt(provider=cfg.name, model=cfg.model)
        started = time.monotonic()

        try:
            await self.connect()
            payload: dict = {
                "model": cfg.model,
                "messages": cfg.contract.build_messages(system_prompt, image_to_data_url(image_b64)),
                "max_tokens": cfg.max_tokens,
            }
            # Omitted unless explicitly configured, so the model's serving
            # default applies rather than a hardcoded greedy 0.0.
            if cfg.temperature is not None:
                payload["temperature"] = cfg.temperature
            if cfg.contract.extra_body:
                payload.update(cfg.contract.extra_body)

            response = await self._post_with_overload_retry(payload)
            response.raise_for_status()
            body = response.json()

            choice = (body.get("choices") or [{}])[0]
            raw = (choice.get("message") or {}).get("content") or ""
            attempt.finish_reason = (choice.get("finish_reason") or "unknown").lower()

            usage = body.get("usage") or {}
            attempt.tokens_in = usage.get("prompt_tokens") or 0
            attempt.tokens_out = usage.get("completion_tokens") or 0

            if attempt.finish_reason in _REJECTED_FINISH_REASONS:
                attempt.error = "Rejected finish reason: %s" % attempt.finish_reason
            elif attempt.finish_reason in _LENGTH_FINISH_REASONS:
                attempt.error = "Token limit reached: %s" % attempt.finish_reason
            else:
                regions: list[PageRegion] = []
                if cfg.contract.extract is not None:
                    raw, regions = cfg.contract.extract(raw)
                content = clean_transcription_output(raw)
                attempt.regions = regions
                if content:
                    attempt.content = content
                    attempt.success = True
                else:
                    # For a fixed-recipe model this nearly always means the
                    # prompt/decode contract drifted, not that the page is blank.
                    attempt.error = "Empty content after cleaning"
        except Exception as e:  # noqa: BLE001 — the cascade collects failures
            self._first_request_done = True
            attempt.error = "%s: %s" % (type(e).__name__, e)
            self.logger.warning(
                "Transcription provider %s failed: %s", cfg.name, e,
            )

        attempt.latency_ms = (time.monotonic() - started) * 1000
        return attempt
