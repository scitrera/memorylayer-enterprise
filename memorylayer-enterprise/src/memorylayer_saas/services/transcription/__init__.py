"""Page transcription for the document ingestion pipeline.

Selected with ``MEMORYLAYER_TRANSCRIPTION_SERVICE``:

``embed_server`` (default)
    Today's path — an embed-server process serving ``POST /v1/transcribe``.

``direct``
    An in-process cascade over model endpoints (sparkrun-served lanes, or any
    OpenAI-compatible endpoint), configured by ``MEMORYLAYER_TRANSCRIBE_CASCADE``
    plus per-profile keys. Removes embed-server from the deployment topology.
"""

from __future__ import annotations

from logging import Logger

from scitrera_app_framework import Variables, ext_parse_bool, get_extension
from scitrera_app_framework.api import Plugin, enabled_option_pattern

from ...config import (
    DEFAULT_MEMORYLAYER_TRANSCRIBE_CASCADE,
    DEFAULT_MEMORYLAYER_TRANSCRIBE_CONTRACT,
    DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MODEL,
    DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_PROFILE,
    DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_REASONING_EFFORT,
    DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTIONS,
    DEFAULT_MEMORYLAYER_TRANSCRIPTION_SERVICE,
    MEMORYLAYER_TRANSCRIBE_CASCADE,
    MEMORYLAYER_TRANSCRIBE_CONCURRENCY,
    MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_CONTEXT_CHARS,
    MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_INSTRUCTION,
    MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MAX_TOKENS,
    MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MODEL,
    MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_PROFILE,
    MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_REASONING_EFFORT,
    MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTIONS,
    MEMORYLAYER_TRANSCRIBE_OVERLOAD_ATTEMPTS,
    MEMORYLAYER_TRANSCRIBE_OVERLOAD_BACKOFF_SEC,
    MEMORYLAYER_TRANSCRIBE_PROFILE_PREFIX,
    MEMORYLAYER_TRANSCRIPTION_SERVICE,
)
from .captions import (
    DEFAULT_FIGURE_CAPTION_CONTEXT_CHARS,
    DEFAULT_FIGURE_CAPTION_INSTRUCTION,
    DEFAULT_FIGURE_CAPTION_MAX_TOKENS,
    DEFAULT_FIGURE_CAPTION_PROFILE,
    FigureCaptioner,
    build_figure_context,
)
from .cascade import CascadeTranscriber
from .contracts import KNOWN_CONTRACTS, OcrModelContract, build_contract, strip_grounding_tokens
from .figures import bbox_to_pixels, crop_figures
from .provider import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLOAD_ATTEMPTS,
    DEFAULT_OVERLOAD_BACKOFF_SEC,
    DEFAULT_TIMEOUT_SEC,
    DEFAULT_TRANSCRIBE_CONCURRENCY,
    KNOWN_AUTH_KINDS,
    OpenAIChatTranscriptionProvider,
    ProviderConfig,
    TranscriptionAttempt,
)
from .regions import (
    BBOX_SCALE,
    DEFAULT_DROP_LABELS,
    PageRegion,
    extract_grounded_page,
    parse_grounded_page,
    render_transcript,
)
from .result import TranscribedPage, pages_from_embed_server_response
from .service import (
    DirectTranscriptionService,
    EmbedServerTranscriptionService,
    TranscriptionService,
)

EXT_TRANSCRIPTION_SERVICE = "memorylayer-enterprise-transcription-service"

__all__ = [
    "EXT_TRANSCRIPTION_SERVICE",
    "BBOX_SCALE",
    "DEFAULT_DROP_LABELS",
    "KNOWN_AUTH_KINDS",
    "KNOWN_CONTRACTS",
    "CascadeTranscriber",
    "FigureCaptioner",
    "DirectTranscriptionService",
    "EmbedServerTranscriptionService",
    "OcrModelContract",
    "OpenAIChatTranscriptionProvider",
    "PageRegion",
    "ProviderConfig",
    "TranscribedPage",
    "TranscriptionAttempt",
    "TranscriptionService",
    "bbox_to_pixels",
    "build_contract",
    "build_figure_captioner",
    "build_figure_context",
    "transcribe_concurrency",
    "crop_figures",
    "extract_grounded_page",
    "build_provider_config",
    "get_transcription_service",
    "pages_from_embed_server_response",
    "parse_cascade",
    "parse_grounded_page",
    "render_transcript",
    "strip_grounding_tokens",
]


def _global_num(v: Variables, key: str, default, cast):
    """Read an optional numeric env key without letting type_fn hit the default."""
    raw = v.environ(key, default=None)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    return cast(raw)


def parse_cascade(raw: str) -> list[str]:
    """Split the ordered cascade spec, preserving order and dropping blanks."""
    return [name.strip() for name in (raw or "").split(",") if name.strip()]


def build_provider_config(v: Variables, profile: str) -> ProviderConfig:
    """Resolve one cascade rung from its ``..._PROFILE_<NAME>_*`` keys.

    Raises ``ValueError`` when the profile has no URL: a rung that cannot be
    reached is a configuration error, and failing at startup beats discovering
    it as a silently short cascade during an ingest.
    """
    prefix = "%s%s_" % (MEMORYLAYER_TRANSCRIBE_PROFILE_PREFIX, profile.upper())

    def opt(field: str, default=None):
        """Raw read. Note ``Variables.environ`` applies ``type_fn`` to the
        DEFAULT as well as to a found value, so an unset optional int would
        raise -- convert here instead of delegating."""
        return v.environ(prefix + field, default=default)

    def opt_num(field: str, default, cast):
        raw = opt(field, None)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return default
        return cast(raw)

    base_url = (opt("URL", "") or "").strip()
    if not base_url:
        raise ValueError(
            "transcription profile %r has no %sURL" % (profile, prefix)
        )

    auth_kind = (opt("AUTH", "none") or "none").strip().lower()
    if auth_kind not in KNOWN_AUTH_KINDS:
        raise ValueError(
            "transcription profile %r has unknown auth %r (known: %s)"
            % (profile, auth_kind, ", ".join(KNOWN_AUTH_KINDS))
        )

    contract_name = (opt("CONTRACT", DEFAULT_MEMORYLAYER_TRANSCRIBE_CONTRACT) or "").strip()
    contract = build_contract(
        contract_name,
        ngram_size=opt_num("NGRAM_SIZE", None, int),
        window_size=opt_num("WINDOW_SIZE", None, int),
    )

    return ProviderConfig(
        name=profile,
        base_url=base_url,
        contract=contract,
        model=(opt("MODEL", "") or "").strip(),
        auth_kind=auth_kind,
        auth_token=opt("AUTH_TOKEN", "") or "",
        auth_key=opt("AUTH_KEY", "") or "",
        auth_secret=opt("AUTH_SECRET", "") or "",
        max_tokens=opt_num("MAX_TOKENS", DEFAULT_MAX_TOKENS, int),
        # Unset by default -> omitted from the request -> model's own default.
        temperature=opt_num("TEMPERATURE", None, float),
        timeout_sec=opt_num("TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC, float),
        cold_start_timeout_sec=opt_num("COLD_START_TIMEOUT_SEC", 0.0, float),
        overload_attempts=_global_num(
            v, MEMORYLAYER_TRANSCRIBE_OVERLOAD_ATTEMPTS, DEFAULT_OVERLOAD_ATTEMPTS, int,
        ),
        overload_backoff_sec=_global_num(
            v, MEMORYLAYER_TRANSCRIBE_OVERLOAD_BACKOFF_SEC, DEFAULT_OVERLOAD_BACKOFF_SEC, float,
        ),
    )


def transcribe_concurrency(v: Variables) -> int:
    """Pages (and captions) in flight at once, per worker process."""
    raw = v.environ(MEMORYLAYER_TRANSCRIBE_CONCURRENCY, default=None)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_TRANSCRIBE_CONCURRENCY
    return max(1, int(raw))


def build_cascade(v: Variables, logger: Logger) -> CascadeTranscriber:
    """Build the configured cascade, in order."""
    profiles = parse_cascade(
        v.environ(MEMORYLAYER_TRANSCRIBE_CASCADE, default=DEFAULT_MEMORYLAYER_TRANSCRIBE_CASCADE)
    )
    if not profiles:
        raise ValueError(
            "%s is empty; the direct transcription service needs at least one profile"
            % MEMORYLAYER_TRANSCRIBE_CASCADE
        )
    providers = [
        OpenAIChatTranscriptionProvider(build_provider_config(v, profile), logger)
        for profile in profiles
    ]
    concurrency = transcribe_concurrency(v)
    logger.info(
        "Direct transcription cascade: %s (concurrency=%d)",
        " -> ".join("%s(%s)" % (p.name, p.config.contract.name) for p in providers),
        concurrency,
    )
    return CascadeTranscriber(providers, logger, concurrency=concurrency)


# === Plugins ===


# noinspection PyAbstractClass
class TranscriptionServicePluginBase(Plugin):
    """Base plugin for the enterprise transcription service."""

    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_TRANSCRIPTION_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_TRANSCRIPTION_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(
            self, v, MEMORYLAYER_TRANSCRIPTION_SERVICE, self_attr="PROVIDER_NAME"
        )

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(
            MEMORYLAYER_TRANSCRIPTION_SERVICE,
            DEFAULT_MEMORYLAYER_TRANSCRIPTION_SERVICE,
        )


class EmbedServerTranscriptionServicePlugin(TranscriptionServicePluginBase):
    """Default: transcription through the embed-server REST client."""

    PROVIDER_NAME = "embed_server"

    def get_dependencies(self, v: Variables):
        from ..document import EXT_EMBED_SERVER_CLIENT

        return (EXT_EMBED_SERVER_CLIENT,)

    def initialize(self, v: Variables, logger: Logger):
        from ..document import EXT_EMBED_SERVER_CLIENT

        return EmbedServerTranscriptionService(
            get_extension(EXT_EMBED_SERVER_CLIENT, v), logger,
        )


def build_figure_captioner(v: Variables, logger: Logger) -> FigureCaptioner | None:
    """Build the figure captioner, or ``None`` when captioning is disabled.

    Routes through the standard LLM profiles mechanism. ``model`` is left unset
    by default so the selected profile's own default model applies — pinning a
    model name here would silently diverge from the profile's configuration.
    """
    if not v.environ(
        MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTIONS,
        default=DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTIONS,
        type_fn=ext_parse_bool,
    ):
        logger.info("Figure captioning disabled by %s=false", MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTIONS)
        return None

    from memorylayer_server.services.llm import EXT_LLM_SERVICE

    llm_service = get_extension(EXT_LLM_SERVICE, v)
    if llm_service is None:
        logger.info("Figure captioning skipped: no LLM service configured")
        return None

    def opt(key: str, default=None):
        return v.environ(key, default=default)

    profile = (
        opt(MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_PROFILE, DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_PROFILE)
        or DEFAULT_FIGURE_CAPTION_PROFILE
    ).strip()
    model = (opt(MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MODEL, DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MODEL) or "").strip()
    max_tokens = opt(MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MAX_TOKENS)
    context_chars = opt(MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_CONTEXT_CHARS)
    instruction = opt(MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_INSTRUCTION)
    reasoning_effort = opt(
        MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_REASONING_EFFORT,
        DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_REASONING_EFFORT,
    )

    logger.info(
        "Figure captioning enabled (profile=%s, model=%s)", profile, model or "<profile default>",
    )
    return FigureCaptioner(
        llm_service=llm_service,
        logger=logger,
        profile=profile,
        model=model or None,
        instruction=instruction or DEFAULT_FIGURE_CAPTION_INSTRUCTION,
        max_tokens=int(max_tokens) if max_tokens else DEFAULT_FIGURE_CAPTION_MAX_TOKENS,
        context_chars=int(context_chars) if context_chars else DEFAULT_FIGURE_CAPTION_CONTEXT_CHARS,
        concurrency=transcribe_concurrency(v),
        reasoning_effort=(reasoning_effort or "").strip() or None,
    )


class DirectTranscriptionServicePlugin(TranscriptionServicePluginBase):
    """In-process cascade against model endpoints — no embed-server."""

    PROVIDER_NAME = "direct"

    def get_dependencies(self, v: Variables):
        from memorylayer_server.services.llm import EXT_LLM_SERVICE

        # Figure captioning resolves the LLM service at build time; declaring the
        # dependency keeps plugin init ordered even though captioning is optional.
        return (EXT_LLM_SERVICE,)

    def initialize(self, v: Variables, logger: Logger):
        return DirectTranscriptionService(
            build_cascade(v, logger), logger, build_figure_captioner(v, logger),
        )


def get_transcription_service(v: Variables = None) -> TranscriptionService:
    return get_extension(EXT_TRANSCRIPTION_SERVICE, v)
