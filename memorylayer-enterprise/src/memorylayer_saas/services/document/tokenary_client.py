# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Proprietary tokenary inference-client adapter.

``TokenaryClient`` subclasses the OSS :class:`EmbedServerClient` so the
unchanged OSS embedding/reranker providers (which resolve their client via the
``EXT_EMBED_SERVER_CLIENT`` extension) keep working when an operator sets
``MEMORYLAYER_EMBED_SERVER_SERVICE=tokenary``. It overrides the high-level
concern methods to hit tokenary's *native* endpoints and translate the wire
shapes, while **returning the exact same Python structures the parent
``EmbedServerClient`` methods return** so every existing caller is unaffected.

Per-concern routing
-------------------
tokenary is one model per process, so each concern (single-vec, multi-vec,
score, visual-tokenize) targets its own instance URL. The parent transport is
single-base-url, so this client holds one lightweight per-concern child
``EmbedServerClient`` (sharing the parent transport/connect/close machinery) and
dispatches each method to the right child. Each per-concern URL defaults to the
shared embed-server URL, so a single-instance deployment still works.

DECISION (ii) — wire is base64-``.npy``
--------------------------------------
tokenary's ``/embed/multi`` returns each multi-vector as a base64-encoded NumPy
``.npy`` ``[N, proj_dim]`` F32 tensor, and ``/score`` accepts precomputed
vectors in the same base64-``.npy`` form. To preserve the existing provider
contracts (the OSS reranker passes the multivec returned by
``embed_texts_multivector`` straight into ``score_maxsim``), this client
**decodes ``.npy`` -> nested float lists at the multi-vec boundary and re-encodes
float lists -> ``.npy`` at the score boundary**. A future ``.npy``-native storage
optimization could keep the vectors in ``.npy`` form end-to-end and remove this
round-trip; that is intentionally out of scope here.
"""

from __future__ import annotations

import base64
import io
import os
from logging import Logger
from typing import Any

# Defensive backstop: cap each single-vector embed input to this many chars so a
# pathologically long text can never hard-500 the embed server with
# "Embedding input length ... exceeds max_model_len". The render path is the
# real fix; this only catches egregiously long inputs on any code path. Lenient
# by design (char-based, not token-exact). Overridable via env.
_EMBED_MAX_INPUT_CHARS_ENV = "MEMORYLAYER_EMBED_MAX_INPUT_CHARS"
_DEFAULT_EMBED_MAX_INPUT_CHARS = 24000

from memorylayer_server.services.document.embed_client import (
    TRANSPORT_HTTP,
    EmbedServerClient,
)

# tokenary native endpoint paths.
_PATH_TEXTVEC = "/v1/embeddings"
_PATH_MULTIVEC = "/embed/multi"
_PATH_SCORE = "/score"
_PATH_ENCODE = "/encode"
_PATH_CHAT = "/v1/chat/completions"


def _npy_b64_to_vectors(multivec_b64: str) -> list[list[float]]:
    """Decode a base64 ``.npy`` ``[N, proj_dim]`` tensor to nested float lists."""
    import numpy as np

    arr = np.load(io.BytesIO(base64.b64decode(multivec_b64)))
    return arr.astype("float32").tolist()


def _vectors_to_npy_b64(vectors: list[list[float]]) -> str:
    """Encode nested float-list vectors to a base64 ``.npy`` ``[N, proj_dim]`` F32."""
    import numpy as np

    arr = np.asarray(vectors, dtype="float32")
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TokenaryClient(EmbedServerClient):
    """``EmbedServerClient`` that targets tokenary's native endpoints.

    Inherits the parent transport (httpx / aether) via per-concern child
    clients. The constructor accepts a per-concern URL for each lane; any lane
    URL left empty falls back to ``base_url`` (the shared embed-server URL).
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        logger: Logger = None,
        *,
        transport: str = TRANSPORT_HTTP,
        aether_connection: Any | None = None,
        textvec_url: str = "",
        multivec_url: str = "",
        score_url: str = "",
        visualtok_url: str = "",
        embedding_dimensions: int | None = None,
        **parent_kwargs: Any,
    ):
        super().__init__(
            base_url,
            timeout,
            logger,
            transport=transport,
            aether_connection=aether_connection,
            **parent_kwargs,
        )
        self._embedding_dimensions = embedding_dimensions

        # One child client per concern, each pointed at its own tokenary
        # instance. Sharing transport/aether settings so connect/close and the
        # http-vs-aether selection behave identically to the parent.
        def _child(url: str) -> EmbedServerClient:
            return EmbedServerClient(
                url or base_url,
                timeout,
                logger,
                transport=transport,
                aether_connection=aether_connection,
                **parent_kwargs,
            )

        self._textvec = _child(textvec_url)
        self._multivec = _child(multivec_url)
        self._score = _child(score_url)
        self._visualtok = _child(visualtok_url)
        self._children = (self._textvec, self._multivec, self._score, self._visualtok)

    # ------------------------------------------------------------------
    # Lifecycle — connect/close every per-concern child.
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        await super().connect()
        for child in self._children:
            await child.connect()

    async def close(self) -> None:
        for child in self._children:
            await child.close()
        await super().close()

    # ------------------------------------------------------------------
    # Single-vector text -> tokenary /v1/embeddings (OpenAI shape matches).
    # ------------------------------------------------------------------

    def _cap_embed_inputs(self, texts: list[str]) -> list[str]:
        """Defensively truncate over-long single-vector embed inputs.

        Caps each text at ``MEMORYLAYER_EMBED_MAX_INPUT_CHARS`` characters
        (default ``24000``) so an egregiously long input can never hard-500 the
        embed server. Lenient by design: the render-to-page-images path is the
        real fix; this is only a last-resort backstop. Logs a warning (original
        vs capped length) whenever it actually truncates.
        """
        try:
            max_chars = int(os.environ.get(
                _EMBED_MAX_INPUT_CHARS_ENV, _DEFAULT_EMBED_MAX_INPUT_CHARS,
            ))
        except (TypeError, ValueError):
            max_chars = _DEFAULT_EMBED_MAX_INPUT_CHARS
        if max_chars <= 0:
            return texts

        capped: list[str] = []
        for text in texts:
            if len(text) > max_chars:
                self.logger.warning(
                    "tokenary embed_texts: truncating over-long input from %d "
                    "to %d chars (%s)",
                    len(text), max_chars, _EMBED_MAX_INPUT_CHARS_ENV,
                )
                capped.append(text[:max_chars])
            else:
                capped.append(text)
        return capped

    async def embed_texts(self, texts: list[str], *, dimensions: int | None = None) -> list[list[float]]:
        # ``dimensions`` matches the base ``EmbedServerClient.embed_texts`` /
        # embed-provider contract (the provider forwards its configured request
        # dimensions on every call). The override previously omitted it, so any
        # single-vector embed routed through this tokenary client (e.g. embedding
        # a decomposed fact's text) raised ``TypeError: unexpected keyword
        # argument 'dimensions'``. Honor an explicit value; otherwise fall back
        # to the client's configured embedding dimensions (unchanged behavior).
        texts = self._cap_embed_inputs(texts)
        payload: dict = {"input": texts}
        dims = dimensions if dimensions is not None else self._embedding_dimensions
        if dims is not None:
            payload["dimensions"] = dims
        self.logger.debug("tokenary embed_texts: %d texts", len(texts))
        data = await self._textvec.request_json("POST", _PATH_TEXTVEC, payload)
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    # ------------------------------------------------------------------
    # Multi-vector text -> tokenary /embed/multi (base64 .npy decode).
    # ------------------------------------------------------------------

    async def embed_texts_multivector(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[dict]:
        # tokenary has no input_type; query/document is a model-side concern.
        del input_type
        payload = {"input": texts, "pool_factor": 1}
        self.logger.debug("tokenary embed_texts_multivector: %d texts", len(texts))
        data = await self._multivec.request_json("POST", _PATH_MULTIVEC, payload)
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [self._multivec_item_to_result(item) for item in sorted_data]

    async def embed_images_multivector(self, images_b64: list[str]) -> list[dict]:
        # Each image becomes its own one-message group carrying an image_url
        # data-URI part (tokenary MultiEmbedInput::Messages, one group per doc).
        message_groups = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _as_data_uri(b64)},
                        }
                    ],
                }
            ]
            for b64 in images_b64
        ]
        payload = {"input": {"messages": message_groups}, "pool_factor": 1}
        self.logger.debug("tokenary embed_images_multivector: %d images", len(images_b64))
        data = await self._multivec.request_json("POST", _PATH_MULTIVEC, payload)
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [self._multivec_item_to_result(item) for item in sorted_data]

    @staticmethod
    def _multivec_item_to_result(item: dict) -> dict:
        """Map a tokenary ``/embed/multi`` item to the parent's result dict.

        Parent contract: ``{"vectors": [[...], ...], "num_vectors": int}``.
        """
        vectors = _npy_b64_to_vectors(item["multivec"])
        return {"vectors": vectors, "num_vectors": item.get("num_tokens", len(vectors))}

    # ------------------------------------------------------------------
    # MaxSim score -> tokenary /score (Precomputed{multivec} form).
    # ------------------------------------------------------------------

    async def score_maxsim(
        self,
        query_vectors: list[list[float]],
        document_vectors: list[list[list[float]]],
    ) -> list[dict]:
        payload = {
            "query": {"multivec": _vectors_to_npy_b64(query_vectors)},
            "documents": [
                {"multivec": _vectors_to_npy_b64(doc)} for doc in document_vectors
            ],
            "pool_factor": 1,
        }
        self.logger.debug(
            "tokenary score_maxsim: query (%d vectors) vs %d documents",
            len(query_vectors),
            len(document_vectors),
        )
        data = await self._score.request_json("POST", _PATH_SCORE, payload)
        # tokenary returns scores as a flat list (one per document, in order);
        # the parent contract / OSS reranker expects [{"index", "score"}] dicts.
        return [
            {"index": i, "score": float(s)} for i, s in enumerate(data["scores"])
        ]

    # ------------------------------------------------------------------
    # Visual tokenizer -> tokenary /encode (single input per request).
    # ------------------------------------------------------------------

    async def encode(self, image_b64: str, *, apply_chat_template: bool = True) -> dict:
        """Encode a single page image via tokenary ``/encode``.

        Returns the raw tokenary ``EncodeResponse`` dict (``prompt_embeds``,
        ``image_embeds`` (base64 ``.npy``), ``image_grid_thw`` (plain JSON list
        ``[[T,H,W], ...]``), ``num_image_tokens``, ``hidden_size``). The
        ``visual_tokenize`` adapter normalizes this to the downstream
        image-embeds consumer shape.
        """
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _as_data_uri(image_b64)},
                        }
                    ],
                }
            ],
            "apply_chat_template": apply_chat_template,
        }
        return await self._visualtok.request_json("POST", _PATH_ENCODE, payload)

    # ------------------------------------------------------------------
    # Chat -> tokenary /v1/chat/completions (flatten image_embeds blocks).
    # ------------------------------------------------------------------

    async def chat_completions(self, payload: dict, *, stream: bool = False):
        flattened = _flatten_image_embeds_payload(payload)
        return await super().chat_completions(flattened, stream=stream)


def _as_data_uri(image_b64: str) -> str:
    """Wrap a bare base64 image as a ``data:`` URI (idempotent)."""
    if image_b64.startswith("data:"):
        return image_b64
    return f"data:image/png;base64,{image_b64}"


def _flatten_image_embeds_payload(payload: dict) -> dict:
    """Return a copy of ``payload`` with nested ``image_embeds`` blocks flattened.

    embed-server / memorylayer build the block nested::

        {"type": "image_embeds",
         "image_embeds": {"image_embeds": "<b64>", "image_grid_thw": <b64|list>}}

    tokenary's chat ``image_embeds`` block is FLAT::

        {"type": "image_embeds", "image_embeds": "<b64>", "image_grid_thw": [...]}

    Non-image_embeds blocks and already-flat blocks pass through unchanged.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload

    new_messages = []
    changed = False
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        new_content = []
        for block in content:
            flat = _flatten_image_embeds_block(block)
            if flat is not block:
                changed = True
            new_content.append(flat)
        new_messages.append({**msg, "content": new_content})

    if not changed:
        return payload
    return {**payload, "messages": new_messages}


def _flatten_image_embeds_block(block: Any) -> Any:
    """Flatten one nested ``image_embeds`` block; pass everything else through.

    The nested embed-server / memorylayer block carries ``image_grid_thw`` as
    a base64 ``.npy`` (or torch.save) blob; tokenary's flat block wants it as a
    plain JSON list. We decode a base64-``.npy`` grid to a list when possible and
    otherwise pass the value through (a plain list is already fine).
    """
    if not isinstance(block, dict) or block.get("type") != "image_embeds":
        return block
    embeds = block.get("image_embeds")
    if not isinstance(embeds, dict):
        return block  # already flat (string payload) or unexpected — leave as-is
    flat: dict = {"type": "image_embeds", "image_embeds": embeds.get("image_embeds")}
    grid = embeds.get("image_grid_thw")
    if grid is not None:
        flat["image_grid_thw"] = _grid_to_list(grid)
    return flat


def _grid_to_list(grid: Any) -> Any:
    """Coerce an ``image_grid_thw`` value to a plain list.

    Accepts a list (passthrough) or a base64 ``.npy`` string (decoded). On any
    decode failure the original value is returned unchanged so a torch.save or
    otherwise-unexpected blob still reaches tokenary verbatim.
    """
    if isinstance(grid, list):
        return grid
    if isinstance(grid, str):
        try:
            import numpy as np

            return np.load(io.BytesIO(base64.b64decode(grid))).tolist()
        except Exception:  # noqa: BLE001 — best-effort decode; fall through
            return grid
    return grid
