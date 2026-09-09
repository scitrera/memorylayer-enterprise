# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Grounded document chat over stored page image embeds.

`POST /v1/documents/chat`: answer a question grounded in document pages, by
injecting each page's PRECOMPUTED image-embeds (raw vision-tower output + the
``image_grid_thw`` grid) as vLLM ``image_embeds`` content parts into a
single-turn completion on the inference LLM. vLLM computes the correct 3-D
M-RoPE positions for these image tokens (the old flat ``prompt_embeds`` path
gave them 1-D positions and is gone).

Design (see discussion):
- ``context`` is an ORDERED list of typed items, each one of the page-selection
  strategies (explicit pages / whole document / maxsim search). A single-item
  list is the "pick one strategy" case; multiple items compose. This is the
  abstraction that a future stateful "grounded context" object would persist.
- ``input`` is the user's question: a plain string (preferred) or OpenAI-style
  content parts (text [+ image_url]). This is a single-turn completion
  (context + Q -> A), NOT a chat history.
- ``model`` is the real model name (e.g. ``Qwen/Qwen3.6-27B-FP8``); its slug
  must match the slug under which each page's image-embeds were stored
  (``page.visual_tokens[slug]``) or the embeds are meaningless. The serving
  profile name stays a config detail (optionally an alias).
- Multi-workspace: defaults to the caller's workspace; cross-workspace is
  allowed when a maxsim item lists workspaces or explicit pages/docs span them,
  gated by per-workspace OBO authorization.
- Response returns the resolved ``context`` descriptor so a follow-up turn can
  resend the SAME page set (now as explicit ids) + the prior answer + a new
  question. Identical leading image-embeds blocks => vLLM prefix-cache hits.

STATUS: feature-flagged by MEMORYLAYER_DOCUMENT_CHAT_ENABLED. The route
requires a prompt-embeds-capable inference target and repairs missing per-page
image-embeds before generation when rendered page images are available.
"""
from __future__ import annotations

import logging
import secrets
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from memorylayer_server.api.v1.deps import get_auth_service, get_authz_service
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.authentication import AuthenticationError, AuthenticationService
from memorylayer_server.services.authorization import AuthorizationService
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables, ext_parse_bool, get_extension

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_ENABLED,
    MEMORYLAYER_DOCUMENT_CHAT_ENABLED,
)
from memorylayer_saas.services.document import (
    EXT_BLOB_STORAGE_SERVICE,
    EXT_EMBED_SERVER_CLIENT,
)
from memorylayer_saas.services.document.chat_embed_repair import (
    missing_image_embed_pages,
    repair_missing_image_embeds,
)
from memorylayer_saas.services.document.image_embed import (
    build_image_embeds_content_blocks,
    page_metadata_text,
)
from memorylayer_saas.services.document.inference_client import (
    default_inference_model as _default_inference_model,
)
from memorylayer_saas.services.document.inference_client import (
    get_inference_client as _get_inference_client,
)
from memorylayer_saas.services.document.inference_client import (
    slugify_model as _slugify_model,
)

router = APIRouter(prefix="/v1/documents", tags=["documents"])

_RESERVED_GENERATION_EXTRA_KEYS = frozenset({
    "messages",
    "model",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
})


# ---------------------------------------------------------------------------
# Request models — context spec (ordered, typed items)
# ---------------------------------------------------------------------------

class PagesItem(BaseModel):
    """Explicit page ids (may span workspaces if the caller is authorized)."""
    type: Literal["pages"] = "pages"
    page_ids: list[str] = Field(..., min_length=1)


class DocumentItem(BaseModel):
    """All (or a range of) pages of one document, in page order."""
    type: Literal["document"] = "document"
    document_id: str
    workspace_id: str | None = Field(None, description="Defaults to the caller's workspace")
    page_range: tuple[int, int] | None = Field(None, description="[start, end) zero-indexed")


class MaxSimItem(BaseModel):
    """Pages selected by ColPali MaxSim visual similarity to a query."""
    type: Literal["maxsim"] = "maxsim"
    query: str
    top_k: int = Field(5, ge=1, le=100)
    workspace_ids: list[str] | None = Field(None, description="Defaults to the caller's workspace")
    doc_ids: list[str] | None = None
    min_score: float | None = None


ContextItem = Annotated[
    PagesItem | DocumentItem | MaxSimItem,
    Field(discriminator="type"),
]


class GenerationParams(BaseModel):
    max_tokens: int = Field(1024, ge=1)
    temperature: float = Field(0.2, ge=0.0)
    top_p: float | None = None
    # passthrough for any other OpenAI sampling params
    extra: dict | None = None


class DocumentChatRequest(BaseModel):
    # NOTE: this ``context_id`` is UNRELATED to memories.context_id / the
    # contexts table. It is an ephemeral grounded-chat session id (see
    # _new_context_id + the in-memory _GROUNDED_CONTEXTS dict) used to resume a
    # document-chat conversation. Same name, different concept — don't conflate.
    #
    # Phase-2 stateful: exactly one of ``context`` (new conversation) or
    # ``context_id`` (resume a server-persisted grounded context) must be set.
    context: list[ContextItem] | None = Field(None, min_length=1)
    context_id: str | None = Field(
        None,
        description="Resume a previously-created grounded context (from a prior "
                    "response). Mutually exclusive with ``context``.",
    )
    # Question: plain text (preferred) or OpenAI content parts (text [+ image_url]).
    input: str | list[dict]
    model: str | None = Field(
        None,
        description="Real model name; its slug must match the stored embeds. "
                    "Omit to use the default inference profile. Ignored on resume "
                    "(the stored context's model is reused).",
    )
    stream: bool = False
    generation: GenerationParams = Field(default_factory=GenerationParams)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ResolvedPage(BaseModel):
    page_id: str
    document_id: str
    workspace_id: str
    page_no: int
    image_embed_tokens: int = 0
    included: bool = True
    skip_reason: str | None = None


class ChatContextDescriptor(BaseModel):
    """Echo of what was actually fed to the model — resend for a follow-up turn."""
    model: str
    model_slug: str
    pages: list[ResolvedPage]
    total_image_embed_tokens: int
    workspaces: list[str]


class DocumentChatResponse(BaseModel):
    completion: dict  # OpenAI-shape chat completion (passthrough)
    context: ChatContextDescriptor
    context_id: str  # server-persisted grounded context; resend to continue


# ---------------------------------------------------------------------------
# Grounded-context store
# ---------------------------------------------------------------------------
# Process-lifetime, in-memory store for grounded contexts. NOT durable: entries
# live only for the lifetime of this process and are not shared across workers.
# This is fine for Phase-1 stateful / single-process dev; a durable backing
# store (e.g. Redis / Postgres) is a separate follow-up.
_GROUNDED_CONTEXTS: dict[str, dict] = {}


def _new_context_id() -> str:
    # secrets is used deliberately (the runtime forbids time/random-based ids).
    return f"gctx_{secrets.token_hex(6)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _authorize_workspaces(ctx, authz_service, workspaces: set[str]) -> None:
    """OBO-authorize read access to every workspace the request touches."""
    for ws in workspaces:
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ws)


async def _resolve_pages(item, *, ctx, storage, embed_client, logger):
    """Resolve one context item to an ordered list of DocumentPage objects.

    Returns (pages, workspaces_touched). Page objects must carry .visual_tokens.
    """
    if isinstance(item, PagesItem):
        # Returns pages in requested order, each carrying workspace_id so we can
        # authorize per workspace below (pages may span workspaces).
        pages = await storage.get_pages_by_ids(item.page_ids)
        return pages, {p.workspace_id for p in pages}

    if isinstance(item, DocumentItem):
        ws = item.workspace_id or ctx.workspace_id
        pages = await storage.get_pages(item.document_id, ws)
        if item.page_range:
            lo, hi = item.page_range
            pages = [p for p in pages if lo <= p.page_no < hi]
        pages.sort(key=lambda p: p.page_no)
        return pages, {ws}

    if isinstance(item, MaxSimItem):
        workspaces = set(item.workspace_ids) if item.workspace_ids else {ctx.workspace_id}
        # Embed the query once, then MaxSim-search each workspace. Do NOT close()
        # here: this is the shared, process-lifetime EXT_EMBED_SERVER_CLIENT
        # singleton (also used for inference in the colocated path). Closing it
        # would break the subsequent generation call. connect() is idempotent.
        await embed_client.connect()
        mv = await embed_client.embed_texts_multivector([item.query])
        qvec = mv[0]["vectors"]
        pages: list = []
        for ws in workspaces:
            hits = await storage.search_pages_by_maxsim(
                workspace_id=ws, query_multivector=qvec,
                limit=item.top_k, doc_ids=item.doc_ids,
            )
            for page, score in hits:
                if item.min_score is None or score >= item.min_score:
                    pages.append(page)
        return pages, workspaces

    raise HTTPException(status_code=400, detail=f"unknown context item type: {item!r}")


async def _resolve_filename(storage, pages, ctx) -> str | None:
    """Best-effort filename for page markers when all pages share one document.

    Used only to enrich the ``[Page N of <filename>]`` interleaved markers. When
    the context spans multiple documents (or the lookup fails), returns None and
    the marker falls back to ``[Page N]``.
    """
    doc_ids = {p.document_id for p in pages}
    if len(doc_ids) != 1:
        return None
    page = pages[0]
    try:
        doc = await storage.get_document(page.document_id, page.workspace_id)
    except Exception:  # noqa: BLE001 - cosmetic marker only, never fatal
        return None
    return getattr(doc, "filename", None) if doc else None


def _normalize_question_parts(input_) -> list[dict]:
    """User question -> OpenAI content parts appended after the page blocks."""
    if isinstance(input_, str):
        return [{"type": "text", "text": input_}]
    return list(input_)  # already content parts


def _apply_generation_extra(payload: dict, extra: dict | None) -> None:
    """Apply caller passthrough options without overriding grounded fields."""
    if not extra:
        return
    reserved = sorted(set(extra) & _RESERVED_GENERATION_EXTRA_KEYS)
    if reserved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="generation.extra may not override reserved field(s): " + ", ".join(reserved),
        )
    payload.update(extra)


async def _stream_with_context(descriptor: ChatContextDescriptor, context_id: str, upstream):
    """SSE generator: emit the resolved-context descriptor first, then proxy.

    The client receives a leading ``event: context`` carrying the resolved page
    set plus the ``context_id`` (for follow-up turns), followed by the upstream
    OpenAI ``data:`` chunks verbatim.
    """
    import json as _json

    payload = {**descriptor.model_dump(), "context_id": context_id}
    yield f"event: context\ndata: {_json.dumps(payload)}\n\n".encode()
    async for chunk in upstream:
        yield chunk


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=DocumentChatResponse)
async def document_chat(
    http_request: Request,
    request: DocumentChatRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
):
    try:
        # Exactly one of (context, context_id) must be provided.
        if bool(request.context) == bool(request.context_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide exactly one of 'context' (new conversation) or "
                       "'context_id' (resume).",
            )

        ctx = await auth_service.build_context(http_request)

        storage = get_extension(EXT_STORAGE_BACKEND, v)
        blob_storage = get_extension(EXT_BLOB_STORAGE_SERVICE, v)
        embed_client = get_extension(EXT_EMBED_SERVER_CLIENT, v)  # preprocessing (maxsim)
        inference_client = await _get_inference_client(v, logger)

        # ``entry`` is the persisted grounded context on resume, else None.
        entry = None
        if request.context_id:
            entry = _GROUNDED_CONTEXTS.get(request.context_id)
            if entry is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Unknown context_id {request.context_id!r}.",
                )
            model = entry["model"]
            model_slug = entry["model_slug"]
        else:
            model = request.model or _default_inference_model(v)
            model_slug = _slugify_model(model)

        # 1) Resolve pages -> ordered pages, collecting workspaces.
        ordered_pages: list = []
        workspaces: set[str] = set()
        if entry is not None:
            # Resume: re-resolve the stored page ids, preserving stored order.
            ordered_pages = await storage.get_pages_by_ids(entry["page_ids"])
            workspaces = set(entry["workspaces"])
        else:
            for item in request.context:
                pages, ws = await _resolve_pages(
                    item, ctx=ctx, storage=storage, embed_client=embed_client, logger=logger,
                )
                ordered_pages.extend(pages)
                workspaces |= ws

            # Dedup by page id, preserving first-occurrence order: context items may
            # overlap (e.g. an explicit page also surfaced by a maxsim item), and we
            # must not feed the same page's image_embeds to the model twice.
            _seen: set[str] = set()
            ordered_pages = [
                p for p in ordered_pages if not (p.id in _seen or _seen.add(p.id))
            ]

        # 2) OBO-authorize EVERY workspace touched (default scope = caller's ws).
        #    On resume this re-checks the stored workspaces every turn.
        await _authorize_workspaces(ctx, authz_service, workspaces or {ctx.workspace_id})

        # 3) Ensure every resolved page has image-embeds for this model slug.
        #    Missing rendered pages are repaired synchronously via the same
        #    precompute path used by ingestion. Remaining gaps are explicit 409s;
        #    document chat never sends a partial page context.
        missing_before = missing_image_embed_pages(ordered_pages, model_slug)
        repaired_pages = 0
        if missing_before:
            repaired_pages = await repair_missing_image_embeds(
                embed_client=embed_client,
                blob_storage=blob_storage,
                storage=storage,
                pages=ordered_pages,
                model_slug=model_slug,
                logger=logger,
            )

        missing_after = missing_image_embed_pages(ordered_pages, model_slug)
        if missing_after:
            missing_ids = [page.id for page in missing_after]
            logger.info(
                "document_chat: %d/%d pages still lack image-embeds for %s after repair",
                len(missing_after), len(ordered_pages), model_slug,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "Document chat requires image-embeds for every selected page.",
                    "model": model,
                    "model_slug": model_slug,
                    "missing_page_ids": missing_ids,
                    "repair_attempted": bool(missing_before),
                    "repaired_pages": repaired_pages,
                },
            )

        included_pages = list(ordered_pages)
        if not included_pages:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "Document chat requires at least one resolved document page.",
                    "model": model,
                    "model_slug": model_slug,
                },
            )

        blocks = await build_image_embeds_content_blocks(
            blob_storage=blob_storage, pages=included_pages, model_slug=model_slug, v=v,
        )
        if len(blocks) != len(included_pages):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "Stored image-embed references could not be loaded for every selected page.",
                    "model": model,
                    "model_slug": model_slug,
                },
            )

        # Build the resolved-context descriptor (for the response + follow-ups).
        resolved: list[ResolvedPage] = []
        for page in ordered_pages:
            ref = (page.visual_tokens or {}).get(model_slug)
            resolved.append(ResolvedPage(
                page_id=page.id, document_id=page.document_id,
                workspace_id=page.workspace_id, page_no=page.page_no,
                image_embed_tokens=(ref or {}).get("num_image_tokens", 0),
                included=True,
                skip_reason=None,
            ))
        total_tokens = sum(p.image_embed_tokens for p in resolved)

        # 4) Build the INTERLEAVED leading content in resolved-page order:
        #      [Page N marker text][page N image_embeds part] ...
        #    This is byte-identical across turns of the same context (the page set
        #    and order are stored), so vLLM prefix-caches it. The serving model
        #    applies the chat template ONCE around the full messages array.
        filename = await _resolve_filename(storage, included_pages, ctx)
        interleaved: list[dict] = []
        for page, block in zip(included_pages, blocks):
            interleaved.append({"type": "text", "text": page_metadata_text(page, filename)})
            interleaved.append(block)

        # 5) Assemble the (multi-turn) messages array.
        history = entry["history"] if entry is not None else []
        if entry is None or not history:
            # New conversation: one user turn = [grounded blocks + the question].
            messages = [{"role": "user", "content": [
                *interleaved, *_normalize_question_parts(request.input),
            ]}]
        else:
            # Resume: replay history (grounded blocks lead the FIRST user turn so
            # the prefix is byte-identical), then append the new question.
            messages = [{"role": "user", "content": [
                *interleaved, *_normalize_question_parts(history[0]["question"]),
            ]}]
            for i, turn in enumerate(history):
                messages.append({"role": "assistant", "content": turn["answer"]})
                next_q = history[i + 1]["question"] if i + 1 < len(history) else request.input
                messages.append({"role": "user", "content": _normalize_question_parts(next_q)})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.generation.max_tokens,
            "temperature": request.generation.temperature,
        }
        if request.generation.top_p is not None:
            payload["top_p"] = request.generation.top_p
        _apply_generation_extra(payload, request.generation.extra)

        descriptor = ChatContextDescriptor(
            model=model, model_slug=model_slug, pages=resolved,
            total_image_embed_tokens=total_tokens, workspaces=sorted(workspaces),
        )

        # The context id is the resumed one, or a freshly minted one (the entry is
        # created/updated below after a successful generation).
        context_id = request.context_id or _new_context_id()

        # 6) Forward to the inference LLM (prompt-embeds-enabled).
        if request.stream:
            # Streaming limitation: we cannot reconstruct the assistant answer text
            # from the proxied SSE deltas here, so streamed turns are NOT appended
            # to history (capturing streamed deltas is out of scope). We still
            # ensure an entry exists and emit the context_id so the client can keep
            # the conversation going via the non-streaming (supported multi-turn)
            # path. For a brand-new streamed conversation we persist the page set /
            # model with empty history.
            if entry is None:
                _GROUNDED_CONTEXTS[context_id] = {
                    "model": model, "model_slug": model_slug,
                    "page_ids": [p.id for p in included_pages],
                    "workspaces": sorted(workspaces), "history": [],
                }
            stream_iter = await inference_client.chat_completions(payload, stream=True)
            return StreamingResponse(
                _stream_with_context(descriptor, context_id, stream_iter),
                media_type="text/event-stream",
            )

        completion = await inference_client.chat_completions(payload, stream=False)

        # Extract the assistant text for history (guard for a malformed shape).
        answer = ""
        try:
            answer = completion["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning("document_chat: completion missing assistant content; "
                           "storing empty answer in history")

        if entry is None:
            _GROUNDED_CONTEXTS[context_id] = {
                "model": model, "model_slug": model_slug,
                "page_ids": [p.id for p in included_pages],
                "workspaces": sorted(workspaces),
                "history": [{"question": request.input, "answer": answer}],
            }
        else:
            entry["history"].append({"question": request.input, "answer": answer})

        return JSONResponse(content=DocumentChatResponse(
            completion=completion, context=descriptor, context_id=context_id,
        ).model_dump())

    except AuthenticationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("document_chat failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="document chat failed")


class DocumentChatRoutePlugin(Plugin):
    """Register /v1/documents/chat when MEMORYLAYER_DOCUMENT_CHAT_ENABLED is true."""

    def extension_point_name(self, v: Variables) -> str:
        from memorylayer_server.api import EXT_MULTI_API_ROUTERS
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        # Multi-extension router plugins must return False here (matches
        # DocumentsAPIPlugin / VisualTokenizerAPIPlugin). Returning True
        # double-registers the router under the single-extension slot and
        # evicts sibling routers that share the slot (e.g. the base
        # /v1/documents router).
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        # Gate the whole plugin on the feature flag here, not in initialize().
        # When this returns False the framework skips the plugin entirely
        # (initialize() never runs, nothing is stored in the EXT_MULTI_API_ROUTERS
        # registry). Gating by returning None from initialize() instead leaves a
        # None value in the registry, which RoutesPlugin.include_router() chokes
        # on ('NoneType' has no attribute 'routes').
        return v.environ(
            MEMORYLAYER_DOCUMENT_CHAT_ENABLED,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_ENABLED,
            type_fn=ext_parse_bool,
        )

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        logger.info("Registering /v1/documents/chat (grounded prompt-embed chat)")
        return router
