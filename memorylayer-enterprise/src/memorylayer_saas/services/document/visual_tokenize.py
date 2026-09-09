"""Enterprise visual-tokenizer client helper.

Owns the ``/v1/visual-tokenize`` contract, which is an enterprise-only
embed-server endpoint (the OSS ``EmbedServerClient`` is a universal transport
and intentionally does not hard-code this endpoint). Keeping the contract
knowledge here lets the OSS client stay universal while enterprise code reaches
the endpoint through the client's public request seam.

tokenary path
-------------
tokenary exposes a synchronous, single-input ``POST /encode`` instead of the
embed-server's batched ``/v1/visual-tokenize`` (with its zstd / async-job /
server-side cache layer). When the active client is a ``TokenaryClient`` this
helper calls ``/encode`` once per image and normalizes the responses into the
SAME ``{"results": [...], "model": ...}`` shape ``precompute_and_store_image_embeds``
consumes — synchronous-only, with caching owned server-side by memorylayer (which
decides when to call /encode at all). The embed-server path is unchanged.
"""

from __future__ import annotations

import base64
import io


async def visual_tokenize(
    client,
    images_b64: list[str],
    metadata: list[dict] | None = None,
    *,
    return_tensors: bool = True,
    force_recompute: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Precompute per-page image-embeds via the active visual tokenizer.

    Posts page images plus optional stable per-page metadata (filename,
    page_no, source) to the enterprise-only ``/v1/visual-tokenize`` endpoint
    using the OSS client's public request seam (:meth:`request_json`), OR — when
    the client is a ``TokenaryClient`` — to tokenary's synchronous ``/encode``
    (one call per image), normalized to the same return shape.

    Args:
        client: The (OSS) ``EmbedServerClient`` providing the universal
            transport and the public ``request_json`` seam, or a
            ``TokenaryClient`` (handled via its ``encode`` method).
        images_b64: Base64-encoded page images.
        metadata: Optional per-page provenance metadata, aligned to
            ``images_b64``.
        return_tensors: Whether the server should return tensor payloads.
        force_recompute: Bypass any server-side cache when set.
        batch_size: Optional server-side batch size override.

    Returns:
        The raw decoded JSON for ``precompute_and_store_image_embeds`` to
        persist, e.g. ``{"results": [{image_embeds_b64, image_grid_thw_b64,
        num_image_tokens, ...}], "stats": ..., "model": ...}``.
    """
    # tokenary: synchronous single-input /encode, one call per image. Imported
    # lazily to avoid a hard dependency for the embed-server path.
    from .tokenary_client import TokenaryClient

    if isinstance(client, TokenaryClient):
        return await _tokenary_visual_tokenize(client, images_b64)

    payload: dict = {"images": images_b64, "return_tensors": return_tensors}
    if metadata is not None:
        payload["metadata"] = metadata
    if force_recompute:
        payload["force_recompute"] = True
    if batch_size is not None:
        payload["batch_size"] = batch_size

    return await client.request_json("POST", "/v1/visual-tokenize", payload)


async def _tokenary_visual_tokenize(client, images_b64: list[str]) -> dict:
    """Adapt tokenary ``/encode`` (single input) to the visual-tokenize shape.

    tokenary returns ``image_embeds`` as a base64 ``.npy`` ``[N, hidden]`` tensor
    and ``image_grid_thw`` as a plain JSON list ``[[T,H,W], ...]``. The downstream
    consumer stores ``image_embeds_b64`` (decoded -> bytes verbatim) and
    ``image_grid_thw_b64`` (decoded -> bytes verbatim); the chat read path
    re-base64s the stored embeds bytes (tokenary's chat ``image_embeds`` block
    decodes base64 ``.npy``). We therefore store:
      * ``image_embeds_b64``: the raw ``.npy`` bytes (the base64 tokenary returned).
      * ``image_grid_thw_b64``: base64 of a ``.npy``-encoded grid array, so the
        chat-flatten can recover the list (see TokenaryClient flatten).
      * ``image_grid_thw``: the plain JSON list (persisted on the page entry).
    """
    import numpy as np

    results: list[dict] = []
    model: str | None = None
    for b64 in images_b64:
        try:
            resp = await client.encode(b64)
            image_embeds_b64 = resp.get("image_embeds")
            grid_list = resp.get("image_grid_thw")
            if not image_embeds_b64 or grid_list is None:
                results.append({"success": False, "error": "tokenary /encode returned no image_embeds"})
                continue

            grid_arr = np.asarray(grid_list, dtype="int64")
            buf = io.BytesIO()
            np.save(buf, grid_arr)
            grid_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            results.append(
                {
                    "success": True,
                    "embed_kind": "image_embeds",
                    "image_embeds_b64": image_embeds_b64,
                    "image_grid_thw_b64": grid_b64,
                    "image_grid_thw": grid_list,
                    "num_image_tokens": resp.get("num_image_tokens", 0),
                    "hidden_dim": resp.get("hidden_size", 0),
                }
            )
            if model is None:
                model = resp.get("model")
        except Exception as exc:  # noqa: BLE001 — per-page failure is non-fatal upstream
            results.append({"success": False, "error": f"tokenary /encode failed: {exc}"})

    out: dict = {"results": results}
    if model is not None:
        out["model"] = model
    return out
