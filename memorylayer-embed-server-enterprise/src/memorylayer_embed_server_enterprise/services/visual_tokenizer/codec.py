# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Wire codec for image-embed + grid tensors.

The visual tokenizer ships two tensors per page over the wire:

* the **vision-tower image embeds** — the projected per-image
  ``pooler_output`` of shape ``[num_image_tokens, hidden_dim]`` — fed to a
  vLLM chat profile as an ``image_embeds`` content part (this is what carries
  M-RoPE-able 2-D positions, unlike the flat ``prompt_embeds`` path).
* the **image grid** ``torch.tensor([t, h, w], dtype=int64)`` that vLLM needs
  alongside the embeds to reconstruct the spatial layout.

Both are serialized as base64-encoded ``torch.save`` payloads loaded with
``weights_only=True``. The image-embeds payload is additionally zstd-compressed
(level 1) to cut the wire/storage footprint; the tiny grid tensor is left raw.
:func:`b64_to_tensor` auto-detects the zstd magic so it reads either form.
"""

from __future__ import annotations

import base64
from io import BytesIO

import torch

# zstd frame magic number (little-endian 0xFD2FB528); see RFC 8878.
ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'


def tensor_to_b64(tensor: torch.Tensor, *, compress: bool = False) -> str:
    """Serialize a tensor to a base64 ``torch.save`` payload.

    The tensor is moved to CPU and made contiguous first. When ``compress`` is
    set the ``torch.save`` bytes are wrapped with zstd level 1 before base64
    encoding (used for the image-embeds tensor; the tiny grid tensor is sent
    raw). No shape/dtype validation is performed — this codec serializes both
    the 2-D float image embeds and the 1-D int64 grid tensor.
    """
    buf = BytesIO()
    torch.save(tensor.detach().to("cpu").contiguous(), buf)
    payload = buf.getvalue()
    if compress:
        import zstandard

        payload = zstandard.ZstdCompressor(level=1).compress(payload)
    return base64.b64encode(payload).decode("ascii")


def b64_to_tensor(encoded: str) -> torch.Tensor:
    """Inverse of :func:`tensor_to_b64` (used by tests and the consumer side).

    Auto-detects zstd-compressed payloads via the frame magic, so it reads
    both compressed (image embeds) and raw (grid) tensors.
    """
    raw = base64.b64decode(encoded, validate=True)
    if raw[:4] == ZSTD_MAGIC:
        import zstandard

        raw = zstandard.ZstdDecompressor().decompress(raw)
    tensor = torch.load(BytesIO(raw), weights_only=True, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("decoded payload is not a torch.Tensor")
    return tensor
