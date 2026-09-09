"""Qwen3.5/3.6 vision-tower image-embed provider.

Produces, per page, the **projected vision-tower image embeds** — the model's
``get_image_features`` ``pooler_output`` of shape
``[num_image_tokens, hidden_dim]`` where
``num_image_tokens == prod(image_grid_thw) // spatial_merge_size**2``. These
embeds, shipped alongside ``image_grid_thw`` as a vLLM ``image_embeds`` content
part, carry M-RoPE-able 2-D positions for the image tokens.

This replaces the earlier flat ``prompt_embeds`` path: vLLM's
``--enable-prompt-embeds`` gives image tokens 1-D positions (no M-RoPE), so
Qwen3.5/3.6 could not read them. Producing the raw vision embeds + grid needs
only HF Transformers (no vLLM) and lets the serving model apply M-RoPE.

Architecture note: Qwen3.5 and Qwen3.6 share the ``Qwen3_5*`` transformers
architecture family (``Qwen3_5ForConditionalGeneration`` / ``Qwen3_5VisionModel``).
This is **distinct** from the ``Qwen3VL*`` family, whose visual processing
differs substantially; do not assume Qwen3-VL conventions here.
"""

import asyncio
import io
import json
import threading
import time

import torch
from PIL import Image
from scitrera_app_framework import Variables

from .base import ExtractionResult, VisualFeatures, VisualTokenizerProvider

EMBED_KIND_IMAGE = "image_embeds"


class Qwen35VisualTokenizer(VisualTokenizerProvider):
    """Qwen3.5/3.6 vision-tower image-embed extractor.

    Emits the projected per-image vision embeds (``pooler_output``) of shape
    ``[num_image_tokens, hidden_dim]`` plus the ``image_grid_thw`` grid.

    Each model size has a different vision architecture, so cached tensors from
    one model cannot be used with another; the cache is keyed by model_slug.
    """

    PROVIDER_NAME = "qwen3.5"

    def __init__(
        self,
        v: Variables = None,
        model_name: str = "Qwen/Qwen3.6-27B-FP8",
        vision_only: bool = False,
        torch_dtype: str = "auto",
        embed_kind: str = EMBED_KIND_IMAGE,
        partial_load: bool = True,
        max_image_dim: int = 0,
    ):
        super().__init__(v)
        self.model_name = model_name
        # The only supported output kind is image_embeds; the parameter is kept
        # for construction-call/config compatibility.
        self.embed_kind = EMBED_KIND_IMAGE
        self.vision_only = vision_only
        self.partial_load = partial_load
        self.torch_dtype = torch_dtype
        # Max of width/height (pixels) the image is downscaled to before the
        # processor/vision tower; 0 disables the resize.
        self.max_image_dim = int(max_image_dim or 0)
        self._processor = None
        self._model = None
        self._model_slug = self._make_model_slug(model_name)
        self._vision_only_loaded = False
        self._partial_loaded = False
        self._image_token_id: int | None = None
        self._model_load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.logger.info(
            "Initialized Qwen35VisualTokenizer: model=%s, embed_kind=%s, vision_only=%s, partial_load=%s, dtype=%s, max_image_dim=%s",
            model_name, self.embed_kind, self.vision_only, self.partial_load, torch_dtype, self.max_image_dim,
        )

    @staticmethod
    def _make_model_slug(model_name: str) -> str:
        """Create a filesystem-safe slug from model name."""
        return model_name.replace("/", "--").replace(" ", "_").lower()

    @property
    def model_slug(self) -> str:
        """Filesystem-safe model identifier for cache keying."""
        return self._model_slug

    def _resolve_dtype(self) -> torch.dtype | str:
        """Resolve torch_dtype config to actual dtype."""
        if self.torch_dtype == "auto":
            return "auto"
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, "auto")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy load model and processor under a process-local lock."""
        if self._model is not None:
            return
        lock = getattr(self, "_model_load_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._model_load_lock = lock
        with lock:
            if self._model is not None:
                return
            self._load_model_unlocked()

    def _load_model_unlocked(self):
        """Lazy load model and processor.

        Tries the vision-only model first (saves VRAM) when ``vision_only`` is
        set; otherwise loads the full conditional-generation model. With
        ``partial_load`` the text decoder layers are elided so only the
        embedding layer + vision tower + merger materialize — the decoder is
        never run for image-embed extraction.
        """
        if self._model is not None:
            return

        from transformers import AutoProcessor

        self.logger.info("Loading Qwen processor: %s", self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        dtype = self._resolve_dtype()

        if self.vision_only:
            try:
                from transformers import Qwen3_5VisionModel

                self.logger.info("Loading Qwen vision-only model: %s", self.model_name)
                self._model = Qwen3_5VisionModel.from_pretrained(
                    self.model_name, torch_dtype=dtype, device_map="auto",
                )
                self._vision_only_loaded = True
                self.logger.info("Qwen vision-only model loaded successfully")
                return
            except (ImportError, OSError, ValueError) as e:
                self.logger.warning(
                    "Vision-only load failed, falling back to full model: %s", e,
                )

        # Full multimodal model.
        try:
            from transformers import Qwen3_5ForConditionalGeneration as _Model
        except ImportError:
            from transformers import AutoModelForImageTextToText as _Model

        load_kwargs = dict(torch_dtype=dtype, device_map="auto", trust_remote_code=True)

        # Partial load: elide the text decoder so only the embedding layer +
        # vision tower + merger materialize. The decoder is never run for
        # image-embed extraction, so the vision output is identical to a full
        # load at a fraction of the VRAM.
        if self.partial_load:
            partial_config = self._build_partial_config()
            if partial_config is not None:
                try:
                    self.logger.info("Loading Qwen model (partial: decoder elided): %s", self.model_name)
                    self._model = _Model.from_pretrained(
                        self.model_name, config=partial_config, **load_kwargs,
                    )
                    self._partial_loaded = True
                    self._strip_generation_head()
                except Exception as e:  # noqa: BLE001 - fall back to full load
                    self.logger.warning(
                        "Partial load failed (%s); falling back to full model load", e,
                    )
                    self._model = None

        if self._model is None:
            self.logger.info("Loading full Qwen model: %s", self.model_name)
            self._model = _Model.from_pretrained(self.model_name, **load_kwargs)
            self._partial_loaded = False

        self._vision_only_loaded = False
        self._image_token_id = self._resolve_image_token_id()
        self.logger.info(
            "Qwen model loaded (partial=%s, image_token_id=%s)",
            self._partial_loaded, self._image_token_id,
        )

    def _build_partial_config(self):
        """Build a config with the text decoder layers elided (0 layers).

        Returns ``None`` if the config can't be introspected (caller then
        does a full load). The vision config is left untouched, so the vision
        tower + merger load fully; only the language decoder is dropped.
        """
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not load config for partial load: %s", e)
            return None

        if not self._elide_decoder_layers(cfg):
            self.logger.warning("Config has no num_hidden_layers; skipping partial load")
            return None
        return cfg

    @staticmethod
    def _elide_decoder_layers(cfg) -> bool:
        """Set the text decoder to 0 layers (and drop MTP heads) in-place.

        Operates on both the top-level config and a nested ``text_config`` if
        present. Returns True if at least one ``num_hidden_layers`` was found
        and zeroed (otherwise a partial load isn't meaningful).

        (``lm_head`` is dropped separately, post-load, by
        :meth:`_strip_generation_head` — this checkpoint ships an untied
        ``lm_head`` that transformers refuses to tie, so config tying is a
        no-op here.)
        """
        targets = [cfg]
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            targets.append(text_cfg)
        applied = False
        for target in targets:
            if hasattr(target, "num_hidden_layers"):
                target.num_hidden_layers = 0
                applied = True
            # Drop any multi-token-prediction / next-n heads too.
            for attr in ("num_nextn_predict_layers", "mtp_num_layers"):
                if hasattr(target, attr):
                    setattr(target, attr, 0)
        return applied

    def _strip_generation_head(self) -> None:
        """Free the generation-only ``lm_head`` (vocab×hidden) after a partial load.

        Image-embed extraction uses only the vision tower, never ``lm_head``.
        On this checkpoint ``lm_head`` is untied (~2.5 GB), so replacing it with
        ``Identity`` and releasing the cached allocation reclaims meaningful
        VRAM — important for the 24 GB L4 deployment where the tokenizer
        colocates with GLM-OCR + ColPali.
        """
        model = self._model
        head = getattr(model, "lm_head", None)
        if head is None or isinstance(head, torch.nn.Identity):
            return
        try:
            model.lm_head = torch.nn.Identity()
            del head
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("Dropped generation-only lm_head (partial image-embed load)")
        except Exception as e:  # noqa: BLE001 - non-fatal optimization
            self.logger.warning("Could not drop lm_head: %s", e)

    def _resolve_image_token_id(self) -> int | None:
        """Resolve the image-placeholder token id from config or tokenizer."""
        config = getattr(self._model, "config", None)
        for attr in ("image_token_id", "image_token_index"):
            tid = getattr(config, attr, None)
            if isinstance(tid, int):
                return tid
        # Fallback: look the placeholder string up in the tokenizer vocab.
        tok = getattr(self._processor, "tokenizer", None)
        if tok is not None:
            for placeholder in ("<|image_pad|>", "<image>", "<|image|>"):
                try:
                    tid = tok.convert_tokens_to_ids(placeholder)
                except Exception:  # noqa: BLE001 - tokenizer-specific
                    tid = None
                if isinstance(tid, int) and tid >= 0:
                    return tid
        return None

    @property
    def _device(self):
        if hasattr(self._model, "device"):
            return self._model.device
        return next(self._model.parameters()).device

    async def preload(self):
        """Preload model onto GPU."""
        self.logger.info("Preloading Qwen model")
        await asyncio.to_thread(self._load_model)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _compute_image_embeds(self, pixel_values, image_grid_thw) -> torch.Tensor:
        """Run the vision tower → ``[num_image_tokens, hidden]`` (version-tolerant)."""
        if self._vision_only_loaded:
            out = self._model(pixel_values, grid_thw=image_grid_thw)
        elif hasattr(self._model, "get_image_features"):
            out = self._model.get_image_features(pixel_values, image_grid_thw)
        else:
            out = self._model.visual(pixel_values, grid_thw=image_grid_thw)
        return self._coerce_image_embeds(out)

    @staticmethod
    def _coerce_image_embeds(out) -> torch.Tensor:
        """Unwrap an HF vision output into a 2-D ``[num_image_tokens, hidden]`` tensor.

        Qwen3.5/3.6's ``get_image_features`` returns an output object whose
        ``pooler_output`` holds the **projected** (text-hidden-space) per-image
        embeds, split per image as a tuple — this is what gets fed as the
        ``image_embeds`` content part (``last_hidden_state`` is the
        pre-projection vision-space output and has the wrong width). Plain
        tensors and lists pass straight through.
        """
        if not isinstance(out, torch.Tensor):
            for attr in ("pooler_output", "last_hidden_state"):
                val = getattr(out, attr, None)
                if val is not None:
                    out = val
                    break
        if isinstance(out, (list, tuple)):
            out = torch.cat(list(out), dim=0)
        return out

    def _spatial_merge_size(self) -> int:
        """Resolve the vision spatial merge factor (default 2) from config."""
        config = getattr(self._model, "config", None)
        for src in (getattr(config, "vision_config", None), config):
            size = getattr(src, "spatial_merge_size", None)
            if isinstance(size, int) and size > 0:
                return size
        return 2

    def _maybe_downscale(self, pil_image: "Image.Image") -> "Image.Image":
        """Downscale so the largest dimension is at most ``self.max_image_dim``.

        Aspect ratio is preserved and each dimension is clamped to >= 1px. When
        ``max_image_dim`` is 0/disabled or the image already fits under the cap
        the original image is returned untouched.
        """
        cap = self.max_image_dim
        if cap <= 0:
            return pil_image
        w, h = pil_image.size
        if max(w, h) <= cap:
            return pil_image
        scale = cap / max(w, h)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        self.logger.info(
            "Downscaling page image %sx%s -> %sx%s (max_image_dim=%s)",
            w, h, new_size[0], new_size[1], cap,
        )
        return pil_image.resize(new_size, Image.LANCZOS)

    def cache_key_suffix(self, metadata: dict | None) -> str:
        """Fingerprint config that changes the output tensor for the same image.

        The cache keys on image bytes; without this, changing ``max_image_dim``
        (or the embed kind) would silently return stale embeds. Folding these
        into the cache key forces a recompute when the producing config changes.
        """
        return f"{self.embed_kind}-d{self.max_image_dim}"

    def _extract_single(self, image_data: bytes, metadata: dict | None) -> VisualFeatures:
        """Extract vision-tower image embeds for a single page (GPU thread)."""
        self._load_model()

        pil_image = Image.open(io.BytesIO(image_data))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        # Capture the TRUE original size for provenance BEFORE any downscale.
        original_size = pil_image.size  # (width, height)

        # Optionally downscale to cap the vision-token count; everything
        # downstream (processor -> grid_thw -> vision tower) then operates on
        # the resized image automatically.
        pil_image = self._maybe_downscale(pil_image)

        lock = getattr(self, "_inference_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._inference_lock = lock

        with lock:
            # The image processor on the image alone yields pixel_values + the grid;
            # no text/chat template is needed for vision-tower extraction.
            inputs = self._processor(images=[pil_image], text="", return_tensors="pt")
            device = self._device
            pixel_values = inputs["pixel_values"].to(device)
            image_grid_thw = inputs["image_grid_thw"].to(device)
            with torch.no_grad():
                image_embeds = self._compute_image_embeds(pixel_values, image_grid_thw)
            image_embeds_cpu = image_embeds.cpu().contiguous()
            grid_list = [int(x) for x in image_grid_thw[0].cpu().tolist()]

            num_image_tokens = image_embeds_cpu.shape[0]
            merge = self._spatial_merge_size()
            expected = 1
            for d in grid_list:
                expected *= d
            expected //= merge * merge
            if num_image_tokens != expected:
                raise RuntimeError(
                    f"image-token count mismatch: vision tower yielded {num_image_tokens} "
                    f"tokens but grid {grid_list} with spatial_merge_size={merge} "
                    f"implies {expected}"
                )

            return VisualFeatures(
                image_embeds=image_embeds_cpu,
                image_grid_thw=grid_list,
                num_visual_tokens=num_image_tokens,
                hidden_dim=image_embeds_cpu.shape[-1],
                dtype=str(image_embeds_cpu.dtype),
                original_image_size=original_size,
                embed_kind=EMBED_KIND_IMAGE,
            )

    async def extract_features(
        self, image_data: bytes, metadata: dict | None = None,
    ) -> ExtractionResult:
        """Extract features from a single page."""
        start_time = time.monotonic()
        result = ExtractionResult(image_index=0)

        try:
            features = await asyncio.to_thread(self._extract_single, image_data, metadata)
            result.success = True
            result.features = features
        except torch.cuda.OutOfMemoryError as e:
            result.error = f"GPU OOM: {e}"
            self.logger.warning("GPU OOM during feature extraction: %s", e)
            torch.cuda.empty_cache()
        except Exception as e:
            result.error = str(e)
            self.logger.warning("Feature extraction failed: %s", e)

        result.latency_ms = (time.monotonic() - start_time) * 1000
        return result

    async def extract_features_batch(
        self,
        images: list[bytes],
        batch_size: int = 4,
        metadatas: list[dict] | None = None,
    ) -> list[ExtractionResult]:
        """Extract features from multiple pages in batches.

        Processes images in chunks of batch_size, clearing GPU cache between
        batches to manage VRAM.
        """
        results = []
        for batch_start in range(0, len(images), batch_size):
            batch = images[batch_start:batch_start + batch_size]
            for offset, image_data in enumerate(batch):
                idx = batch_start + offset
                meta = metadatas[idx] if metadatas and idx < len(metadatas) else None
                result = await self.extract_features(image_data, meta)
                result.image_index = idx
                results.append(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results

    def get_model_info(self) -> dict:
        """Return metadata about the loaded model."""
        info = {
            "provider": self.PROVIDER_NAME,
            "model_name": self.model_name,
            "model_slug": self._model_slug,
            "embed_kind": self.embed_kind,
            "vision_only": self._vision_only_loaded,
            "partial_loaded": self._partial_loaded,
            "torch_dtype": self.torch_dtype,
            "max_image_dim": self.max_image_dim,
            "loaded": self._model is not None,
        }
        if self._model is not None:
            config = getattr(self._model, "config", None)
            if config:
                info["vision_config"] = {
                    "hidden_size": getattr(config, "hidden_size", None),
                }
            info["image_token_id"] = self._image_token_id
        return info


# Retained so callers can json-encode model info dicts if needed.
_dump_json = json.dumps
