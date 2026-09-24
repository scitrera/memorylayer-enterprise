# Portable embedding and OCR appliance

This directory owns the portable combined serving profile, image builder, pinned
recipes, cache preparation, entrypoint and synthetic smoke checks. Deployment
composition and consumer wiring live in platform-integration. Hosted deployment
wrappers, account settings and qualification records are maintained separately by
operators; building this image requires no deployment account or private repository.

One 24 GB L4, one embed API on port 61051, and three native vLLM 0.25.0 processes launched by sparkrun's local executor:

| Role | Model | Output | GPU fraction | Explicit KV cache | Context | Internal port |
| --- | --- | --- | --- | --- | --- | --- |
| Single vector | Qwen/Qwen3-VL-Embedding-2B | 1920-component prefix, no renormalization; text and images | 0.32 | 2,415,919,104 bytes (2.25 GiB) | 8192 | 18000 |
| Multivector | ModernVBERT/colmodernvbert-merged | 128-dimensional token/patch vectors | 0.10 | None (encoder pooling) | 4096 | 18001 |
| Transcription | baidu/Unlimited-OCR | Clean transcript plus raw grounded output, `unlimited_ocr` contract | 0.40 | 3,435,973,837 bytes (~3.2 GiB) | 8192 / 4096 generated | 18012 |

Model revisions live in the recipes. The vLLM 0.25.0 ColModernVBERT implementation uses its native Idefics3 image path with `do_image_splitting=False`; this is a distinct preprocessing contract from tiled Sentence Transformers/ColPali ingestion. Do not mix their indexes without re-embedding. The pinned Torchvision 0.26 processor can substitute bicubic for Lanczos; representative visual-retrieval acceptance is required before rollout. NER, document chat, visual tokenizer, other OCR providers and the Aether sidecar are disabled. Only the embed API binds outside loopback. Use one Uvicorn worker. Do not mount a Docker socket. Deploy behind a private network or authenticated ingress.

## Startup and memory allocation

The L4 profile enables `MEMORYLAYER_EMBED_PARALLEL_STARTUP=true`. OCR launches first,
then the embedding processes launch while OCR initializes. Sparkrun's shared
controller operations are serialized; model initialization and HTTP health waits
overlap. All three engines must pass readiness before the application serves.
Set the flag to `false` to retain sequential OCR → Qwen → ColModernVBERT startup.
Other embed-server deployments default to sequential startup unless opted in.

Initial parallel qualification used OCR/Qwen KV caches of 2 and 1.25 GiB,
just below the prior automatic allocations of about 2.10 and 1.28 GiB. The current
profile expands them to approximately 3.2 and 2.25 GiB, respectively. ColModernVBERT
has no KV cache to expand. These settings avoid cross-process automatic memory
profiling during overlapping startup. They cap **KV cache only**, not model
weights, activations or total GPU memory. With an explicit KV size, vLLM does not
use the GPU fraction to calculate cache capacity; the fraction remains relevant
to its initial free-memory check. ColModernVBERT's encoder pooling path has no
attention KV cache. All three models retain their existing context/request limits; extra cache capacity
does not itself increase the one-request-at-a-time limit.

A cancelled launch is settled before cleanup decides which detached jobs it owns.
Startup failure cancels sibling preloads, waits for their cleanup and aborts the
appliance. Failed stops are logged and retain ownership for a cleanup retry.

## Build an image context

The implementation needs the companion OSS MemoryLayer runner/client changes and sparkrun's `uv-venv` `verify_only` option. The sparkrun changes are published on [`develop-next` at `e2354585f2c463f96b1c40371d48f66c94654621`](https://github.com/spark-arena/sparkrun/commit/e2354585f2c463f96b1c40371d48f66c94654621); pin that commit when cloning/building. Build MemoryLayer wheels from the reviewed checkouts until their changes are released; older published versions will not work. The independent API was introduced at `ccfbb625877199a6252cbe42c578450d274b5e32`; the pinned `develop-next` commit also contains the required verify-only change. The manifest records base commits **and wheel SHA-256**, so an uncommitted checkout cannot masquerade as that base commit's contents.

```sh
uv run --no-project --with packaging==26.3 --with pyyaml==6.0.3 python build_context.py --oss /path/to/memorylayer --sparkrun /path/to/sparkrun --output /tmp/embed-l4-context
```

This allowlists four Python packages, builds wheels, and stages only deployment code/recipes/locks. It does not upload repository roots, local configuration, customer documents, training sets or credentials. Wheels are built in fresh temporary directories containing only package source and build metadata, excluding stale checkout `build/` products, egg-info and unrelated files. Wheel timestamps use the source commit timestamp for repeatable builds. It copies the checked-in `requirements.lock` and `controller.lock`; `--refresh-lock` is an explicit dependency update requiring renewed qualification. Review `manifest.json` and retain it with the image.

```sh
docker buildx build --platform linux/amd64 --load -t memorylayer-embed:l4-candidate /tmp/embed-l4-context
```

The CUDA base is pinned to its AMD64 manifest. The image build creates a small locked controller environment, then calls public `BuildOptions` → `plan_build` → `build` with model distribution disabled. That prebuilds `/opt/memorylayer/venvs/vllm` at its final path, retaining the builder hash. Wheels are installed into that environment and dependency consistency/imports checked. Runtime recipes use `verify_only: true`: mismatched or absent environments fail before uv/pip or Python downloads. The activation file is written to `/run/memorylayer`, outside the preseeded environment.

## Prepare caches, then serve

Cache preparation uses the same build API with model distribution enabled. It does not launch vLLM or allocate an inference GPU. Sparkrun's cache argument is the **HF home**, while `snapshot_download(cache_dir=...)` takes its **hub subdirectory**. A local-only snapshot check verifies pinned configs and safetensors before writing `prepared.json`.

```sh
docker volume create embed-models
docker run --rm -v embed-models:/cache/huggingface --entrypoint /opt/memorylayer/venvs/vllm/bin/python memorylayer-embed:l4-candidate /opt/memorylayer/prepare.py models
docker run --rm --gpus 'device=0' --shm-size=4g -v embed-models:/cache/huggingface -p 127.0.0.1:61051:61051 memorylayer-embed:l4-candidate
```

Serving forces Hugging Face and uv offline mode. Missing weights fail instead of downloading at cold start. Keep sparkrun state container-local. Sharing the model cache is supported; sharing controller PID/job state is not.

`/health/ready` checks all three actual engines. Failed required startup aborts and cleans up owned children. `/health` is process liveness only. Request limits serialize work, accept at most eight text inputs within 6144 UTF-8 bytes (3072 for multivector), one image/page up to 2048 pixels per edge and 4M pixels total, 12 MiB JSON bodies, and 4096 generated tokens. Oversized inputs return an error. The consumer's `MEMORYLAYER_EMBED_IMAGE_BATCH_SIZE=1` splits batches while preserving page indexes. Rasterizing ordinary pages at 150 DPI fits; oversized scans need explicit resizing before submission.

## Installation and verification

Use platform-integration's `compose/profiles/embed-gpu.yaml` or
`charts/memorylayer-embed` with a reviewed image digest. The image serves on port
61051; use a private network or authenticated ingress. Model cache preparation is
a separate CPU-only step. This image does not migrate customer storage.

Run `python /opt/memorylayer/smoke.py` inside the running appliance to exercise
synthetic text/image embeddings, exact prefix extraction, native multivectors,
MaxSim and OCR. Validate GPU headroom, startup/failure cleanup, and representative
retrieval/citations for each installation. Benchmark records and hosted endpoints
are operator artifacts, not part of this public distribution.

The image manifest hashes source wheels, recipes and runtime files only. Deployment
wrappers should record their own hashes alongside that manifest without coupling
the portable build to a cloud SDK or a private checkout.
