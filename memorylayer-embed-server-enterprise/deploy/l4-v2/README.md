# Split L4 document services v2

Two independent enterprise embed-server images share the existing HTTP API behind
one path-routing proxy. The combined `deploy/l4` v1 profile is retained. These
profiles do not modify customers, migrate data, or enable a tenant NER extractor.

| Role | Models | Native sequences | Explicit KV cache | API concurrency / queue |
| --- | --- | --- | --- | --- |
| embedding | Qwen3-VL-Embedding-2B, ColModernVBERT, GLiNER2-base-v1 | Qwen 32, ColModern 16; one NER batch at a time | Qwen 8 GiB; pooling/NER have no generation KV cache | 32 / 64 |
| transcription | Unlimited-OCR, DeepSeek-OCR-2 | 8 per engine | 1.5 GiB per engine | 16 / 32 |

Each role uses one 24 GB L4. Qwen returns the exact first 1920 dimensions for both
text and images, without renormalization. ColModern returns native 128-dimensional
multivectors with pool factor 1. GLiNER2 uses an owned CUDA FP32 Python worker in a
separate locked venv; it does not use Tokenary. All required resources preload
concurrently and gate readiness. Model and dependency revisions are pinned.

Unlimited is the default first provider; DeepSeek is fallback, not load balancing.
Send `provider: "unlimited-ocr"` or `provider: "deepseek-ocr"` to `/v1/transcribe` to
choose one engine explicitly. Both return raw grounded output and the appropriate
`output_contract`. The recipes incorporate the last Thunder Unlimited and DeepSeek
OCR2 recipes: logits processors, data-parallel vision encoding, disabled prefix/MM
processor caches, and their request-side repetition/grounding contracts. A6000
utilization fractions were replaced with measured L4 KV budgets.

## Build, qualify, deploy

Use a new context path for every changed artifact. Source wheels and deployment
files are hashed in `manifest.json`; a base Git SHA alone does not describe dirty
local sources. The reviewed sparkrun revision is
`e2354585f2c463f96b1c40371d48f66c94654621` from `develop-next`.

```bash
uv run --no-project --with pyyaml --with packaging python ../l4/build_context.py \
  --profile embedding --oss /path/to/memorylayer \
  --sparkrun /path/to/pinned-sparkrun --output /path/to/new-embedding-context
# Repeat with --profile transcription and a different output directory.

docker build --platform linux/amd64 -t memorylayer-embedding:l4-v2 /path/to/new-embedding-context
# Build the transcription image from its separate context as well.
```

The image prebuilds sparkrun's `uv_venv` environments with the independent build
API. Model preparation happens separately on CPU; serving uses cached weights and
locked offline environments, with no pip/Hugging Face download on GPU cold start.
The NER lock intentionally uses Transformers 5.1 in its isolated environment;
vLLM 0.25 uses its own newer Transformers dependency. Both resolve normally.

Prepare each role's model volume with `/opt/memorylayer/prepare.py models` before
starting its image. Use the Compose profile or Helm releases described below.
Hosted deployment wrappers and operator/account configuration live outside this
public repository; no cloud SDK or private checkout is required to build these images.

`smoke.py` uses synthetic pages, checks model contracts and sweeps concurrency.
Run it inside a serving image with `MEMORYLAYER_EMBED_ROLE=embedding` or
`MEMORYLAYER_EMBED_ROLE=transcription` as appropriate. It talks to the local API
and native model ports; it is not a remote-endpoint probe. Verify at least 2 GiB
measured GPU headroom, repeated startup, failure cleanup and workload-specific
quality before deploying a changed profile. Short synthetic benchmarks are
capacity evidence, not customer throughput guarantees.

## Request envelope

- One image/page per request; at most 2048 pixels on either edge and 4 million pixels.
- Up to eight texts; total UTF-8 bytes 6144 for single vectors or 3072 for multivectors.
- NER: up to eight texts, 8192 total UTF-8 bytes, 32 labels, and 512 actual encoded
  tokens per text including its schema. Overlong input returns 422; no silent truncation.
- OCR output budget at most 4096 tokens, model context 8192; input body at most 12 MiB.
- Bounded admission returns 503 with Retry-After when full; application queue deadline
  120 seconds. Clients must not automatically replay ambiguous inference POST failures.

The proxy in platform-integration routes embedding/image/multivector/score/NER
paths to the embedding endpoint and transcription to the OCR endpoint, with separate
pools/credentials and same-origin 303 continuation handling. Local `/health/live`
never wakes GPUs. Role-specific health probes wake only the selected role; aggregate
readiness intentionally wakes both. The current synchronous enterprise NER consumer
still needs async offloading before enabling it in an async ingestion path.

## Compose and Helm

Build these x86-64 portable contexts with `docker build --platform linux/amd64`;
publish reviewed immutable
image digests before remote installation. `platform-integration/compose/profiles/
embed-gpu-v2.yaml` runs two preparation jobs, two GPU services, separate caches and
one CPU proxy. Assign different physical GPU UUIDs. Kubernetes uses two releases of
`platform-integration/charts/memorylayer-embed`, one image/PVC/preparation job/GPU per
role, then the v2 tenant proxy overlay. See platform-integration's
`docs/document-services.md` for credential references and rendering.

Compose and Helm rendering is validated separately from live GPU execution.
Acceptance remains installation-specific. Images are not published and consumer
configuration is not changed automatically by building these profiles.
