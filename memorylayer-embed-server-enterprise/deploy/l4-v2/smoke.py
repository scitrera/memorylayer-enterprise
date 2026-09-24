# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic split-profile contracts and repeatable per-container load sweeps."""

import asyncio
import base64
import io
import json
import os
import statistics
import sys
import time

import httpx
import numpy as np
from PIL import Image, ImageDraw


def page(maximum=False):
    im = Image.new("RGB", (2048, 1953) if maximum else (800, 600), "white")
    draw = ImageDraw.Draw(im)
    draw.text(
        (40, 40), "MEMORYLAYER TEST\nInvoice total: 125 dollars\nDate: September 17, 2026", fill="black", font_size=48 if maximum else 28
    )
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def validate(path, body, data):
    if path == "/v1/transcribe":
        result = data["results"][0]
        assert result["success"] and result["raw_content"], data
        assert result["output_contract"] in {"unlimited_ocr", "deepseek_ocr"}, data
        if body.get("provider"):
            assert result["provider_used"] == body["provider"] and len(result["attempts"]) == 1, data
        assert "125" in result["content"], data
    elif path == "/v1/ner":
        assert data["results"] and data["results"][0]["entities"].get("person"), data
    else:
        assert data["data"], data
        for item in data["data"]:
            if "embedding" in item:
                assert len(item["embedding"]) == 1920 and np.isfinite(item["embedding"]).all()
            else:
                vectors = item["vectors"]
                assert len(vectors) > 1 and all(len(v) == 128 for v in vectors)


async def check_exact_prefix(encoded):
    """Observe each real backend response before the production provider slices it.

    Comparing independent vision inference calls conflates numeric repeatability
    with truncation. This validates the exact response used by both provider paths.
    No model is launched or stopped by this observer.
    """
    from memorylayer_embed_server.services.embedding.vllm_subprocess import VLLMSubprocessEmbeddingProvider
    from openai import AsyncOpenAI

    captured = []

    async def capture(response):
        await response.aread()
        response.raise_for_status()
        value = response.json()["data"][0]["embedding"]
        if isinstance(value, str):
            value = np.frombuffer(base64.b64decode(value), dtype="<f4").tolist()
        captured.append(value)

    wire = httpx.AsyncClient(base_url="http://127.0.0.1:18000/v1", timeout=300, event_hooks={"response": [capture]}, trust_env=False)
    provider = VLLMSubprocessEmbeddingProvider(
        model_name="Qwen/Qwen3-VL-Embedding-2B", output_dimensions=1920, host="127.0.0.1", port=18000, max_concurrent=1
    )
    provider._ready = True
    provider._client = AsyncOpenAI(base_url=wire.base_url, api_key="x", http_client=wire)
    provider._mm_http = wire
    result = {}
    try:
        for kind, value in [("text", await provider.embed("Invoice total: 125 dollars")), ("image", await provider.embed_image(encoded))]:
            full = captured.pop(0)
            assert len(full) == 2048 and value == full[:1920], kind
            result[kind] = {"native_dimensions": len(full), "exact_prefix": True, "output_norm": float(np.linalg.norm(value))}
    finally:
        await provider._client.close()
    return result


async def qualify(url="http://127.0.0.1:61051"):
    role = os.environ["MEMORYLAYER_EMBED_ROLE"]
    encoded, maximum = page(), page(True)
    reports = {"role": role, "sweeps": []}
    if role == "embedding":
        workloads = {
            "single_text": [("/v1/embeddings", {"input": ["Invoice total: 125 dollars"]})],
            "single_batch": [("/v1/embeddings", {"input": ["Invoice total: 125 dollars"] * 8})],
            "multi_text": [("/v1/embeddings/multi", {"input": ["Invoice total: 125 dollars"]})],
            "mixed_images": [("/v1/embeddings/images", {"images": [encoded], "mode": mode}) for mode in ("single", "multi")],
            "maximum_images": [("/v1/embeddings/images", {"images": [maximum], "mode": mode}) for mode in ("single", "multi")],
        }
        if os.path.exists("/opt/memorylayer/ner.json"):
            ner = ("/v1/ner", {"texts": ["Alice works at Scitrera in Chicago."], "labels": ["person", "organization", "location"]})
            workloads["ner"] = [ner]
            labels = ["person", "organization", "location"] + [f"category{i}" for i in range(29)]
            # Exercise all eight batch slots and 32 labels, close to the UTF-8 budget.
            text = "Alice works at Scitrera in Chicago. " + "document " * 100
            workloads["maximum_ner"] = [("/v1/ner", {"texts": [text] * 8, "labels": labels})]
            workloads["mixed_ner_embeddings"] = [*workloads["mixed_images"], ner, *workloads["single_batch"]]
        disabled = "/v1/transcribe"
    else:
        workloads = {
            provider: [("/v1/transcribe", {"images": [encoded], "provider": provider})] for provider in ("unlimited-ocr", "deepseek-ocr")
        }
        workloads["mixed_ocr"] = [item for items in workloads.values() for item in items]
        workloads["maximum_ocr"] = [
            ("/v1/transcribe", {"images": [maximum], "provider": provider}) for provider in ("unlimited-ocr", "deepseek-ocr")
        ]
        disabled = "/v1/embeddings"
    async with httpx.AsyncClient(base_url=url, timeout=600, trust_env=False, limits=httpx.Limits(keepalive_expiry=4)) as client:
        (await client.get("/health/ready")).raise_for_status()
        assert (await client.post(disabled, json={})).status_code == 404
        # Warm every workload before timing; surface contract errors before sweeps.
        for jobs in workloads.values():
            for path, body in jobs:
                r = await client.post(path, json=body)
                r.raise_for_status()
                validate(path, body, r.json())
        if role == "embedding":
            reports["exact_prefix"] = await check_exact_prefix(encoded)
            r = await client.post("/v1/embeddings/multi", json={"input": ["Invoice total: 125 dollars"]})
            r.raise_for_status()
            vectors = r.json()["data"][0]["vectors"]
            score = await client.post("/v1/score", json={"query_vectors": vectors, "document_vectors": [vectors]})
            score.raise_for_status()
            assert score.json()["scores"][0]["score"] > 0
            if os.path.exists("/opt/memorylayer/ner.json"):
                r = await client.post("/v1/ner", json={"texts": ["word " * 1000], "labels": ["person"]})
                assert r.status_code == 422, r.text
        if role == "transcription":
            r = await client.post("/v1/transcribe", json={"images": [encoded]})
            r.raise_for_status()
            assert r.json()["results"][0]["provider_used"] == "unlimited-ocr", r.text
        for name, jobs in workloads.items():
            for concurrency in (1, 2, 4, 8, 16, 32) if role == "embedding" else (1, 2, 4, 8, 16):
                # Fixed work for meaningful throughput comparisons, bounded tasks.
                count = (
                    8
                    if name == "maximum_ner"
                    else 64
                    if "maximum" not in name and role == "embedding"
                    else 32
                    if role == "embedding"
                    else 16
                )
                sem = asyncio.Semaphore(concurrency)
                times = []
                load = []

                async def once(index):
                    async with sem:
                        began = time.monotonic()
                        path, body = jobs[index % len(jobs)]
                        response = await client.post(path, json=body)
                        response.raise_for_status()
                        validate(path, body, response.json())
                        times.append(time.monotonic() - began)

                done = asyncio.Event()

                async def sample():
                    while not done.is_set():
                        load.append((await client.get("/health/load")).json())
                        try:
                            await asyncio.wait_for(done.wait(), 0.2)
                        except TimeoutError:
                            pass

                sampler = asyncio.create_task(sample())
                started = time.monotonic()
                try:
                    async with asyncio.TaskGroup() as group:
                        for index in range(count):
                            group.create_task(once(index))
                finally:
                    done.set()
                    await sampler
                elapsed = time.monotonic() - started
                report = {
                    "workload": name,
                    "concurrency": concurrency,
                    "requests": count,
                    "seconds": elapsed,
                    "requests_per_second": count / elapsed,
                    "p50_seconds": statistics.median(times),
                    "p95_seconds": sorted(times)[int((len(times) - 1) * 0.95)],
                    "peak_provider_active": {
                        key: max(s.get("providers", {}).get(key, {}).get("in_flight", 0) for s in load)
                        for key in set(k for s in load for k in s.get("providers", {}))
                    },
                }
                reports["sweeps"].append(report)
                print(json.dumps(report), file=sys.stderr, flush=True)
        (await client.get("/health/ready")).raise_for_status()
    return reports


if __name__ == "__main__":
    print(json.dumps(asyncio.run(qualify()), indent=2))
