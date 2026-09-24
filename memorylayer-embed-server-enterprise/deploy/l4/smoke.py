# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic-only endpoint qualification; exits nonzero on contract failure."""
import asyncio
import base64
import io
import json
import math
import sys
import time

import httpx
import numpy as np
from PIL import Image, ImageDraw


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
        value = response.json()['data'][0]['embedding']
        if isinstance(value, str):
            value = np.frombuffer(base64.b64decode(value), dtype='<f4').tolist()
        captured.append(value)

    wire = httpx.AsyncClient(base_url='http://127.0.0.1:18000/v1', timeout=300,
                             event_hooks={'response': [capture]}, trust_env=False)
    provider = VLLMSubprocessEmbeddingProvider(model_name='Qwen/Qwen3-VL-Embedding-2B', output_dimensions=1920,
                                               host='127.0.0.1', port=18000, max_concurrent=1)
    provider._ready = True
    provider._client = AsyncOpenAI(base_url=wire.base_url, api_key='x', http_client=wire)
    provider._mm_http = wire
    result = {}
    try:
        for kind, value in [('text', await provider.embed('Invoice total: 125 dollars')),
                            ('image', await provider.embed_image(encoded))]:
            full = captured.pop(0)
            assert len(full) == 2048 and value == full[:1920], kind
            result[kind] = {'native_dimensions': len(full), 'exact_prefix': True,
                            'output_norm': float(np.linalg.norm(value))}
    finally:
        await provider._client.close()
    return result


def qualify(url='http://127.0.0.1:61051'):
    image = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), 'MEMORYLAYER TEST\nInvoice total: 125 dollars\nDate: September 16, 2026', fill='black', font_size=28)
    output = io.BytesIO()
    image.save(output, format='PNG')
    encoded = base64.b64encode(output.getvalue()).decode()
    reports = {}
    with httpx.Client(base_url=url, timeout=600, trust_env=False) as client:
        client.get('/health/ready').raise_for_status()
        for path, body in [('/v1/embeddings', {'input': ['Invoice total: 125 dollars']}),
                           ('/v1/embeddings/images', {'images': [encoded], 'mode': 'single'}),
                           ('/v1/embeddings/multi', {'input': ['invoice total'], 'input_type': 'query'}),
                           ('/v1/transcribe', {'images': [encoded], 'max_tokens': 4096})]:
            start = time.monotonic()
            response = client.post(path, json=body)
            response.raise_for_status()
            data = response.json()
            reports[path] = {'seconds': time.monotonic() - start}
            if path.endswith('multi'):
                vectors = data['data'][0]['vectors']
                assert len(vectors) > 1 and all(len(v) == 128 for v in vectors), data
                reports[path]['vectors'] = len(vectors)
                query_vectors = np.asarray(vectors)
            elif path == '/v1/transcribe':
                page = data['results'][0]
                assert page['success'] and page['output_contract'] == 'unlimited_ocr' and page['raw_content'], data
                assert '125' in page['content'], data
                reports[path]['transcript'] = page['content']
            else:
                vector = data['data'][0]['embedding']
                assert len(vector) == 1920, len(vector)
                reports[path]['dimensions'] = len(vector)
                native_payload = {'model': 'Qwen/Qwen3-VL-Embedding-2B'}
                if path == '/v1/embeddings':
                    native_payload['input'] = body['input']
                else:
                    native_payload['messages'] = [{'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + encoded}}]}]
                native = client.post('http://127.0.0.1:18000/v1/embeddings', json=native_payload)
                native.raise_for_status()
                full = native.json()['data'][0]['embedding']
                assert len(full) == 2048, len(full)
                error = max(abs(a - b) for a, b in zip(vector, full[:1920], strict=True))
                prefix_norm = math.sqrt(sum(x * x for x in full[:1920]))
                norm = math.sqrt(sum(x * x for x in vector))
                reports[path].update(prefix_max_absolute_error=error, prefix_norm=prefix_norm, output_norm=norm)
                cosine = float(np.dot(vector, full[:1920]) / (norm * prefix_norm))
                reports[path]['cross_request_cosine'] = cosine
                # Record independent-call variation; test truncation against the
                # exact captured backend response separately below.
                assert np.isfinite(vector).all(), path
                repeated = client.post('http://127.0.0.1:18000/v1/embeddings', json=native_payload)
                repeated.raise_for_status()
                again = repeated.json()['data'][0]['embedding'][:1920]
                reports[path]['native_repeat_max_absolute_error'] = max(abs(a - b) for a, b in zip(full[:1920], again, strict=True))
            print(json.dumps({path: reports[path]}), file=sys.stderr, flush=True)
        reports['exact_provider_prefix'] = asyncio.run(check_exact_prefix(encoded))
        print(json.dumps({'exact_provider_prefix': reports['exact_provider_prefix']}), file=sys.stderr, flush=True)
        # Exercise the maximum decoded-pixel envelope on each image lane.
        large = Image.new('RGB', (2048, 1953), 'white')
        ImageDraw.Draw(large).text((100, 100), 'Maximum page test. Total: 125 dollars.', fill='black', font_size=48)
        buffer = io.BytesIO()
        large.save(buffer, format='PNG')
        maximum = base64.b64encode(buffer.getvalue()).decode()
        for mode in ('single', 'multi'):
            response = client.post('/v1/embeddings/images', json={'images': [maximum], 'mode': mode})
            response.raise_for_status()
            reports['maximum_image_' + mode] = {'passed': True}
        response = client.post('/v1/transcribe', json={'images': [maximum], 'max_tokens': 4096})
        response.raise_for_status()
        assert response.json()['results'][0]['success'], response.text
        reports['maximum_ocr_page'] = {'passed': True}
        # A small semantic MaxSim check catches malformed token vectors or
        # padding overwhelming the relevant document score. It is not a benchmark.
        response = client.post('/v1/embeddings/multi', json={'input': [
            'Invoice total: 125 dollars', 'Penguins swim in the icy ocean.']})
        response.raise_for_status()
        scores = [float((query_vectors @ np.asarray(item['vectors']).T).max(axis=1).sum())
                  for item in response.json()['data']]
        assert scores[0] > scores[1], scores
        reports['maxsim_ranking'] = {'scores': scores, 'relevant_document_rank': 1}
        # Repeat mixed traffic with all engines resident.
        for _ in range(3):
            response = client.post('/v1/embeddings', json={'input': ['invoice'] * 8})
            response.raise_for_status()
            response = client.post('/v1/embeddings/multi', json={'input': ['invoice total']})
            response.raise_for_status()
        client.get('/health/ready').raise_for_status()
    print(json.dumps(reports), file=sys.stderr, flush=True)
    return reports


if __name__ == '__main__':
    print(json.dumps(qualify(), indent=2))
