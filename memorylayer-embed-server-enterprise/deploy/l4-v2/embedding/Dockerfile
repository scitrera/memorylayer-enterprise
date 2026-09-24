# syntax=docker/dockerfile:1
# SPDX-License-Identifier: AGPL-3.0-only
ARG CUDA_IMAGE=nvidia/cuda:13.1.1-devel-ubuntu24.04@sha256:18f75176b0fb66e32553e3a0e47e50d9c24f7e183f6e166dbecd29bf7a5ebb32
FROM ${CUDA_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/cache/huggingface HF_HUB_CACHE=/cache/huggingface/hub \
    XDG_CACHE_HOME=/run/memorylayer/cache XDG_CONFIG_HOME=/run/memorylayer/config \
    XDG_STATE_HOME=/run/memorylayer/state PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev ca-certificates curl git build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv
WORKDIR /opt/memorylayer
COPY requirements.lock controller.lock prepare.py sparkrun.yaml /opt/memorylayer/
COPY recipes /opt/memorylayer/recipes/
COPY wheels/sparkrun-*.whl /opt/memorylayer/wheels/
RUN uv venv /opt/memorylayer/controller --python /usr/bin/python3.12 && \
    uv pip install --python /opt/memorylayer/controller/bin/python --require-hashes -r /opt/memorylayer/controller.lock && \
    uv pip install --python /opt/memorylayer/controller/bin/python --no-deps /opt/memorylayer/wheels/sparkrun-*.whl && \
    /opt/memorylayer/controller/bin/python /opt/memorylayer/prepare.py environment
COPY wheels /opt/memorylayer/wheels/
RUN uv pip install --python /opt/memorylayer/venvs/vllm/bin/python --no-deps /opt/memorylayer/wheels/*.whl && \
    uv pip check --python /opt/memorylayer/venvs/vllm/bin/python && \
    /opt/memorylayer/venvs/vllm/bin/python -c "from sparkrun.api import BuildOptions, plan_build, build; import memorylayer_embed_server; import memorylayer_embed_server_enterprise; from vllm.model_executor.models.unlimited_ocr import NGramPerReqLogitsProcessor"
COPY ner.lock ner_worker.py ner.json /opt/memorylayer/
RUN uv venv /opt/memorylayer/venvs/ner --python /usr/bin/python3.12 && \
    uv pip install --python /opt/memorylayer/venvs/ner/bin/python --require-hashes -r /opt/memorylayer/ner.lock && \
    uv pip check --python /opt/memorylayer/venvs/ner/bin/python && \
    /opt/memorylayer/venvs/ner/bin/python -c "from gliner2 import GLiNER2"
COPY serve.py profile.env smoke.py manifest.json /opt/memorylayer/
ENV PATH=/opt/memorylayer/venvs/vllm/bin:/usr/local/cuda/bin:$PATH
EXPOSE 61051
ENTRYPOINT ["/opt/memorylayer/venvs/vllm/bin/python", "/opt/memorylayer/serve.py"]
