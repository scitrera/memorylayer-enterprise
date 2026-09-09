"""Unit tests for GLiNER2 NER service failure semantics."""

import pytest
from pydantic import ValidationError

from memorylayer_embed_server_enterprise.models.ner import NERRequest
from memorylayer_embed_server_enterprise.services.ner import GLiNER2NERService


class _FailingModel:
    def extract_entities(self, text, labels):
        raise RuntimeError("model failed")


@pytest.mark.asyncio
async def test_extract_batch_propagates_text_inference_failure():
    """A model failure must surface to the HTTP layer so SaaS regex fallback runs."""
    service = GLiNER2NERService(model_name="test-model", default_labels=["person"])
    service._model = _FailingModel()

    with pytest.raises(RuntimeError, match="NER inference failed"):
        await service.extract_batch(["Alice met Bob"], ["person"])


def test_ner_request_rejects_oversized_text_batch():
    with pytest.raises(ValidationError):
        NERRequest(texts=["Alice"] * 129)


def test_ner_request_rejects_oversized_text_payload():
    with pytest.raises(ValidationError):
        NERRequest(texts=["a" * 20_001])


def test_ner_request_rejects_oversized_label_batch():
    with pytest.raises(ValidationError):
        NERRequest(texts=["Alice"], labels=["person"] * 129)
