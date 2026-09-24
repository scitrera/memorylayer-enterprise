"""Validate the actual deployment selection without launching models."""
import logging
from pathlib import Path

from memorylayer_embed_server.dependencies import _setup_dual_embedding_service, _setup_transcription_cascade
from scitrera_app_framework import Variables


def test_all_three_lanes_use_sparkrun_and_expected_budgets(monkeypatch):
    for line in (Path(__file__).parent / 'profile.env').read_text().splitlines():
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            monkeypatch.setenv(key, value)
    v = Variables()
    _setup_dual_embedding_service(v, logging.getLogger('test'))
    _setup_transcription_cascade(v, logging.getLogger('test'))
    dual = v.get('dual_embedding_service')
    providers = [dual._single_vector, dual._multi_vector, *v.get('cascade_transcriber').providers]
    assert len(providers) == 3 and all(providers)
    assert [type(p._runner).__name__ for p in providers] == ['SparkrunVLLMRunner'] * 3
    assert [p._runner.gpu_memory_utilization for p in providers] == [.32, .10, .40]
    assert [p._runner.port for p in providers] == [18000, 18001, 18012]
    assert [p._runner.effective_max_concurrent for p in providers] == [1, 1, 1]
    assert dual.single_vector.dimensions == 1920
    assert dual.multi_vector.dimensions == 128
    assert providers[2].default_max_tokens == 4096
