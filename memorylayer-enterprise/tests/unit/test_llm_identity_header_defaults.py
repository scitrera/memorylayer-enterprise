"""Public distributions require an operator to opt in to identity headers."""
from memorylayer_saas.config import DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS
from memorylayer_server.config import (
    DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS as OSS_DEFAULT,
    MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS,
)
from memorylayer_server.services.llm.attribution import host_allows_identity, parse_host_patterns
from scitrera_app_framework.api import Variables


def test_public_defaults_send_no_identity_headers():
    assert DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS == OSS_DEFAULT == ""
    patterns = parse_host_patterns(DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS)
    assert patterns == ()
    for url in (None, "https://gateway.example.test/v1", "https://api.openai.com/v1"):
        assert host_allows_identity(url, patterns) is False


def test_explicit_gateway_allowlist_survives_registry_read(monkeypatch):
    monkeypatch.setenv(MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS, "gateway.example.test")
    v = Variables()
    v.set_default_value(MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS, DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS)
    patterns = parse_host_patterns(v.environ(MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS))
    assert host_allows_identity("https://gateway.example.test/v1", patterns)
    assert not host_allows_identity("https://api.openai.com/v1", patterns)
    assert not host_allows_identity("https://gateway.example.test.evil.test/v1", patterns)
