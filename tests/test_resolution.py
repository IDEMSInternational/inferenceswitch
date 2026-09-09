"""Routing tests — no SDKs or network required (adapters are never built)."""
from __future__ import annotations

import pytest

from inferenceswitch import LLMClient, ProviderResolutionError


@pytest.fixture
def client():
    return LLMClient()


def test_prefix_routes_native_providers(client):
    assert client.resolve("claude-opus-4-8").spec.name == "anthropic"
    assert client.resolve("gemini-3.5-flash").spec.name == "gemini"
    assert client.resolve("gpt-4o").spec.name == "openai"


def test_known_model_routes_by_exact_name(client):
    r = client.resolve("claude-haiku-4-5")
    assert r.spec.name == "anthropic"
    assert r.model == "claude-haiku-4-5"


def test_explicit_provider_wins(client):
    r = client.resolve("llama-3.3-70b-versatile", provider="groq")
    assert r.spec.name == "groq"
    assert r.model == "llama-3.3-70b-versatile"


def test_provider_slash_model_syntax(client):
    r = client.resolve("groq/llama-3.3-70b-versatile")
    assert r.spec.name == "groq"
    assert r.model == "llama-3.3-70b-versatile"


def test_hf_style_slash_names_are_not_treated_as_providers(client):
    # "meta-llama" is not a registered provider, so it must not be split off;
    # the whole string stays the model and, unroutable, raises.
    with pytest.raises(ProviderResolutionError):
        client.resolve("meta-llama/Llama-3-8b")


def test_local_servers_require_explicit_selection(client):
    r = client.resolve("qwen2.5-coder", provider="ollama")
    assert r.spec.name == "ollama"
    assert r.spec.is_local is True
    r2 = client.resolve("lmstudio/some-local-model")
    assert r2.spec.name == "lmstudio"


def test_unroutable_model_is_a_hard_error(client):
    with pytest.raises(ProviderResolutionError):
        client.resolve("totally-unknown-model")


def test_known_models_are_returned_fully_qualified(client):
    models = client.known_models()
    assert "anthropic/claude-opus-4-8" in models
    assert "gemini/gemini-3.5-flash" in models
    # Every entry must be routable back to its provider.
    for qualified in models:
        assert client.resolve(qualified).spec.name == qualified.split("/", 1)[0]
