"""Capability introspection tests — offline (no adapters built)."""
from __future__ import annotations

import pytest

from inferenceswitch import (
    Capabilities,
    Capability,
    Client,
    ModelChoice,
    ProviderSpec,
    Registry,
    SchemaDialect,
    StructuredMode,
    UnsupportedCapabilityError,
)
from inferenceswitch.registry import KIND_OPENAI


@pytest.fixture
def client():
    return Client()


def test_supports_by_provider_and_by_model(client):
    # By provider name, and by model string (routed) — same answer.
    assert client.supports(Capability.EXPLICIT_PROMPT_CACHING, "anthropic")
    assert client.supports(Capability.EXPLICIT_PROMPT_CACHING, model="claude-opus-4-8")


def test_anthropic_sampling_params_fail_closed(client):
    # Per-model in reality (Opus rejects, Haiku accepts); modeled fail-closed at
    # the provider level so require() never green-lights a 400.
    assert not client.supports(Capability.SAMPLING_PARAMS, "anthropic")


def test_local_servers_have_a_minimal_feature_set(client):
    assert client.supports(Capability.TOOL_CALLING, "ollama")
    # Local structured output is best-effort, not strictly enforced.
    assert not client.supports(Capability.STRICT_SCHEMA_OUTPUT, "ollama")
    assert not client.supports(Capability.EXPLICIT_PROMPT_CACHING, "lmstudio")


def test_require_raises_with_a_helpful_message(client):
    with pytest.raises(UnsupportedCapabilityError) as exc:
        client.require(Capability.AUDIO_INPUT, "openai")
    # Message names the capability and lists what IS supported.
    assert "audio_input" in str(exc.value)
    assert "Supported:" in str(exc.value)


def test_require_passes_for_supported_capability(client):
    client.require(Capability.TOOL_CALLING, "openai")  # no raise
    client.require(Capability.AUDIO_INPUT, model="gemini-3.5-flash")  # no raise


def test_capabilities_object_is_introspectable(client):
    caps = client.capabilities("gemini")
    assert Capability.VISION in caps.features
    assert caps.schema_dialect.value == "gemini"
    assert caps.has(Capability.PDF_INPUT)


def test_models_for_curated_entries_route_as_plain_strings(client):
    choices = client.models_for(Capability.EXPLICIT_PROMPT_CACHING)
    opus = next(c for c in choices if c == "anthropic/claude-opus-4-8")
    # Still a str, so it passes straight to chat() as it always did.
    assert isinstance(opus, str)
    assert opus.provider == "anthropic" and opus.model == "claude-opus-4-8"
    assert not opus.needs_model
    assert client.resolve(opus).spec.name == "anthropic"


def test_models_for_includes_providers_without_a_catalog(client):
    choices = client.models_for(Capability.TOOL_CALLING)
    by_provider = {c.provider for c in choices}
    # Catalog-less providers qualify on capabilities and must not be hidden.
    assert {"groq", "ollama", "lmstudio"} <= by_provider
    groq = next(c for c in choices if c.provider == "groq")
    assert groq == "groq"
    assert groq.needs_model and groq.model is None


def test_catalog_less_provider_contributes_exactly_one_entry(client):
    choices = client.models_for(Capability.TOOL_CALLING)
    assert [c for c in choices if c.provider == "ollama"] == [ModelChoice("ollama")]
    # Curated providers still contribute one entry per model.
    assert len([c for c in choices if c.provider == "anthropic"]) == 3


def test_models_for_still_filters_on_capabilities(client):
    # Locals lack strict schema output; Groq lacks batch — neither should appear
    # just because they are catalog-less.
    assert not [
        c for c in client.models_for(Capability.STRICT_SCHEMA_OUTPUT)
        if c.provider in {"ollama", "lmstudio"}
    ]
    assert not [c for c in client.models_for(Capability.BATCH) if c.provider == "groq"]


def test_needs_model_filter_gives_the_routable_subset(client):
    routable = [c for c in client.models_for(Capability.TOOL_CALLING) if not c.needs_model]
    assert routable  # sanity
    for choice in routable:
        assert client.resolve(choice).spec.name == choice.provider


def test_uncurated_custom_provider_appears_in_the_menu():
    registry = Registry(
        [
            ProviderSpec(
                name="mylab",
                kind=KIND_OPENAI,
                base_url="https://llm.mylab.example/v1",
                api_key_env="MYLAB_API_KEY",
                capabilities=Capabilities(
                    structured_output=StructuredMode.RESPONSE_FORMAT_JSON_SCHEMA,
                    schema_dialect=SchemaDialect.OPENAI,
                    features=frozenset({Capability.TOOL_CALLING}),
                ),
            )
        ]
    )
    (choice,) = Client(registry=registry).models_for(Capability.TOOL_CALLING)
    assert choice == "mylab" and choice.needs_model


def test_provider_specific_feature_differences(client):
    # Mistral advertises vision + batch; DeepSeek neither (reasoning there is via
    # model choice, not an effort knob, so it's not REASONING_EFFORT either).
    assert client.supports(Capability.BATCH, "mistral")
    assert client.supports(Capability.VISION, "mistral")
    assert not client.supports(Capability.BATCH, "deepseek")
    assert not client.supports(Capability.REASONING_EFFORT, "deepseek")
