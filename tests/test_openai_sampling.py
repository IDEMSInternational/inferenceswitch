"""OpenAI request parameters: temperature per model, max_completion_tokens.

The reasoning models (o-series, GPT-5 and later) reject a non-default
``temperature`` and reject ``max_tokens`` with a 400. Offline: the SDK client
is a fake that records each request.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from inferenceswitch import (
    Capability,
    Client,
    ModelCard,
    Reliability,
    Route,
    Rung,
    default_cards,
    user,
)
from inferenceswitch.adapters.openai_compat import OpenAICompatibleAdapter
from inferenceswitch.registry import default_registry

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


class _RecordingOpenAI:
    def __init__(self):
        self.requests: list[dict] = []
        response = NS(
            choices=[NS(message=NS(content=json.dumps({"a": "x"}), tool_calls=None),
                        finish_reason="stop")],
            usage=NS(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None),
        )

        def create(**kwargs):
            self.requests.append(kwargs)
            return response

        self.chat = NS(completions=NS(create=create))


def _client(provider: str = "openai", **kwargs) -> tuple[Client, _RecordingOpenAI]:
    client = Client(key_provider=lambda spec: "k", **kwargs)
    fake = _RecordingOpenAI()
    spec = default_registry().get(provider)
    client._adapters[(provider, "k")] = OpenAICompatibleAdapter(spec, "k", raw_client=fake)
    return client, fake


def _every_call(client: Client, model: str) -> None:
    client.generate_structured(model=model, schema=SCHEMA, prompt="?", max_tokens=100)
    client.generate_text(model=model, prompt="?", max_tokens=100)
    client.chat(model=model, messages=[user("?")], max_tokens=100)


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5.4", "o4-mini", "gpt-6-sol", "gpt-9-new"])
def test_reasoning_and_unknown_models_get_no_temperature(model):
    client, fake = _client()
    _every_call(client, f"openai/{model}")
    assert len(fake.requests) == 3
    for request in fake.requests:
        assert "temperature" not in request


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1", "gpt-4.1-nano", "gpt-3.5-turbo"])
def test_non_reasoning_models_keep_temperature(model):
    client, fake = _client()
    _every_call(client, f"openai/{model}")
    assert [r["temperature"] for r in fake.requests] == [0.1, 0.1, 0.1]


def test_openai_sends_max_completion_tokens():
    client, fake = _client()
    _every_call(client, "openai/gpt-5-mini")
    for request in fake.requests:
        assert request["max_completion_tokens"] == 100
        assert "max_tokens" not in request


def test_no_cap_sends_neither_parameter():
    client, fake = _client()
    client.generate_text(model="openai/gpt-5-mini", prompt="?")
    assert "max_tokens" not in fake.requests[0]
    assert "max_completion_tokens" not in fake.requests[0]


@pytest.mark.parametrize("provider", ["ollama", "groq", "mistral", "deepseek"])
def test_other_openai_compatible_servers_are_unchanged(provider):
    client, fake = _client(provider)
    _every_call(client, f"{provider}/some-model")
    for request in fake.requests:
        assert request["temperature"] == 0.1
        assert request["max_tokens"] == 100
        assert "max_completion_tokens" not in request


def test_supports_agrees_with_what_is_sent():
    client = Client()
    assert not client.supports(Capability.SAMPLING_PARAMS, "openai")
    assert client.supports(Capability.SAMPLING_PARAMS, model="openai/gpt-4o-mini")
    assert not client.supports(Capability.SAMPLING_PARAMS, model="openai/gpt-5-mini")


def test_a_card_can_turn_temperature_off_for_a_model():
    card = ModelCard(
        id="house",
        rung=Rung.SMALL,
        routes=(Route("ollama", ("house:7b",)),),
        structured_output=Reliability.BEST_EFFORT,
        features_remove=frozenset({Capability.SAMPLING_PARAMS}),
    )
    client, fake = _client("ollama", cards=default_cards().merged([card]))
    client.generate_text(model="ollama/house:7b", prompt="?")
    client.generate_text(model="ollama/other:7b", prompt="?")
    assert "temperature" not in fake.requests[0]
    assert fake.requests[1]["temperature"] == 0.1
