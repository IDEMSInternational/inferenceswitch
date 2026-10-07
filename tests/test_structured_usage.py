"""Structured calls report token usage: ``generate_structured`` returns a
``StructuredResult`` whose ``usage`` uses the same mapping as each adapter's
``chat`` path, and ``generate_structured_json`` still returns just the value.

Offline — each adapter is driven with a fake SDK client whose response carries
(or omits) usage, so the assertions are about how inferenceswitch reads it.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from inferenceswitch import Client, StopReason, StructuredResult, Usage
from inferenceswitch.adapters.anthropic import AnthropicAdapter
from inferenceswitch.adapters.gemini import GeminiAdapter
from inferenceswitch.adapters.openai_compat import OpenAICompatibleAdapter
from inferenceswitch.registry import default_registry

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
VALUE = {"answer": "42"}


class _FakeAnthropic:
    def __init__(self, usage):
        self.message = NS(
            content=[NS(type="text", text=json.dumps(VALUE))],
            stop_reason="end_turn",
            stop_details=None,
            usage=usage,
        )
        self.messages = NS(stream=self._stream)

    def _stream(self, **kwargs):
        message = self.message

        class _Stream:
            def __enter__(inner):
                return inner

            def __exit__(inner, *exc):
                return False

            def get_final_message(inner):
                return message

        return _Stream()


class _FakeGenai:
    def __init__(self, usage_metadata):
        self.response = NS(
            candidates=[NS(content=NS(parts=[]), finish_reason=NS(name="STOP"))],
            text=json.dumps(VALUE),
            usage_metadata=usage_metadata,
        )
        self.models = NS(generate_content=lambda **kwargs: self.response)


class _FakeOpenAI:
    def __init__(self, usage):
        self.response = NS(
            choices=[NS(message=NS(content=json.dumps(VALUE)), finish_reason="stop")],
            usage=usage,
        )
        self.chat = NS(completions=NS(create=lambda **kwargs: self.response))


def _anthropic(usage):
    fake = _FakeAnthropic(usage)
    return AnthropicAdapter(default_registry().get("anthropic"), "", raw_client=fake), fake.message


def _gemini(usage_metadata):
    fake = _FakeGenai(usage_metadata)
    return GeminiAdapter(default_registry().get("gemini"), "", raw_client=fake), fake.response


def _openai(usage):
    fake = _FakeOpenAI(usage)
    return OpenAICompatibleAdapter(default_registry().get("openai"), "", raw_client=fake), fake.response


def _call(adapter, model):
    return adapter.generate_structured(model=model, prompt="?", schema=SCHEMA)


# ── usage is filled where the provider reports it ────────────────────────────


def test_anthropic_reports_usage():
    adapter, raw = _anthropic(
        NS(
            input_tokens=120,
            output_tokens=30,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=5,
        )
    )
    result = _call(adapter, "claude-opus-4-8")
    assert isinstance(result, StructuredResult)
    assert result.value == VALUE
    assert result.usage == Usage(
        input_tokens=120, output_tokens=30, cache_read_tokens=100, cache_write_tokens=5
    )
    assert result.stop_reason is StopReason.END_TURN
    assert result.raw is raw


def test_gemini_reports_usage():
    adapter, raw = _gemini(
        NS(
            prompt_token_count=200,
            candidates_token_count=40,
            cached_content_token_count=150,
            thoughts_token_count=12,
        )
    )
    result = _call(adapter, "gemini-3.5-flash")
    assert result.value == VALUE
    assert result.usage == Usage(
        input_tokens=200, output_tokens=40, cache_read_tokens=150, reasoning_tokens=12
    )
    assert result.stop_reason is StopReason.END_TURN
    assert result.raw is raw


def test_openai_reports_usage():
    adapter, raw = _openai(
        NS(prompt_tokens=50, completion_tokens=8, prompt_tokens_details=NS(cached_tokens=32))
    )
    result = _call(adapter, "gpt-5")
    assert result.value == VALUE
    assert result.usage == Usage(input_tokens=50, output_tokens=8, cache_read_tokens=32)
    assert result.stop_reason is StopReason.END_TURN
    assert result.raw is raw


def test_openai_without_cache_details_leaves_cache_none():
    adapter, _ = _openai(NS(prompt_tokens=50, completion_tokens=8, prompt_tokens_details=None))
    assert _call(adapter, "gpt-5").usage == Usage(input_tokens=50, output_tokens=8)


# ── a response with no usage gives Usage() with every field None ─────────────


@pytest.mark.parametrize(
    "build, model",
    [(_anthropic, "claude-opus-4-8"), (_gemini, "gemini-3.5-flash"), (_openai, "gpt-5")],
    ids=["anthropic", "gemini", "openai"],
)
def test_missing_usage_gives_empty_usage(build, model):
    adapter, _ = build(None)
    result = _call(adapter, model)
    assert result.value == VALUE
    assert result.usage == Usage()


# ── the old call is unchanged ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "build, model",
    [(_anthropic, "claude-opus-4-8"), (_gemini, "gemini-3.5-flash"), (_openai, "gpt-5")],
    ids=["anthropic", "gemini", "openai"],
)
def test_adapter_generate_structured_json_returns_just_the_value(build, model):
    adapter, _ = build(None)
    assert adapter.generate_structured_json(model=model, prompt="?", schema=SCHEMA) == VALUE


def test_client_routes_both_calls():
    client = Client(key_provider=lambda spec: "k")
    adapter, _ = _openai(NS(prompt_tokens=50, completion_tokens=8))
    client._adapters[("openai", "k")] = adapter

    result = client.generate_structured(model="openai/gpt-5", schema=SCHEMA, prompt="?")
    assert result.value == VALUE
    assert result.usage == Usage(input_tokens=50, output_tokens=8)

    assert client.generate_structured_json(model="openai/gpt-5", schema=SCHEMA, prompt="?") == VALUE
