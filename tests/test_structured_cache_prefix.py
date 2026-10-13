"""``cache_prefix`` on structured calls: what each adapter sends, and the usage
that comes back.

Offline — each adapter is driven with a fake SDK client that records the
request, so every assertion is about what inferenceswitch *sends*.
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace as NS

import pytest

from inferenceswitch import Capability, Client, StructuredMode, Usage
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
PREFIX = "Book scope and run context. " * 50
EPHEMERAL = {"type": "ephemeral"}


class _FakeAnthropic:
    def __init__(self, usage=None):
        self.calls: list[dict] = []
        self.usage = usage
        self.content = [NS(type="text", text=json.dumps(VALUE))]
        self.stop_reason = "end_turn"
        self.messages = NS(stream=self._stream)

    def reply_tool(self, name="generate_json"):
        self.content = [NS(type="tool_use", id="toolu_1", name=name, input=VALUE)]
        self.stop_reason = "tool_use"

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        message = NS(
            content=self.content,
            stop_reason=self.stop_reason,
            stop_details=None,
            usage=self.usage,
        )

        class _Stream:
            def __enter__(inner):
                return inner

            def __exit__(inner, *exc):
                return False

            def get_final_message(inner):
                return message

        return _Stream()


class _FakeGenai:
    def __init__(self):
        self.calls: list[dict] = []
        self.models = NS(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return NS(
            candidates=[NS(content=NS(parts=[]), finish_reason=NS(name="STOP"))],
            text=json.dumps(VALUE),
            usage_metadata=None,
        )


class _FakeOpenAI:
    def __init__(self):
        self.calls: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return NS(
            choices=[NS(message=NS(content=json.dumps(VALUE)), finish_reason="stop")],
            usage=None,
        )


def _anthropic(spec=None, usage=None):
    fake = _FakeAnthropic(usage)
    spec = spec or default_registry().get("anthropic")
    return AnthropicAdapter(spec, "", raw_client=fake), fake


def _cache_controls(request: dict) -> list:
    """Every cache_control in a Messages request, wherever it sits."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "cache_control" in node:
                found.append(node["cache_control"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk({k: request.get(k) for k in ("system", "tools", "messages")})
    return found


# ── Anthropic: one breakpoint, on the prefix block ───────────────────────────


def test_anthropic_without_prefix_is_unchanged():
    a, fake = _anthropic()
    a.generate_structured(model="claude-opus-5-5", prompt="judge", schema=SCHEMA, system="be fair")
    sent = fake.calls[0]
    assert sent["messages"] == [{"role": "user", "content": "judge"}]
    assert _cache_controls(sent) == []


def test_anthropic_structured_outputs_marks_the_prefix():
    a, fake = _anthropic()
    result = a.generate_structured(
        model="claude-opus-5-5",
        prompt="judge this change",
        schema=SCHEMA,
        system="be fair",
        cache_prefix=PREFIX,
    )
    assert result.value == VALUE
    sent = fake.calls[0]
    assert sent["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PREFIX, "cache_control": EPHEMERAL},
                {"type": "text", "text": "judge this change"},
            ],
        }
    ]
    # The system prompt stays a plain string: the one breakpoint below it
    # caches it already.
    assert sent["system"] == "be fair"
    assert "output_config" in sent
    assert _cache_controls(sent) == [EPHEMERAL]


def test_anthropic_tool_fallback_marks_the_prefix():
    a, fake = _anthropic()
    fake.reply_tool()
    result = a.generate_structured(
        model="claude-opus-4-1-20250805", prompt="judge", schema=SCHEMA, cache_prefix=PREFIX
    )
    assert result.value == VALUE
    sent = fake.calls[0]
    assert sent["tools"][0]["name"] == "generate_json"
    assert sent["messages"][0]["content"][0] == {
        "type": "text",
        "text": PREFIX,
        "cache_control": EPHEMERAL,
    }
    assert sent["messages"][0]["content"][1] == {"type": "text", "text": "judge"}
    assert _cache_controls(sent) == [EPHEMERAL]


def test_anthropic_strict_tool_marks_the_prefix():
    spec = default_registry().get("anthropic")
    caps = dataclasses.replace(
        spec.capabilities,
        model_structured_output={"claude-opus-5-5": StructuredMode.STRICT_TOOL_USE},
    )
    a, fake = _anthropic(dataclasses.replace(spec, capabilities=caps))
    fake.reply_tool("verdict")
    a.generate_structured(
        model="claude-opus-5-5", prompt="p", schema=SCHEMA, tool_name="verdict", cache_prefix=PREFIX
    )
    sent = fake.calls[0]
    assert sent["tools"][0]["strict"] is True
    assert sent["messages"][0]["content"][0]["cache_control"] == EPHEMERAL
    assert _cache_controls(sent) == [EPHEMERAL]


def test_anthropic_empty_prompt_sends_only_the_prefix():
    # Anthropic rejects an empty text block, so none is sent.
    a, fake = _anthropic()
    a.generate_structured(model="claude-opus-5-5", prompt="", schema=SCHEMA, cache_prefix=PREFIX)
    assert fake.calls[0]["messages"][0]["content"] == [
        {"type": "text", "text": PREFIX, "cache_control": EPHEMERAL}
    ]


def test_anthropic_empty_prefix_is_no_prefix():
    a, fake = _anthropic()
    a.generate_structured(model="claude-opus-5-5", prompt="p", schema=SCHEMA, cache_prefix="")
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "p"}]


# ── usage: the write, then the read ──────────────────────────────────────────


def test_client_reports_cache_write_then_read():
    write = NS(
        input_tokens=12, output_tokens=5, cache_creation_input_tokens=72000,
        cache_read_input_tokens=0,
    )
    read = NS(
        input_tokens=12, output_tokens=5, cache_creation_input_tokens=0,
        cache_read_input_tokens=72000,
    )
    client = Client(key_provider=lambda spec: "k")
    adapter, fake = _anthropic()
    client._adapters[("anthropic", "k")] = adapter

    fake.usage = write
    first = client.generate_structured(
        model="anthropic/claude-opus-5-5", schema=SCHEMA, prompt="try 1", cache_prefix=PREFIX
    )
    fake.usage = read
    second = client.generate_structured(
        model="anthropic/claude-opus-5-5", schema=SCHEMA, prompt="try 2", cache_prefix=PREFIX
    )
    assert first.usage == Usage(
        input_tokens=12, output_tokens=5, cache_write_tokens=72000, cache_read_tokens=0
    )
    assert second.usage == Usage(
        input_tokens=12, output_tokens=5, cache_write_tokens=0, cache_read_tokens=72000
    )
    # Same prefix block both times; only the tail differs.
    assert fake.calls[0]["messages"][0]["content"][0] == fake.calls[1]["messages"][0]["content"][0]


def test_client_generate_structured_json_forwards_the_prefix():
    client = Client(key_provider=lambda spec: "k")
    adapter, fake = _anthropic()
    client._adapters[("anthropic", "k")] = adapter
    value = client.generate_structured_json(
        model="anthropic/claude-opus-5-5", schema=SCHEMA, prompt="p", cache_prefix=PREFIX
    )
    assert value == VALUE
    assert fake.calls[0]["messages"][0]["content"][0]["cache_control"] == EPHEMERAL


# ── elsewhere: prefix first, no marker, no error ─────────────────────────────


def test_gemini_sends_the_prefix_as_the_first_part():
    fake = _FakeGenai()
    client = Client(key_provider=lambda spec: "k")
    client._adapters[("gemini", "k")] = GeminiAdapter(
        default_registry().get("gemini"), "", raw_client=fake
    )
    # Gemini has no inline breakpoints, and the call still goes through.
    assert not client.supports(Capability.EXPLICIT_PROMPT_CACHING, "gemini")
    value = client.generate_structured_json(
        model="gemini/gemini-3.5-flash", schema=SCHEMA, prompt="judge", cache_prefix=PREFIX
    )
    assert value == VALUE
    assert fake.calls[0]["contents"] == [
        {"role": "user", "parts": [{"text": PREFIX}, {"text": "judge"}]}
    ]


def test_gemini_without_prefix_is_unchanged():
    fake = _FakeGenai()
    adapter = GeminiAdapter(default_registry().get("gemini"), "", raw_client=fake)
    adapter.generate_structured(model="gemini-3.5-flash", prompt="judge", schema=SCHEMA)
    assert fake.calls[0]["contents"] == "judge"


@pytest.mark.parametrize("provider", ["openai", "deepseek"])
def test_openai_compatible_puts_the_prefix_first(provider):
    fake = _FakeOpenAI()
    adapter = OpenAICompatibleAdapter(default_registry().get(provider), "", raw_client=fake)
    adapter.generate_structured(model="m", prompt="judge", schema=SCHEMA, cache_prefix=PREFIX)
    user_turn = fake.calls[0]["messages"][-1]
    assert user_turn["role"] == "user"
    assert user_turn["content"].startswith(f"{PREFIX}\n\njudge")


def test_openai_compatible_without_prefix_is_unchanged():
    fake = _FakeOpenAI()
    adapter = OpenAICompatibleAdapter(default_registry().get("openai"), "", raw_client=fake)
    adapter.generate_structured(model="gpt-5", prompt="judge", schema=SCHEMA)
    assert fake.calls[0]["messages"][-1] == {"role": "user", "content": "judge"}
