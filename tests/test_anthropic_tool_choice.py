"""Anthropic forced tool choice in ``chat()``: per-model, and emulated where the
model rejects it.

Offline — the adapter is driven with a fake SDK client that records each request
and replays a scripted response, so every assertion is about what inferenceswitch
*sends* and how it reads what comes back.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from inferenceswitch import StopReason, Tool, ToolChoice, ToolChoiceError, system, user
from inferenceswitch.adapters.anthropic import AnthropicAdapter
from inferenceswitch.registry import default_registry

#: The models that return a 400 for tool_choice "tool" and "any".
NO_FORCED_CHOICE = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"]

WEATHER = Tool("get_weather", "Current weather for a city.", {"type": "object"})
TIME = Tool("get_time", "Current time in a city.", {"type": "object"})


class _FakeAnthropic:
    """A stand-in anthropic.Anthropic whose next response is settable."""

    def __init__(self):
        self.calls: list[dict] = []
        self.content: list = [SimpleNamespace(type="text", text="It is sunny.")]
        self.stop_reason = "end_turn"
        self.messages = SimpleNamespace(stream=self._stream)

    def reply_tool(self, name: str, value=None):
        self.content = [SimpleNamespace(type="tool_use", id="toolu_1", name=name, input=value or {})]
        self.stop_reason = "tool_use"

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(
            content=self.content, stop_reason=self.stop_reason, usage=None
        )

        class _Stream:
            def __enter__(inner):
                return inner

            def __exit__(inner, *exc):
                return False

            def get_final_message(inner):
                return message

        return _Stream()


def _adapter(spec=None):
    fake = _FakeAnthropic()
    spec = spec or default_registry().get("anthropic")
    return AnthropicAdapter(spec, "", raw_client=fake), fake


def _chat(adapter, model, **kwargs):
    kwargs.setdefault("tools", [WEATHER, TIME])
    return adapter.chat(model=model, messages=[user("weather in Paris?")], **kwargs)


# ── the registry: which models honor a forced choice ──────────────────────────


@pytest.mark.parametrize(
    "model",
    NO_FORCED_CHOICE
    + [
        "claude-opus-5-5-20260901",       # dated snapshot of a point release
        "anthropic.claude-sonnet-5-5",    # Bedrock-prefixed
        "claude-opus-7",                  # not in the table: the newest behavior
    ],
)
def test_models_without_forced_tool_choice(model):
    caps = default_registry().get("anthropic").capabilities
    assert caps.forced_tool_choice_for(model) is False


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "anthropic.claude-opus-4-7",
        "claude-3-5-haiku-20241022",
    ],
)
def test_older_models_keep_forced_tool_choice(model):
    caps = default_registry().get("anthropic").capabilities
    assert caps.forced_tool_choice_for(model) is True


def test_other_providers_default_to_forced_tool_choice():
    registry = default_registry()
    assert registry.get("openai").capabilities.forced_tool_choice_for("gpt-5") is True
    assert registry.get("gemini").capabilities.forced_tool_choice_for("gemini-3-flash") is True


# ── older models: the forced choice goes on the wire unchanged ────────────────


def test_force_tool_is_forced_on_older_model():
    adapter, fake = _adapter()
    fake.reply_tool("get_weather")
    response = _chat(adapter, "claude-opus-4-8", force_tool="get_weather", system="Be brief.")
    sent = fake.calls[0]
    assert sent["tool_choice"] == {"type": "tool", "name": "get_weather"}
    assert sent["system"] == "Be brief."
    assert response.tool_calls[0].name == "get_weather"


def test_required_is_any_on_older_model():
    adapter, fake = _adapter()
    fake.reply_tool("get_time")
    _chat(adapter, "claude-sonnet-4-6", tool_choice=ToolChoice.REQUIRED)
    sent = fake.calls[0]
    assert sent["tool_choice"] == {"type": "any"}
    assert "system" not in sent


def test_older_model_without_a_tool_call_still_returns():
    # Server-enforced there; a turn without the call (e.g. cut off at max_tokens)
    # is returned as before, not raised.
    adapter, fake = _adapter()
    fake.stop_reason = "max_tokens"
    response = _chat(adapter, "claude-opus-5", force_tool="get_weather")
    assert response.stop_reason is StopReason.MAX_TOKENS
    assert response.tool_calls == []


# ── newer models: auto + an instruction ───────────────────────────────────────


@pytest.mark.parametrize("model", NO_FORCED_CHOICE)
def test_force_tool_becomes_auto_with_instruction(model):
    adapter, fake = _adapter()
    fake.reply_tool("get_weather", {"city": "Paris"})
    response = _chat(adapter, model, force_tool="get_weather", system="Be brief.")
    sent = fake.calls[0]
    assert sent["tool_choice"] == {"type": "auto"}
    # The caller's system prompt comes first, the instruction after it.
    assert sent["system"] == "Be brief.\n\nRespond by calling the `get_weather` tool."
    assert [t["name"] for t in sent["tools"]] == ["get_weather", "get_time"]
    assert response.tool_calls[0].input == {"city": "Paris"}


def test_force_tool_instruction_is_the_whole_system_prompt_when_none_given():
    adapter, fake = _adapter()
    fake.reply_tool("get_weather")
    _chat(adapter, "claude-opus-5-5", force_tool="get_weather")
    assert fake.calls[0]["system"] == "Respond by calling the `get_weather` tool."


def test_force_tool_instruction_follows_system_messages():
    adapter, fake = _adapter()
    fake.reply_tool("get_weather")
    adapter.chat(
        model="claude-opus-5-5",
        messages=[system("Use metric units."), user("weather in Paris?")],
        tools=[WEATHER],
        force_tool="get_weather",
        system="Be brief.",
    )
    assert fake.calls[0]["system"] == (
        "Be brief.\n\nUse metric units.\n\nRespond by calling the `get_weather` tool."
    )


def test_required_becomes_auto_with_instruction_naming_every_tool():
    adapter, fake = _adapter()
    fake.reply_tool("get_time")
    _chat(adapter, "claude-sonnet-5-5", tool_choice=ToolChoice.REQUIRED)
    sent = fake.calls[0]
    assert sent["tool_choice"] == {"type": "auto"}
    assert sent["system"] == (
        "Respond by calling at least one of these tools: `get_weather`, `get_time`."
    )


@pytest.mark.parametrize("choice", [ToolChoice.AUTO, ToolChoice.NONE])
def test_unforced_choices_are_unchanged_on_newer_model(choice):
    adapter, fake = _adapter()
    response = _chat(adapter, "claude-fable-5-1", tool_choice=choice)
    sent = fake.calls[0]
    assert sent["tool_choice"] == {"type": choice.value}
    assert "system" not in sent
    assert response.text == "It is sunny."   # no tool call, and none was required


# ── newer models: a turn that ignored the instruction ─────────────────────────


def test_force_tool_without_the_call_raises_with_the_response():
    adapter, fake = _adapter()
    with pytest.raises(ToolChoiceError) as excinfo:
        _chat(adapter, "claude-opus-5-5", force_tool="get_weather")
    err = excinfo.value
    assert "get_weather" in str(err)
    assert err.response.text == "It is sunny."
    assert err.response.stop_reason is StopReason.END_TURN
    assert err.context == {
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "force_tool": "get_weather",
        "tool_choice": "auto",
        "stop_reason": "end_turn",
        "tool_calls": [],
    }


def test_force_tool_with_a_different_tool_raises():
    adapter, fake = _adapter()
    fake.reply_tool("get_time")
    with pytest.raises(ToolChoiceError) as excinfo:
        _chat(adapter, "claude-mythos-5-1", force_tool="get_weather")
    assert excinfo.value.context["tool_calls"] == ["get_time"]
    assert excinfo.value.response.tool_calls[0].name == "get_time"


def test_required_without_any_call_raises():
    adapter, fake = _adapter()
    with pytest.raises(ToolChoiceError) as excinfo:
        _chat(adapter, "claude-sonnet-5-5", tool_choice=ToolChoice.REQUIRED)
    assert excinfo.value.context["tool_choice"] == "required"
    assert excinfo.value.context["force_tool"] is None


def test_truncated_turn_without_the_call_raises_with_its_stop_reason():
    adapter, fake = _adapter()
    fake.stop_reason = "max_tokens"
    with pytest.raises(ToolChoiceError) as excinfo:
        _chat(adapter, "claude-opus-5-5", force_tool="get_weather")
    assert excinfo.value.context["stop_reason"] == "max_tokens"


# ── the table is the control surface ──────────────────────────────────────────


def test_per_model_override_restores_forced_choice():
    spec = default_registry().get("anthropic")
    caps = dataclasses.replace(
        spec.capabilities,
        model_forced_tool_choice={
            **spec.capabilities.model_forced_tool_choice,
            "claude-opus-5-5": True,
        },
    )
    adapter, fake = _adapter(dataclasses.replace(spec, capabilities=caps))
    fake.reply_tool("get_weather")
    _chat(adapter, "claude-opus-5-5", force_tool="get_weather")
    assert fake.calls[0]["tool_choice"] == {"type": "tool", "name": "get_weather"}
    assert "system" not in fake.calls[0]
