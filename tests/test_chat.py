"""Tool-calling: message translation, response normalization, discovery, gating.

All offline — encode functions are pure, decode functions are exercised against
SimpleNamespace fakes shaped like each provider's SDK response, so no SDK or
network is needed.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from inferenceswitch import (
    Capability,
    LLMClient,
    LLMResponse,
    Message,
    StopReason,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolUse,
    UnsupportedCapabilityError,
    system,
    user,
)
from inferenceswitch.adapters.anthropic import (
    decode_anthropic_response,
    encode_anthropic_messages,
)
from inferenceswitch.adapters.base import split_system
from inferenceswitch.adapters.gemini import decode_gemini_response, encode_gemini_contents
from inferenceswitch.adapters.openai_compat import (
    decode_openai_response,
    encode_openai_messages,
)


@pytest.fixture
def convo():
    return [
        user("weather in Paris?"),
        Message("assistant", [ToolUse("call_1", "get_weather", {"city": "Paris"})]),
        Message("user", [ToolResult("call_1", "20C", name="get_weather")]),
    ]


# ── system extraction ─────────────────────────────────────────────────────────


def test_split_system_merges_param_and_message():
    combined, rest = split_system([system("A"), user("hi")], "B")
    assert combined == "B\n\nA"
    assert [m.role for m in rest] == ["user"]


# ── encode: same conversation, three native shapes ────────────────────────────


def test_encode_openai(convo):
    out = encode_openai_messages(convo)
    assert out[0] == {"role": "user", "content": "weather in Paris?"}
    assert out[1]["role"] == "assistant"
    assert out[1]["tool_calls"][0]["id"] == "call_1"
    assert out[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    # tool result becomes its own "tool" message keyed by call id
    assert out[2] == {"role": "tool", "tool_call_id": "call_1", "content": "20C"}


def test_encode_anthropic(convo):
    out = encode_anthropic_messages(convo)
    assert out[1]["content"][0]["type"] == "tool_use"
    assert out[1]["content"][0]["id"] == "call_1"
    result_block = out[2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "call_1"
    assert result_block["content"] == "20C"


def test_encode_gemini_uses_model_role_and_function_name(convo):
    out = encode_gemini_contents(convo)
    assert out[1]["role"] == "model"  # assistant -> model
    assert out[1]["parts"][0]["function_call"]["name"] == "get_weather"
    # Gemini keys results by function NAME, not id
    fr = out[2]["parts"][0]["function_response"]
    assert fr["name"] == "get_weather"
    assert fr["response"] == {"result": "20C"}


# ── decode: three native responses -> one normalized shape ────────────────────


def test_decode_openai():
    raw = NS(
        choices=[
            NS(
                message=NS(
                    content="checking",
                    tool_calls=[
                        NS(id="call_1", function=NS(name="get_weather", arguments='{"city":"Paris"}'))
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=NS(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=NS(cached_tokens=2)),
    )
    resp = decode_openai_response(raw)
    assert resp.text == "checking"
    assert resp.stop_reason is StopReason.TOOL_USE
    assert resp.wants_tools
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].input == {"city": "Paris"}
    assert resp.usage.input_tokens == 10
    assert resp.usage.cache_read_tokens == 2


def test_decode_anthropic():
    raw = NS(
        content=[
            NS(type="text", text="sure"),
            NS(type="tool_use", id="tu_1", name="get_weather", input={"city": "Paris"}),
        ],
        stop_reason="tool_use",
        usage=NS(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=2,
            cache_creation_input_tokens=0,
        ),
    )
    resp = decode_anthropic_response(raw)
    assert resp.text == "sure"
    assert resp.stop_reason is StopReason.TOOL_USE
    assert resp.tool_calls[0].id == "tu_1"
    assert resp.tool_calls[0].input == {"city": "Paris"}
    assert resp.usage.cache_read_tokens == 2


def test_decode_gemini_synthesizes_ids_and_infers_tool_use():
    raw = NS(
        candidates=[
            NS(
                content=NS(
                    parts=[
                        NS(text="ok", function_call=None),
                        NS(text=None, function_call=NS(name="get_weather", args={"city": "Paris"})),
                    ]
                ),
                finish_reason=NS(name="STOP"),  # STOP, but a tool call is present
            )
        ],
        usage_metadata=NS(
            prompt_token_count=10,
            candidates_token_count=5,
            cached_content_token_count=0,
            thoughts_token_count=3,
        ),
    )
    resp = decode_gemini_response(raw)
    assert resp.text == "ok"
    # tool call present -> normalized to TOOL_USE even though Gemini said STOP
    assert resp.stop_reason is StopReason.TOOL_USE
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].id.startswith("call_get_weather")
    assert resp.usage.reasoning_tokens == 3


# ── discovery: "which models can serve this call?" ────────────────────────────


def test_discovery_lists_capable_providers_and_models():
    c = LLMClient()
    provs = c.providers_for(Capability.TOOL_CALLING)
    assert {"anthropic", "openai", "gemini", "ollama"} <= set(provs)

    models = c.models_for(Capability.TOOL_CALLING)
    assert "anthropic/claude-opus-4-8" in models

    # combined requirements narrow the field
    vision_pdf = set(c.providers_for(Capability.VISION, Capability.PDF_INPUT))
    assert vision_pdf == {"anthropic", "gemini"}

    reasoning = c.providers_for(Capability.REASONING_EFFORT)
    assert "anthropic" in reasoning and "groq" not in reasoning


# ── gating: tools on a provider that can't -> loud failure ────────────────────


def test_chat_with_tools_requires_capability():
    from inferenceswitch.capabilities import Capabilities, SchemaDialect, StructuredMode
    from inferenceswitch.registry import KIND_OPENAI, ProviderSpec, Registry

    no_tools = ProviderSpec(
        name="notools",
        kind=KIND_OPENAI,
        capabilities=Capabilities(
            StructuredMode.JSON_OBJECT_BEST_EFFORT, SchemaDialect.OPENAI, features=frozenset()
        ),
        api_key_env=None,
        is_local=True,
    )
    client = LLMClient(registry=Registry([no_tools]))
    # Raises at the capability gate, before any adapter/SDK is built.
    with pytest.raises(UnsupportedCapabilityError):
        client.chat(
            model="x",
            provider="notools",
            messages=[user("hi")],
            tools=[Tool("t", "d", {"type": "object"})],
            tool_choice=ToolChoice.AUTO,
        )


# ── run_tools: the auto loop (scripted, no SDK) ───────────────────────────────


def test_run_tools_drives_the_loop(monkeypatch):
    client = LLMClient()
    turns = {"n": 0}

    def fake_chat(*, model, messages, tools, **kw) -> LLMResponse:
        turns["n"] += 1
        if turns["n"] == 1:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall("id1", "add", {"a": 2, "b": 3})],
                stop_reason=StopReason.TOOL_USE,
            )
        # Second turn: the model has seen the tool result and wraps up.
        return LLMResponse(text="The sum is 5.", stop_reason=StopReason.END_TURN)

    monkeypatch.setattr(client, "chat", fake_chat)

    seen: dict = {}

    def add(inp: dict) -> int:
        seen.update(inp)
        return inp["a"] + inp["b"]

    final = client.run_tools(
        model="x",
        messages=[user("add 2 and 3")],
        tools=[Tool("add", "adds two numbers", {"type": "object"})],
        handlers={"add": add},
    )
    assert final.text == "The sum is 5."
    assert seen == {"a": 2, "b": 3}   # handler actually ran
    assert turns["n"] == 2            # looped exactly twice, then stopped
