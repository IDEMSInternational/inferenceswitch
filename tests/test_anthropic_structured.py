"""Anthropic structured JSON: the mechanism per model, and its failure modes.

Offline — the adapter is driven with a fake SDK client that records each request
and replays a scripted response, so every assertion is about what inferenceswitch
*sends* and how it reads what comes back.
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from inferenceswitch import StructuredMode, StructuredOutputError, UnsupportedSchemaError
from inferenceswitch.adapters.anthropic import AnthropicAdapter
from inferenceswitch.registry import default_registry


class _FakeAnthropic:
    """A stand-in anthropic.Anthropic whose next response is settable."""

    def __init__(self):
        self.calls: list[dict] = []
        self.content: list = []
        self.stop_reason = "end_turn"
        self.stop_details = None
        self.messages = SimpleNamespace(stream=self._stream)

    def reply_text(self, value):
        self.content = [SimpleNamespace(type="text", text=json.dumps(value))]

    def reply_tool(self, value, name="generate_json"):
        self.content = [
            SimpleNamespace(type="text", text="Here you go."),
            SimpleNamespace(type="tool_use", id="toolu_1", name=name, input=value),
        ]
        self.stop_reason = "tool_use"

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(
            content=self.content,
            stop_reason=self.stop_reason,
            stop_details=self.stop_details,
            usage=SimpleNamespace(output_tokens=7),
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


@pytest.fixture
def adapter():
    return _adapter()


#: Shaped like terraso-planning's verdict models: an enum, a bounded score, an
#: optional field, a nested $ref'd object and a list.
VERDICT = {
    "type": "object",
    "title": "Verdict",
    "properties": {
        "changed": {"type": "array", "items": {"$ref": "#/$defs/Field"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 400},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "evidence": {"$ref": "#/$defs/Evidence"},
    },
    "required": ["changed", "confidence", "rationale", "evidence"],
    "$defs": {
        "Field": {"type": "string", "enum": ["direction", "magnitude", "timing"]},
        "Evidence": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "quotes": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            },
            "required": ["source", "quotes"],
        },
    },
}

GOOD_VERDICT = {
    "changed": ["direction"],
    "confidence": 0.85,
    "rationale": "The trend reversed.",
    "evidence": {"source": "report", "quotes": ["q1"]},
}


# ── mechanism per model ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "anthropic.claude-opus-5-5",
        "claude-something-new",
    ],
)
def test_current_and_unknown_models_use_structured_outputs(model):
    caps = default_registry().get("anthropic").capabilities
    assert caps.structured_output_for(model) is StructuredMode.OUTPUT_CONFIG_JSON_SCHEMA


@pytest.mark.parametrize(
    "model",
    [
        "claude-3-5-haiku-20241022",
        "claude-3-7-sonnet-latest",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-0",
    ],
)
def test_models_without_structured_outputs_fall_back_to_a_tool(model):
    caps = default_registry().get("anthropic").capabilities
    assert caps.structured_output_for(model) is StructuredMode.TOOL_USE


def test_forced_tool_use_is_a_deprecated_alias():
    assert StructuredMode.FORCED_TOOL_USE is StructuredMode.TOOL_USE


# ── structured outputs (output_config.format) ────────────────────────────────


def test_structured_outputs_request_shape(adapter):
    a, fake = adapter
    fake.reply_text(GOOD_VERDICT)
    result = a.generate_structured_json(
        model="claude-opus-5-5", prompt="judge", schema=VERDICT, system="be fair"
    )
    assert result == GOOD_VERDICT

    call = fake.calls[-1]
    assert "tools" not in call and "tool_choice" not in call
    assert call["system"] == "be fair"
    fmt = call["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    sent = fmt["schema"]
    # objects closed, required kept as the caller gave it
    assert sent["additionalProperties"] is False
    assert sent["required"] == VERDICT["required"]
    assert sent["$defs"]["Evidence"]["additionalProperties"] is False
    # unsupported constraints stripped (and noted for the model)
    confidence = sent["properties"]["confidence"]
    assert "minimum" not in confidence and "maximum" not in confidence
    assert "minimum: 0" in confidence["description"]
    assert "maxLength" not in sent["properties"]["rationale"]
    assert "maxItems" not in sent["$defs"]["Evidence"]["properties"]["quotes"]
    # annotations dropped, refs left for the API to resolve
    assert "default" not in sent["properties"]["note"]
    assert sent["properties"]["evidence"] == {"$ref": "#/$defs/Evidence"}
    # the caller's schema is untouched
    assert "additionalProperties" not in VERDICT


def test_text_block_after_thinking_is_parsed(adapter):
    a, fake = adapter
    fake.content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=json.dumps(GOOD_VERDICT)),
    ]
    assert a.generate_structured_json(model="claude-opus-5-5", prompt="p", schema=VERDICT) == GOOD_VERDICT


def test_constraints_the_dialect_drops_are_validated_client_side(adapter):
    a, fake = adapter
    fake.reply_text({**GOOD_VERDICT, "confidence": 1.5})
    with pytest.raises(StructuredOutputError) as excinfo:
        a.generate_structured_json(model="claude-opus-5-5", prompt="p", schema=VERDICT)
    assert "$.confidence: 1.5 is greater than the maximum 1" in str(excinfo.value)
    assert excinfo.value.context["schema_errors"]


def test_stray_key_is_not_unwrapped_under_constrained_decoding(adapter):
    # Constrained decoding cannot produce this; if it somehow does, it is an error.
    a, fake = adapter
    fake.reply_text({"$PARAMETER_NAME": GOOD_VERDICT})
    with pytest.raises(StructuredOutputError):
        a.generate_structured_json(model="claude-opus-5-5", prompt="p", schema=VERDICT)


def test_map_fields_round_trip(adapter):
    a, fake = adapter
    schema = {
        "type": "object",
        "properties": {"labels": {"type": "object", "additionalProperties": {"type": "string"}}},
        "required": ["labels"],
    }
    fake.reply_text({"labels": [{"key": "cat", "value": "gato"}]})
    result = a.generate_structured_json(model="claude-sonnet-5-5", prompt="p", schema=schema)
    assert result == {"labels": {"cat": "gato"}}
    sent = fake.calls[-1]["output_config"]["format"]["schema"]
    assert sent["properties"]["labels"]["type"] == "array"


def test_invalid_json_text_raises(adapter):
    a, fake = adapter
    fake.content = [SimpleNamespace(type="text", text='{"changed": [')]
    with pytest.raises(StructuredOutputError, match="not valid JSON"):
        a.generate_structured_json(model="claude-opus-5-5", prompt="p", schema=VERDICT)


def test_missing_text_block_raises(adapter):
    a, fake = adapter
    fake.content = []
    with pytest.raises(StructuredOutputError, match="no text block"):
        a.generate_structured_json(model="claude-opus-5-5", prompt="p", schema=VERDICT)


def test_refusal_raises_with_its_category(adapter):
    a, fake = adapter
    fake.content = [SimpleNamespace(type="text", text="I can't help with that.")]
    fake.stop_reason = "refusal"
    fake.stop_details = SimpleNamespace(type="refusal", category="cyber", explanation=None)
    with pytest.raises(StructuredOutputError) as excinfo:
        a.generate_structured_json(model="claude-opus-5-5", prompt="p", schema=VERDICT)
    assert "refusal" in str(excinfo.value)
    assert excinfo.value.context["refusal_category"] == "cyber"
    assert excinfo.value.context["stop_reason"] == "refusal"


def test_max_tokens_raises_before_parsing(adapter):
    a, fake = adapter
    fake.content = [SimpleNamespace(type="text", text='{"changed": ["dir')]
    fake.stop_reason = "max_tokens"
    with pytest.raises(StructuredOutputError) as excinfo:
        a.generate_structured_json(
            model="claude-opus-5-5", prompt="p", schema=VERDICT, max_tokens=50
        )
    assert "truncated" in str(excinfo.value)
    assert excinfo.value.context["max_tokens"] == 50
    assert excinfo.value.context["mode"] == "output_config_json_schema"


def test_recursive_schema_is_rejected_before_sending(adapter):
    a, fake = adapter
    tree = {
        "$ref": "#/$defs/Node",
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
            }
        },
    }
    with pytest.raises(UnsupportedSchemaError, match="Recursive"):
        a.generate_structured_json(model="claude-opus-5-5", prompt="p", schema=tree)
    assert fake.calls == []


# ── tool fallback ────────────────────────────────────────────────────────────


def test_tool_fallback_request_shape(adapter):
    a, fake = adapter
    fake.reply_tool(GOOD_VERDICT)
    result = a.generate_structured_json(
        model="claude-opus-4-1-20250805", prompt="judge", schema=VERDICT, system="be fair"
    )
    assert result == GOOD_VERDICT

    call = fake.calls[-1]
    assert "output_config" not in call
    (tool,) = call["tools"]
    assert tool["name"] == "generate_json"
    assert "strict" not in tool
    assert tool["input_schema"] is VERDICT  # permissive dialect: sent as given
    # never forced: the newest models reject forced tool_choice with a 400
    assert call["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert call["system"].startswith("be fair\n\n")
    assert "`generate_json` tool" in call["system"]


def test_strict_tool_mode_sends_a_strict_translated_tool():
    spec = default_registry().get("anthropic")
    caps = dataclasses.replace(
        spec.capabilities,
        model_structured_output={"claude-opus-5-5": StructuredMode.STRICT_TOOL_USE},
    )
    a, fake = _adapter(dataclasses.replace(spec, capabilities=caps))
    fake.reply_tool(GOOD_VERDICT, name="verdict")
    result = a.generate_structured_json(
        model="claude-opus-5-5", prompt="p", schema=VERDICT, tool_name="verdict"
    )
    assert result == GOOD_VERDICT

    call = fake.calls[-1]
    (tool,) = call["tools"]
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False
    assert call["tool_choice"]["type"] == "auto"
    assert call["system"].startswith("Respond only by calling the `verdict` tool")


def test_missing_tool_call_on_the_fallback_raises(adapter):
    a, fake = adapter
    fake.content = [SimpleNamespace(type="text", text="The verdict is: changed.")]
    fake.stop_reason = "end_turn"
    with pytest.raises(StructuredOutputError, match="did not call the 'generate_json' tool") as excinfo:
        a.generate_structured_json(model="claude-3-7-sonnet-latest", prompt="p", schema=VERDICT)
    assert excinfo.value.context["tool_name"] == "generate_json"
    assert excinfo.value.context["mode"] == "tool_use"


def test_stray_key_wrapper_is_unwrapped_on_the_non_strict_tool(adapter):
    # terraso-planning#308: the answer nested under a tool-parameter template name.
    a, fake = adapter
    fake.reply_tool({"$PARAMETER_NAME": GOOD_VERDICT})
    assert (
        a.generate_structured_json(model="claude-opus-4-1", prompt="p", schema=VERDICT)
        == GOOD_VERDICT
    )


def test_wrapper_that_does_not_validate_is_an_error(adapter):
    a, fake = adapter
    fake.reply_tool({"parameter": {"changed": ["sideways"]}})
    with pytest.raises(StructuredOutputError, match="does not match the schema"):
        a.generate_structured_json(model="claude-opus-4-1", prompt="p", schema=VERDICT)


def test_refusal_on_the_tool_fallback_raises(adapter):
    a, fake = adapter
    fake.content = []
    fake.stop_reason = "refusal"
    with pytest.raises(StructuredOutputError, match="refusal"):
        a.generate_structured_json(model="claude-opus-4-1", prompt="p", schema=VERDICT)


def test_max_tokens_on_the_tool_fallback_raises(adapter):
    a, fake = adapter
    fake.reply_tool({"changed": []})
    fake.stop_reason = "max_tokens"
    with pytest.raises(StructuredOutputError, match="truncated"):
        a.generate_structured_json(model="claude-opus-4-1", prompt="p", schema=VERDICT)
