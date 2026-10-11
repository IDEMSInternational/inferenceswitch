"""Strict json_schema for OpenAI — translation and the round trip.

Offline: the SDK client is a fake that records each request.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from inferenceswitch import Client, StructuredOutputError, UnsupportedSchemaError
from inferenceswitch.adapters.openai_compat import OpenAICompatibleAdapter, openai_strict_for
from inferenceswitch.registry import default_registry
from inferenceswitch.schema import restore_openai_strict, to_openai_strict_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 3},
        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
        "tags": {"type": "array", "items": {"type": "string"}},
        "author": {"$ref": "#/$defs/Person"},
        "counts": {"type": "object", "additionalProperties": {"type": "integer"}},
        "mood": {"type": "string", "enum": ["up", "down"]},
    },
    "required": ["title", "rating"],
    "$defs": {
        "Person": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
            },
            "required": ["name"],
        }
    },
}


# ── translation ──────────────────────────────────────────────────────────────


def test_every_property_is_required_and_optional_ones_are_nullable():
    out = to_openai_strict_schema(SCHEMA)
    assert out["required"] == list(SCHEMA["properties"])
    assert out["additionalProperties"] is False
    assert out["properties"]["title"]["type"] == "string"  # required: unchanged
    assert out["properties"]["tags"]["type"] == ["array", "null"]
    assert out["properties"]["mood"]["enum"] == ["up", "down", None]
    assert out["properties"]["author"] == {
        "anyOf": [{"$ref": "#/$defs/Person"}, {"type": "null"}]
    }
    person = out["$defs"]["Person"]
    assert person["required"] == ["name", "email"]
    assert person["properties"]["email"] == {"type": ["string", "null"], "format": "email"}


def test_supported_constraints_are_kept_and_the_rest_noted():
    out = to_openai_strict_schema(SCHEMA)
    rating = out["properties"]["rating"]
    assert rating["minimum"] == 1 and rating["maximum"] == 5
    title = out["properties"]["title"]
    assert "minLength" not in title
    assert "minLength: 3" in title["description"]


def test_maps_become_key_value_arrays():
    counts = to_openai_strict_schema(SCHEMA)["properties"]["counts"]
    assert counts["type"] == ["array", "null"]  # optional, so nullable
    assert counts["items"]["required"] == ["key", "value"]


def test_single_allof_is_inlined_and_oneof_becomes_anyof():
    schema = {
        "type": "object",
        "properties": {
            "a": {"allOf": [{"$ref": "#/$defs/A"}], "description": "an A"},
            "b": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["a", "b"],
        "$defs": {
            "A": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        },
    }
    out = to_openai_strict_schema(schema)
    assert out["properties"]["a"] == {"$ref": "#/$defs/A"}
    assert out["properties"]["b"] == {"anyOf": [{"type": "string"}, {"type": "integer"}]}


def test_recursion_is_allowed():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}, "next": {"$ref": "#"}},
        "required": ["value"],
    }
    out = to_openai_strict_schema(schema)
    assert out["properties"]["next"] == {"anyOf": [{"$ref": "#"}, {"type": "null"}]}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": {"type": "string"}},               # root not an object
        {"type": "object", "properties": {"x": {"type": "object"}}},  # free-form object
        {"type": "object", "properties": {"x": {"allOf": [{"type": "string"}, {"minLength": 1}]}}},
        {"type": "object", "properties": {"x": {"$ref": "https://example.com/s.json"}}},
    ],
)
def test_inexpressible_schemas_raise(schema):
    with pytest.raises(UnsupportedSchemaError):
        to_openai_strict_schema(schema)


def test_restore_drops_nulls_for_optional_fields_and_folds_maps():
    data = {
        "title": "Dune",
        "rating": 5,
        "tags": None,
        "author": {"name": "Frank", "email": None},
        "counts": [{"key": "a", "value": 1}],
        "mood": None,
    }
    assert restore_openai_strict(data, SCHEMA) == {
        "title": "Dune",
        "rating": 5,
        "author": {"name": "Frank"},
        "counts": {"a": 1},
    }


def test_restore_keeps_a_null_the_schema_requires():
    schema = {
        "type": "object",
        "properties": {"x": {"type": ["string", "null"]}},
        "required": ["x"],
    }
    assert restore_openai_strict({"x": None}, schema) == {"x": None}


def test_restore_follows_recursion():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}, "next": {"$ref": "#"}},
        "required": ["value"],
    }
    data = {"value": "a", "next": {"value": "b", "next": None}}
    assert restore_openai_strict(data, schema) == {"value": "a", "next": {"value": "b"}}


# ── which providers get strict ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "provider, model, strict",
    [
        ("openai", "gpt-5.4-mini", True),
        ("openai", "gpt-4o-mini", True),
        ("groq", "openai/gpt-oss-120b", False),
        ("mistral", "mistral-small-2603", False),
        ("ollama", "qwen3:8b", False),
    ],
)
def test_strict_is_on_for_openai_only(provider, model, strict):
    config = default_registry().get(provider).capabilities.config
    assert openai_strict_for(config, model) is strict


# ── the round trip through the adapter ───────────────────────────────────────


class _RecordingOpenAI:
    def __init__(self, reply):
        self.requests: list[dict] = []
        response = NS(
            choices=[NS(message=NS(content=json.dumps(reply)), finish_reason="stop")],
            usage=NS(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None),
        )

        def create(**kwargs):
            self.requests.append(kwargs)
            return response

        self.chat = NS(completions=NS(create=create))


def _client(provider, reply):
    client = Client(key_provider=lambda spec: "k")
    fake = _RecordingOpenAI(reply)
    spec = default_registry().get(provider)
    client._adapters[(provider, "k")] = OpenAICompatibleAdapter(spec, "k", raw_client=fake)
    return client, fake


GOOD = {"title": "Dune", "rating": 5, "tags": None, "author": None, "counts": None, "mood": "up"}


def test_openai_request_is_strict_and_the_result_is_restored():
    client, fake = _client("openai", GOOD)
    value = client.generate_structured_json(model="openai/gpt-5.4-mini", schema=SCHEMA, prompt="?")
    sent = fake.requests[0]["response_format"]["json_schema"]
    assert sent["strict"] is True
    assert sent["schema"] == to_openai_strict_schema(SCHEMA)
    assert value == {"title": "Dune", "rating": 5, "mood": "up"}


def test_strict_result_is_checked_against_the_callers_schema():
    # minLength was dropped for strict mode, so it is checked afterwards.
    client, _ = _client("openai", {**GOOD, "title": "D"})
    with pytest.raises(StructuredOutputError, match="title"):
        client.generate_structured_json(model="openai/gpt-5.4-mini", schema=SCHEMA, prompt="?")


def test_inexpressible_schema_falls_back_to_non_strict():
    free_form = {"type": "object", "properties": {"meta": {"type": "object"}}}
    client, fake = _client("openai", {"meta": {"anything": 1}})
    value = client.generate_structured_json(
        model="openai/gpt-5.4-mini", schema=free_form, prompt="?"
    )
    sent = fake.requests[0]["response_format"]["json_schema"]
    assert "strict" not in sent
    assert sent["schema"] == free_form
    assert value == {"meta": {"anything": 1}}


def test_other_openai_compatible_providers_are_unchanged():
    client, fake = _client("mistral", GOOD)
    client.generate_structured_json(model="mistral/mistral-small-2603", schema=SCHEMA, prompt="?")
    sent = fake.requests[0]["response_format"]["json_schema"]
    assert "strict" not in sent and sent["schema"] == SCHEMA
