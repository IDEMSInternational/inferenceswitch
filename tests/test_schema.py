"""Schema-dialect translation tests — the divergences a plain OpenAI shim cannot
handle — and the client-side validator."""
from __future__ import annotations

import pytest

from inferenceswitch import UnsupportedSchemaError
from inferenceswitch.schema import (
    restore_gemini_dicts,
    schema_errors,
    to_anthropic_schema,
    to_gemini_schema,
)


def test_dict_field_becomes_key_value_array_and_back():
    schema = {
        "type": "object",
        "properties": {
            "translation_map": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "term -> translation",
            },
            "name": {"type": "string"},
        },
    }

    gemini = to_gemini_schema(schema)
    tm = gemini["properties"]["translation_map"]
    assert tm["type"] == "array"
    assert tm["items"]["properties"]["key"]["type"] == "string"
    assert tm["items"]["properties"]["value"] == {"type": "string"}
    # non-dict fields are untouched
    assert gemini["properties"]["name"] == {"type": "string"}

    # What Gemini would return under the translated schema:
    gemini_output = {
        "translation_map": [
            {"key": "cat", "value": "gato"},
            {"key": "dog", "value": "perro"},
        ],
        "name": "es",
    }
    restored = restore_gemini_dicts(gemini_output, schema)
    assert restored == {"translation_map": {"cat": "gato", "dog": "perro"}, "name": "es"}


def test_restoration_resolves_refs_in_nested_defs():
    # Mimics a Pydantic model_json_schema() with a $ref + a dict field in the def.
    schema = {
        "type": "object",
        "properties": {"row": {"$ref": "#/$defs/Row"}},
        "$defs": {
            "Row": {
                "type": "object",
                "properties": {
                    "attrs": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    }
                },
            }
        },
    }
    data = {"row": {"attrs": [{"key": "color", "value": "red"}]}}
    assert restore_gemini_dicts(data, schema) == {"row": {"attrs": {"color": "red"}}}


def test_non_dict_lists_are_left_alone():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }
    data = {"tags": ["a", "b"]}
    assert restore_gemini_dicts(data, schema) == {"tags": ["a", "b"]}


def test_optional_dict_field_round_trips():
    # Optional[dict[str, str]] as Pydantic emits it. The dict-ness lives in an
    # anyOf branch, not on the property node — the shape that made restoration a
    # silent no-op across a whole model whose dict fields were all Optional.
    schema = {
        "type": "object",
        "properties": {
            "attrs": {
                "anyOf": [
                    {"type": "object", "additionalProperties": {"type": "string"}},
                    {"type": "null"},
                ],
                "default": None,
            }
        },
    }

    gemini = to_gemini_schema(schema)
    assert gemini["properties"]["attrs"]["anyOf"][0]["type"] == "array"

    data = {"attrs": [{"key": "region", "value": "north-west"}]}
    assert restore_gemini_dicts(data, schema) == {"attrs": {"region": "north-west"}}


def test_optional_dict_nested_in_a_list_of_objects_round_trips():
    # The hard case: an Optional dict reachable only through $refs *and* array
    # items — `groups[].items[].metadata`. Restoration has to see through both
    # the ref indirection and the array nesting to find the field, and must
    # leave a null sibling alone.
    schema = {
        "type": "object",
        "properties": {
            "groups": {"type": "array", "items": {"$ref": "#/$defs/Group"}}
        },
        "$defs": {
            "Group": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Item"},
                    }
                },
            },
            "Item": {
                "type": "object",
                "properties": {
                    "metadata": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                    }
                },
            },
        },
    }

    data = {
        "groups": [
            {
                "items": [
                    {"metadata": [{"key": "region", "value": "north-west"}]},
                    {"metadata": None},
                ]
            }
        ]
    }
    assert restore_gemini_dicts(data, schema) == {
        "groups": [
            {
                "items": [
                    {"metadata": {"region": "north-west"}},
                    {"metadata": None},
                ]
            }
        ]
    }


def test_anyof_field_that_is_not_dict_typed_is_untouched():
    # Guards the widened check: looking through anyOf must not start claiming
    # every optional field. A list-valued one would be corrupted into a dict.
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ],
                "default": None,
            },
            "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
    }
    data = {"tags": ["a", "b"], "count": None}
    assert restore_gemini_dicts(data, schema) == {"tags": ["a", "b"], "count": None}


# ── Gemini: boolean additionalProperties ─────────────────────────────────────


def test_gemini_drops_additional_properties_false_instead_of_rewriting():
    # Pydantic extra="forbid" emits additionalProperties: false. It is not a map,
    # and used to be rewritten into a key/value array of `false` values.
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "required": ["inner"],
        "additionalProperties": False,
    }
    assert to_gemini_schema(schema) == {
        "type": "object",
        "properties": {
            "inner": {"type": "object", "properties": {"a": {"type": "string"}}},
        },
        "required": ["inner"],
    }
    data = {"inner": {"a": "x"}}
    assert restore_gemini_dicts(data, schema) == {"inner": {"a": "x"}}


# ── Anthropic dialect ────────────────────────────────────────────────────────


def test_anthropic_closes_objects_and_keeps_required_as_given():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],
    }
    out = to_anthropic_schema(schema)
    assert out["additionalProperties"] is False
    assert out["required"] == ["a"]
    assert "additionalProperties" not in schema  # input untouched


def test_anthropic_strips_unsupported_constraints():
    out = to_anthropic_schema(
        {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "minimum": 1, "maximum": 5, "multipleOf": 1},
                "s": {"type": "string", "minLength": 2, "maxLength": 9, "pattern": "^x",
                      "format": "email"},
                "f": {"type": "string", "format": "phone"},
                "xs": {"type": "array", "items": {"type": "string"}, "minItems": 2,
                       "maxItems": 4},
                "ys": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
        }
    )
    props = out["properties"]
    assert set(props["n"]) == {"type", "description"}
    assert set(props["s"]) == {"type", "format", "description"}
    assert props["s"]["format"] == "email"
    assert "format" not in props["f"]
    assert set(props["xs"]) == {"type", "items", "description"}
    assert props["ys"]["minItems"] == 1


def test_anthropic_one_of_becomes_any_of():
    out = to_anthropic_schema(
        {"type": "object", "properties": {"v": {"oneOf": [{"type": "string"}, {"type": "null"}]}}}
    )
    assert out["properties"]["v"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}


@pytest.mark.parametrize(
    "schema, message",
    [
        ({"type": "object"}, "Free-form object"),
        ({"type": "object", "additionalProperties": True}, "Free-form object"),
        ({"type": "object", "properties": {"x": {"$ref": "https://e.g/x"}}}, "External"),
        (
            {"type": "object", "properties": {"self": {"$ref": "#"}}},
            "Recursive",
        ),
        (
            {
                "$defs": {
                    "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
                    "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}},
                },
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
            },
            r"#/\$defs/A -> #/\$defs/B -> #/\$defs/A",
        ),
    ],
)
def test_anthropic_rejects_what_it_cannot_express(schema, message):
    with pytest.raises(UnsupportedSchemaError, match=message):
        to_anthropic_schema(schema)


def test_shared_non_recursive_refs_are_fine():
    schema = {
        "$defs": {"Leaf": {"type": "object", "properties": {"v": {"type": "string"}}}},
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Leaf"}, "b": {"$ref": "#/$defs/Leaf"}},
    }
    assert to_anthropic_schema(schema)["$defs"]["Leaf"]["additionalProperties"] is False


def test_closed_empty_object_is_allowed():
    out = to_anthropic_schema({"type": "object", "additionalProperties": False})
    assert out == {"type": "object", "properties": {}, "additionalProperties": False}


# ── validator ────────────────────────────────────────────────────────────────


def test_validator_reports_paths():
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 0},
            "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "uniqueItems": True},
            "name": {"type": "string", "pattern": "^[a-z]+$"},
        },
        "required": ["n", "name"],
        "additionalProperties": False,
    }
    assert schema_errors({"n": 1, "name": "ok", "tags": ["a"]}, schema) == []
    errors = schema_errors({"n": -1, "tags": ["a", "c", "a"], "extra": 1}, schema)
    assert "$: missing required property 'name'" in errors
    assert "$: unexpected property 'extra'" in errors
    assert "$.n: -1 is less than the minimum 0" in errors
    assert any(e.startswith("$.tags[1]:") for e in errors)
    assert "$.tags: items are not unique" in errors


def test_validator_type_edge_cases():
    assert schema_errors(True, {"type": "integer"}) == ["$: expected integer, got boolean"]
    assert schema_errors(2.0, {"type": "integer"}) == []
    assert schema_errors(None, {"type": ["string", "null"]}) == []
    assert schema_errors(0.3, {"type": "number", "multipleOf": 0.1}) == []
    assert schema_errors(1, {"enum": [True]}) != []
    assert schema_errors(5, {"exclusiveMaximum": 5}) != []
