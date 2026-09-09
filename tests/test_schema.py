"""Gemini schema-dialect translation tests — the divergence a plain OpenAI shim
cannot handle."""
from __future__ import annotations

from llmswitchboard.schema import restore_gemini_dicts, to_gemini_schema


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
