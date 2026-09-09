"""Schema-dialect translation.

Currently only Gemini needs it. Gemini's ``response_schema`` is an OpenAPI
subset that rejects ``additionalProperties`` (i.e. open-ended dicts / maps), so:

  * on the way IN, every ``{"type": "object", "additionalProperties": T}`` node
    is rewritten to an array of ``{"key": string, "value": T}`` objects, and
  * on the way OUT, those key/value arrays are folded back into real dicts.

The restoration is driven by the *original* schema rather than a hardcoded list
of field names, so it generalizes to any caller's model — including Pydantic
schemas that use ``$defs`` / ``$ref``.
"""
from __future__ import annotations

from typing import Any


def to_gemini_schema(schema: Any) -> Any:
    """Rewrite dict-typed object nodes to key/value-array nodes for Gemini.

    A structural, name-agnostic transform over the whole schema document
    (including ``$defs``): any object node carrying an ``additionalProperties``
    subschema becomes an array-of-{key,value} node.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "additionalProperties" in schema:
            val_type = schema["additionalProperties"]
            new_schema: dict[str, Any] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": val_type,
                    },
                    "required": ["key", "value"],
                },
            }
            if "description" in schema:
                new_schema["description"] = schema["description"]
            return new_schema
        return {k: to_gemini_schema(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return [to_gemini_schema(item) for item in schema]
    return schema


def restore_gemini_dicts(data: Any, original_schema: dict) -> Any:
    """Fold Gemini's key/value arrays back into dicts, in place.

    Uses ``original_schema`` (the *untransformed* schema) to discover which field
    names were dict-typed, then walks ``data`` converting any list-of-{key,value}
    under those names back to a dict.
    """
    dict_fields = _collect_dict_field_names(original_schema)
    return _restore(data, dict_fields)


# ── internals ────────────────────────────────────────────────────────────────


def _restore(data: Any, dict_fields: set[str]) -> Any:
    if isinstance(data, dict):
        for field in list(data.keys()):
            value = data[field]
            if field in dict_fields and isinstance(value, list):
                restored: dict[str, Any] = {}
                for item in value:
                    if isinstance(item, dict) and "key" in item and "value" in item:
                        restored[str(item["key"])] = _restore(item["value"], dict_fields)
                data[field] = restored
            else:
                data[field] = _restore(value, dict_fields)
        return data
    if isinstance(data, list):
        return [_restore(item, dict_fields) for item in data]
    return data


def _resolve_ref(node: Any, root: dict) -> Any:
    """Follow a ``{"$ref": "#/$defs/Foo"}`` node to its definition."""
    if isinstance(node, dict) and "$ref" in node:
        target: Any = root
        for part in node["$ref"].lstrip("#/").split("/"):
            if not isinstance(target, dict) or part not in target:
                return node  # dangling ref — leave as-is rather than crash
            target = target[part]
        return target
    return node


def _is_dict_typed(node: Any, root: dict, seen: set[int] | None = None) -> bool:
    """Whether ``node`` denotes a dict, looking through ``anyOf``/``oneOf``/``allOf``.

    ``Optional[dict[...]]`` never serializes as a bare ``{"type": "object",
    "additionalProperties": T}`` node — Pydantic emits ``anyOf: [<that>, null]``
    — so a check that only inspects the node itself misses the most common shape
    a real model produces. Any branch being dict-typed is enough: the branch is
    what ``to_gemini_schema`` rewrote to a key/value array, so it is what may
    come back needing restoration.

    Deliberately does *not* descend into ``items``. A ``list[dict[str, str]]``
    field carries a genuine list at that name, and marking it would make
    ``_restore`` mangle the outer list.
    """
    node = _resolve_ref(node, root)
    if not isinstance(node, dict):
        return False
    if seen is None:
        seen = set()
    if id(node) in seen:  # recursive $ref — a cycle proves nothing
        return False
    seen.add(id(node))

    if node.get("type") == "object" and isinstance(node.get("additionalProperties"), dict):
        return True
    return any(
        _is_dict_typed(branch, root, seen)
        for combinator in ("anyOf", "oneOf", "allOf")
        for branch in node.get(combinator, [])
    )


def _collect_dict_field_names(schema: dict) -> set[str]:
    """Property names whose subschema is an ``additionalProperties`` object.

    Resolves ``$ref``/``$defs`` and guards against recursive schemas. Name-based
    (not path-based) to mirror how the data is walked; a collision between a
    dict-typed and non-dict field sharing a name is an accepted v1 limitation.
    """
    root = schema
    names: set[str] = set()
    seen: set[int] = set()

    def walk(node: Any) -> None:
        node = _resolve_ref(node, root)
        if not isinstance(node, dict):
            return
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)

        if node.get("type") == "object":
            for name, sub in node.get("properties", {}).items():
                if _is_dict_typed(sub, root):
                    names.add(name)
                walk(sub)
            ap = node.get("additionalProperties")
            if isinstance(ap, dict):
                walk(ap)
        if node.get("type") == "array":
            walk(node.get("items", {}))
        for combinator in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(combinator, []):
                walk(sub)
        for definition in node.get("$defs", {}).values():
            walk(definition)

    walk(schema)
    return names
