"""Schema-dialect translation and client-side validation.

Gemini's ``response_schema`` is an OpenAPI subset that rejects
``additionalProperties`` (i.e. open-ended dicts / maps), so:

  * on the way IN, every ``{"type": "object", "additionalProperties": T}`` node
    whose ``T`` is a subschema is rewritten to an array of
    ``{"key": string, "value": T}`` objects (a boolean ``additionalProperties``
    is simply dropped), and
  * on the way OUT, those key/value arrays are folded back into real dicts.

The restoration is driven by the *original* schema rather than a hardcoded list
of field names, so it generalizes to any caller's model — including Pydantic
schemas that use ``$defs`` / ``$ref``.

Anthropic structured outputs accept a stricter dialect (see
:func:`to_anthropic_schema`): closed objects only, no recursion, no numeric or
string-length constraints. Maps get the same key/value rewrite as Gemini, and
the constraints the dialect drops are checked afterwards by
:func:`schema_errors`, a small validator that keeps this library free of a
``jsonschema`` dependency.
"""
from __future__ import annotations

import math
import re
from typing import Any

from .errors import UnsupportedSchemaError


def to_gemini_schema(schema: Any) -> Any:
    """Rewrite dict-typed object nodes to key/value-array nodes for Gemini.

    A structural, name-agnostic transform over the whole schema document
    (including ``$defs``): any object node carrying an ``additionalProperties``
    subschema becomes an array-of-{key,value} node. A boolean
    ``additionalProperties`` (Pydantic's ``extra="forbid"`` emits ``false``)
    says nothing about a map, so it is dropped rather than rewritten.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and isinstance(
            schema.get("additionalProperties"), dict
        ):
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
        return {
            k: to_gemini_schema(v)
            for k, v in schema.items()
            if not (k == "additionalProperties" and isinstance(v, bool))
        }
    if isinstance(schema, list):
        return [to_gemini_schema(item) for item in schema]
    return schema


def restore_gemini_dicts(data: Any, original_schema: dict) -> Any:
    """Fold key/value arrays back into dicts, in place.

    Undoes the map rewrite of both :func:`to_gemini_schema` and
    :func:`to_anthropic_schema`.

    Uses ``original_schema`` (the *untransformed* schema) to discover which field
    names were dict-typed, then walks ``data`` converting any list-of-{key,value}
    under those names back to a dict.
    """
    dict_fields = _collect_dict_field_names(original_schema)
    return _restore(data, dict_fields)


# ── Anthropic ────────────────────────────────────────────────────────────────

#: String formats Anthropic structured outputs accept. Any other ``format`` is
#: dropped from the request.
ANTHROPIC_STRING_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri",
     "ipv4", "ipv6", "uuid"}
)

#: Keywords passed through to Anthropic unchanged.
_ANTHROPIC_KEEP = ("type", "enum", "const", "description", "title")

#: Annotations with no bearing on validity: dropped silently.
_ANNOTATIONS = frozenset(
    {"$schema", "$id", "$comment", "default", "examples", "readOnly", "writeOnly",
     "deprecated"}
)

#: Keywords :func:`_to_anthropic` handles itself; anything else left on a node
#: is a constraint the dialect lacks.
_ANTHROPIC_HANDLED = frozenset(
    {*_ANTHROPIC_KEEP, *_ANNOTATIONS, "$defs", "definitions", "anyOf", "oneOf",
     "allOf", "properties", "required", "additionalProperties", "items"}
)


def to_anthropic_schema(schema: dict) -> dict:
    """Translate a JSON schema to the dialect Anthropic structured outputs accept.

    * Every object is closed (``additionalProperties: false``). The caller's
      ``required`` list is kept as given: Anthropic allows optional properties,
      so forcing them all required would change the caller's contract.
    * A map (``additionalProperties`` holding a subschema) becomes a key/value
      array, as for Gemini; fold the result back with
      :func:`restore_gemini_dicts`.
    * ``oneOf`` becomes ``anyOf``.
    * Constraints the dialect lacks (``minimum``, ``maximum``, ``multipleOf``,
      ``minLength``, ``maxLength``, ``pattern``, ``maxItems``, ``minItems``
      above 1, unsupported ``format`` values, ...) are removed and noted in the
      description so the model can still aim for them. Check the result
      against the *original* schema with :func:`schema_errors`.

    Raises :class:`UnsupportedSchemaError` for what cannot be expressed at all:
    recursive schemas, external ``$ref`` and free-form objects. The input is not
    mutated.
    """
    if not isinstance(schema, dict):
        raise UnsupportedSchemaError("A structured-output schema must be a JSON object.")
    _reject_recursion(schema)
    return _to_anthropic(schema, "#")


def _to_anthropic(node: Any, path: str) -> Any:
    if isinstance(node, bool):
        raise UnsupportedSchemaError(
            f"Boolean subschema at {path} is not supported by Anthropic structured "
            "outputs; give it a type."
        )
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        if key in node:
            out[key] = {
                name: _to_anthropic(sub, f"{path}/{key}/{name}")
                for name, sub in node[key].items()
            }
    if "$ref" in node:
        ref = node["$ref"]
        if not ref.startswith("#"):
            raise UnsupportedSchemaError(
                f"External $ref {ref!r} at {path} is not supported; inline it."
            )
        out["$ref"] = ref
        return out

    for key in _ANTHROPIC_KEEP:
        if key in node:
            out[key] = node[key]
    for combinator, target in (("anyOf", "anyOf"), ("oneOf", "anyOf"), ("allOf", "allOf")):
        if combinator in node:
            out.setdefault(target, []).extend(
                _to_anthropic(sub, f"{path}/{combinator}/{i}")
                for i, sub in enumerate(node[combinator])
            )

    types = node.get("type")
    types = types if isinstance(types, list) else [types]
    if "object" in types or "properties" in node:
        additional = node.get("additionalProperties")
        if isinstance(additional, dict):
            # A map: rewrite to key/value pairs, as for Gemini.
            map_node: dict[str, Any] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": _to_anthropic(additional, f"{path}/additionalProperties"),
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            }
            if "description" in node:
                map_node["description"] = node["description"]
            return map_node
        if "properties" not in node and additional is not False:
            raise UnsupportedSchemaError(
                f"Free-form object at {path} is not supported by Anthropic structured "
                "outputs: declare its properties, or give additionalProperties a "
                "value schema to make it a map."
            )
        out["properties"] = {
            name: _to_anthropic(sub, f"{path}/properties/{name}")
            for name, sub in node.get("properties", {}).items()
        }
        if "required" in node:
            out["required"] = list(node["required"])
        out["additionalProperties"] = False
    if "items" in node:
        out["items"] = _to_anthropic(node["items"], f"{path}/items")
    if node.get("minItems") in (0, 1):
        out["minItems"] = node["minItems"]
    if node.get("format") in ANTHROPIC_STRING_FORMATS:
        out["format"] = node["format"]

    notes = {
        key: value
        for key, value in node.items()
        if key not in _ANTHROPIC_HANDLED and out.get(key) != value
    }
    if notes:
        note = "{" + ", ".join(f"{k}: {v}" for k, v in notes.items()) + "}"
        description = out.get("description")
        out["description"] = f"{description}\n\n{note}" if description else note
    return out


def _reject_recursion(schema: dict) -> None:
    """Raise :class:`UnsupportedSchemaError` if any local ``$ref`` chain loops."""

    def refs_in(node: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            if isinstance(node.get("$ref"), str):
                found.add(node["$ref"])
            for key, value in node.items():
                if key not in ("$defs", "definitions"):
                    found |= refs_in(value)
        elif isinstance(node, list):
            for item in node:
                found |= refs_in(item)
        return found

    visiting: list[str] = []
    done: set[str] = set()

    def visit(ref: str) -> None:
        if ref in done or not ref.startswith("#"):
            return
        if ref in visiting:
            cycle = " -> ".join(visiting[visiting.index(ref):] + [ref])
            raise UnsupportedSchemaError(
                f"Recursive schema ({cycle}) is not supported by Anthropic "
                "structured outputs: bound the nesting depth explicitly."
            )
        node = schema if ref == "#" else _resolve_ref({"$ref": ref}, schema)
        if node == {"$ref": ref}:  # dangling: nothing to follow
            return
        visiting.append(ref)
        for inner in sorted(refs_in(node)):
            visit(inner)
        visiting.pop()
        done.add(ref)

    for ref in sorted(refs_in(schema)):
        visit(ref)


# ── validation ───────────────────────────────────────────────────────────────

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
    or (isinstance(v, float) and v.is_integer()),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def schema_errors(instance: Any, schema: Any, *, root: Any = None) -> list[str]:
    """Where ``instance`` fails ``schema``, as ``"<json path>: <problem>"`` strings.

    An empty list means it conforms. Covers the keywords structured-output
    schemas use in practice — types, properties/required/additionalProperties,
    items, enum/const, anyOf/oneOf/allOf, local ``$ref``, and the numeric,
    string and array constraints providers drop — and ignores the rest
    (``format``, annotations). Not a full JSON Schema implementation.
    """
    errors: list[str] = []
    _validate(instance, schema, root if root is not None else schema, "$", errors)
    return errors


def _validate(value: Any, schema: Any, root: Any, path: str, errors: list[str]) -> None:
    if schema is False:
        errors.append(f"{path}: no value is allowed here")
        return
    if not isinstance(schema, dict):
        return
    if "$ref" in schema:
        resolved = _resolve_ref(schema, root)
        if resolved is not schema:
            _validate(value, resolved, root, path, errors)
        return

    for sub in schema.get("allOf", []):
        _validate(value, sub, root, path, errors)
    if "anyOf" in schema and all(
        schema_errors(value, sub, root=root) for sub in schema["anyOf"]
    ):
        errors.append(f"{path}: matches none of the anyOf alternatives")
    if "oneOf" in schema:
        matching = sum(not schema_errors(value, sub, root=root) for sub in schema["oneOf"])
        if matching != 1:
            errors.append(f"{path}: matches {matching} of the oneOf alternatives, not 1")
    if "enum" in schema and not any(_json_equal(value, v) for v in schema["enum"]):
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")
    if "const" in schema and not _json_equal(value, schema["const"]):
        errors.append(f"{path}: {value!r} is not {schema['const']!r}")

    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPE_CHECKS.get(t, lambda v: True)(value) for t in types):
            errors.append(f"{path}: expected {' or '.join(types)}, got {_json_type(value)}")
            return

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            child = f"{path}.{name}"
            if name in properties:
                _validate(item, properties[name], root, child, errors)
            elif additional is False:
                errors.append(f"{path}: unexpected property {name!r}")
            else:
                _validate(item, additional, root, child, errors)
    elif isinstance(value, list):
        if "items" in schema:
            for i, item in enumerate(value):
                _validate(item, schema["items"], root, f"{path}[{i}]", errors)
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems") and any(
            _json_equal(a, b) for i, a in enumerate(value) for b in value[i + 1:]
        ):
            errors.append(f"{path}: items are not unique")
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
    elif _TYPE_CHECKS["number"](value):
        _check_number(value, schema, path, errors)


def _check_number(value: float, schema: dict, path: str, errors: list[str]) -> None:
    minimum, maximum = schema.get("minimum"), schema.get("maximum")
    excl_min, excl_max = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
    # Draft 4 spells exclusivity as a boolean beside minimum/maximum.
    if excl_min is True:
        excl_min, minimum = minimum, None
    if excl_max is True:
        excl_max, maximum = maximum, None
    if _is_number(minimum) and value < minimum:
        errors.append(f"{path}: {value} is less than the minimum {minimum}")
    if _is_number(maximum) and value > maximum:
        errors.append(f"{path}: {value} is greater than the maximum {maximum}")
    if _is_number(excl_min) and value <= excl_min:
        errors.append(f"{path}: {value} is not greater than {excl_min}")
    if _is_number(excl_max) and value >= excl_max:
        errors.append(f"{path}: {value} is not less than {excl_max}")
    step = schema.get("multipleOf")
    if _is_number(step) and step > 0:
        quotient = value / step
        if not math.isclose(quotient, round(quotient), rel_tol=0, abs_tol=1e-9):
            errors.append(f"{path}: {value} is not a multiple of {step}")


def _is_number(value: Any) -> bool:
    return _TYPE_CHECKS["number"](value)


def _json_type(value: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _TYPE_CHECKS[name](value):
            return name
    return type(value).__name__


def _json_equal(a: Any, b: Any) -> bool:
    """Equality as JSON sees it: ``True`` is not ``1``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


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
