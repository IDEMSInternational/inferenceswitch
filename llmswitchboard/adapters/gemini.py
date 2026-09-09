"""Native Gemini adapter.

Gemini enforces schemas via ``response_mime_type="application/json"`` +
``response_schema``, but its schema dialect rejects open-ended dicts
(``additionalProperties``). The translation is applied here and nowhere else:
:func:`llmswitchboard.schema.to_gemini_schema` rewrites dict fields to key/value
arrays on the way in, and :func:`~llmswitchboard.schema.restore_gemini_dicts` folds
them back on the way out.
"""
from __future__ import annotations

import json
from typing import Any

from .. import schema as schema_mod
from ..errors import StructuredOutputError
from ..messages import (
    CacheHandle,
    Effort,
    LLMResponse,
    Message,
    StopReason,
    Text,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolUse,
    Usage,
)
from .base import Adapter, require_sdk, split_system

#: Per-model output ceilings for the Gemini line, mirroring
#: :data:`~llmswitchboard.adapters.anthropic.CLAUDE_MAX_OUTPUT_TOKENS` and for the same
#: reason: llmswitchboard has to pick a default when the caller doesn't, and the right
#: default is *that model's* maximum. Gemini differs from Anthropic in that
#: ``max_output_tokens`` is optional rather than required — but leaving it unset
#: is not the safe option it looks like. It hands the ceiling to whatever the
#: provider currently defaults to, which is (a) undocumented, (b) free to change
#: under you, and (c) far below what the same caller gets from the Anthropic
#: adapter for the identical call. Output is billed per token generated, not per
#: token requested, so asking for the model's maximum costs nothing and removes
#: an invisible floor on how large a document the caller can process.
#:
#: The values below are the documented "Output token limit" from each model's
#: page on ai.google.dev, checked 2026-08. The whole current line — Gemini 3.x
#: and 2.5, pro/flash/flash-lite alike — is uniformly 65,536, so this table has
#: no interesting variation *today*. It exists anyway, because the Claude table
#: didn't either until Haiku 4.5 shipped with half the ceiling of its siblings,
#: and because a flat table is the only thing that makes a future exception
#: expressible as one line rather than a refactor.
#:
#: Like the Claude table this is a plain mutable dict, and is the control
#: surface — add a row for a model llmswitchboard doesn't know yet, or lower one to
#: put a hard ceiling on a model process-wide::
#:
#:     from llmswitchboard import GEMINI_MAX_OUTPUT_TOKENS
#:     GEMINI_MAX_OUTPUT_TOKENS["gemini-3.7-flash"] = 16_000   # house limit
#:
#: A per-call ``max_tokens=`` still wins over the table, and is never clamped to
#: it — an explicit number is the caller's decision, and an out-of-range one
#: surfaces Gemini's own 400 rather than being silently rewritten.
GEMINI_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "gemini-3.7-flash": 65_536,
    "gemini-3.6-flash": 65_536,
    "gemini-3.5-flash": 65_536,
    "gemini-3.5-flash-lite": 65_536,
    "gemini-3.1-pro": 65_536,
    "gemini-3.1-flash-lite": 65_536,
    "gemini-3-flash": 65_536,
    "gemini-2.5-pro": 65_536,
    "gemini-2.5-flash": 65_536,
    "gemini-2.5-flash-lite": 65_536,
}

#: Used when a model ID matches nothing in the table. 65536 is the ceiling shared
#: by every model in the *current* line, so it is the largest value that is safe
#: to send blind to any of them — and, since the table is currently flat, it is
#: also what an unrecognised new Gemini most likely supports.
#:
#: The models this is wrong for are old ones: the retired Gemini 1.5 and 2.0
#: generations capped output at 8,192, and a blind 65536 to one of those gets a
#: 400 rather than a silent truncation. That is the correct failure — loud, at
#: the provider, naming the real limit — but if you are pinned to a pre-2.5
#: model, add an explicit row above rather than relying on this.
#:
#: Named with the provider in it, unlike the Anthropic constant it mirrors:
#: ``UNKNOWN_MODEL_MAX_OUTPUT_TOKENS`` is already exported from the package root
#: for Claude, and renaming an exported name to gain symmetry would break
#: callers for nothing.
UNKNOWN_GEMINI_MODEL_MAX_OUTPUT_TOKENS = 65_536


def gemini_max_output_tokens(model: str) -> int:
    """This model's maximum output tokens, per :data:`GEMINI_MAX_OUTPUT_TOKENS`.

    Falls back to substring matching on the longest known ID so dated and
    staged snapshots (``gemini-3-flash-preview``, ``gemini-2.5-flash-002``) and
    provider-prefixed IDs (``google/gemini-3.5-flash`` on a gateway) resolve to
    the same ceiling as the bare alias, then to
    :data:`UNKNOWN_GEMINI_MODEL_MAX_OUTPUT_TOKENS`.

    Longest-match matters more here than it does for Claude, because Gemini's IDs
    nest: ``gemini-3.5-flash`` is a substring of ``gemini-3.5-flash-lite``, so a
    first-match rule would resolve the lite variant to the wrong row the moment
    the two ceilings diverge.
    """
    exact = GEMINI_MAX_OUTPUT_TOKENS.get(model)
    if exact is not None:
        return exact
    matches = [name for name in GEMINI_MAX_OUTPUT_TOKENS if name in model]
    if matches:
        return GEMINI_MAX_OUTPUT_TOKENS[max(matches, key=len)]
    return UNKNOWN_GEMINI_MODEL_MAX_OUTPUT_TOKENS


def resolve_max_tokens(model: str, max_tokens: int | None) -> int:
    """The caller's ``max_tokens`` if given, else the model's own maximum."""
    return max_tokens if max_tokens is not None else gemini_max_output_tokens(model)


# Gemini controls reasoning by a thinking-token budget, not a level. -1 = let the
# model decide (dynamic / effectively max); 0 = off where the model allows it.
_GEMINI_BUDGET = {
    Effort.NONE: 0,
    Effort.LOW: 1024,
    Effort.MEDIUM: 8192,
    Effort.HIGH: 24576,
    Effort.MAX: -1,
}


def gemini_thinking_budget(effort: Effort | None) -> int | None:
    return None if effort is None else _GEMINI_BUDGET[effort]

_FINISH_REASON = {
    "STOP": StopReason.END_TURN,
    "MAX_TOKENS": StopReason.MAX_TOKENS,
    "SAFETY": StopReason.REFUSAL,
    "RECITATION": StopReason.REFUSAL,
    "PROHIBITED_CONTENT": StopReason.REFUSAL,
}


def gemini_finish_reason(raw) -> str | None:
    """The first candidate's ``finish_reason``, as the plain string Gemini names it.

    The SDK returns an enum whose ``.name`` is the wire value, but a hand-rolled
    fake or a raw dict-shaped response yields the string directly, so both are
    accepted. Returns ``None`` when there is no candidate at all — which is
    itself a real case (a prompt blocked before generation), and one the caller
    must not confuse with "finished normally".
    """
    candidates = getattr(raw, "candidates", None) or []
    if not candidates:
        return None
    fr = getattr(candidates[0], "finish_reason", None)
    if fr is None:
        return None
    return getattr(fr, "name", None) or str(fr)


def _prompt_token_count(raw) -> int | None:
    meta = getattr(raw, "usage_metadata", None)
    return None if meta is None else getattr(meta, "prompt_token_count", None)


def encode_gemini_contents(messages: list[Message]) -> list[dict]:
    """Normalized messages -> Gemini ``contents`` (role user/model + typed parts).

    Gemini keys tool results by function *name*, not call id, so a ToolResult
    without a name falls back to its id — carry the name (the ``tool_result``
    helper does) for reliable round-trips.
    """
    out: list[dict] = []
    for message in messages:
        role = "model" if message.role == "assistant" else "user"
        parts: list[dict] = []
        for part in message.parts():
            if isinstance(part, Text):
                if part.text:
                    parts.append({"text": part.text})
            elif isinstance(part, ToolUse):
                parts.append({"function_call": {"name": part.name, "args": part.input}})
            elif isinstance(part, ToolResult):
                parts.append(
                    {
                        "function_response": {
                            "name": part.name or part.tool_call_id,
                            "response": {"result": part.content},
                        }
                    }
                )
        out.append({"role": role, "parts": parts})
    return out


def decode_gemini_response(raw, tools_by_name: dict[str, Tool] | None = None) -> LLMResponse:
    """Gemini response -> normalized LLMResponse.

    Synthesizes stable ids for tool calls (Gemini emits none) and restores any
    dict-typed tool arguments that were flattened to key/value arrays by the
    Gemini schema dialect.
    """
    tools_by_name = tools_by_name or {}
    candidate = raw.candidates[0]
    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, part in enumerate(candidate.content.parts or []):
        if getattr(part, "text", None):
            text_chunks.append(part.text)
        fc = getattr(part, "function_call", None)
        if fc is not None:
            args = dict(fc.args) if fc.args else {}
            declared = tools_by_name.get(fc.name)
            if declared is not None:
                args = schema_mod.restore_gemini_dicts(args, declared.input_schema)
            tool_calls.append(
                ToolCall(id=f"call_{fc.name}_{index}", name=fc.name, input=args)
            )

    fr_name = gemini_finish_reason(raw)
    stop = StopReason.TOOL_USE if tool_calls else _FINISH_REASON.get(fr_name, StopReason.END_TURN)

    usage = Usage()
    meta = getattr(raw, "usage_metadata", None)
    if meta is not None:
        usage.input_tokens = getattr(meta, "prompt_token_count", None)
        usage.output_tokens = getattr(meta, "candidates_token_count", None)
        usage.cache_read_tokens = getattr(meta, "cached_content_token_count", None)
        usage.reasoning_tokens = getattr(meta, "thoughts_token_count", None)

    return LLMResponse(
        text="".join(text_chunks), tool_calls=tool_calls, stop_reason=stop, usage=usage, raw=raw
    )


class GeminiAdapter(Adapter):
    def _build_client(self, api_key: str) -> Any:
        genai = require_sdk("google.genai", "gemini")
        return genai.Client(api_key=api_key)

    def _config(self, *, system, response_schema, temperature, max_tokens):
        types = require_sdk("google.genai", "gemini").types
        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def generate_structured_json(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict,
        system: str | None = None,
        tool_name: str = "generate_json",  # unused; Gemini has no tool-name concept
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Any:
        max_tokens = resolve_max_tokens(model, max_tokens)
        gemini_schema = schema_mod.to_gemini_schema(schema)
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=self._config(
                system=system,
                response_schema=gemini_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

        # Check the finish reason BEFORE parsing. A response cut at the ceiling is
        # a prefix of valid JSON, so json.loads fails with a character offset deep
        # inside the document ("Unterminated string ... char 85119") — a message
        # that sends the caller to inspect a schema that was never the problem.
        # The stop reason is the only place the truth is written down, and it is
        # written down before we ever touch the text.
        if gemini_finish_reason(response) == "MAX_TOKENS":
            model_max = gemini_max_output_tokens(model)
            remedy = (
                " — raise max_tokens toward it."
                if max_tokens < model_max
                else ", so this response does not fit in one call: shorten the "
                "prompt, split the request, or narrow the schema."
            )
            input_tokens = _prompt_token_count(response)
            sized = (
                f" The prompt was {input_tokens} tokens; output size tracks input "
                "size, so this recurs on any request at least this large."
                if input_tokens is not None
                else " Output size tracks input size, so this recurs on any "
                "request at least this large."
            )
            raise StructuredOutputError(
                f"Gemini hit the {max_tokens}-token output cap before finishing the "
                f"JSON, so the response is truncated mid-value rather than "
                f"malformed. This model's maximum is {model_max}{remedy}{sized}",
                context={
                    "provider": self.spec.name,
                    "model": model,
                    "max_tokens": max_tokens,
                    "model_max_tokens": model_max,
                    "stop_reason": "MAX_TOKENS",
                    "input_tokens": input_tokens,
                    "output_tokens": getattr(
                        getattr(response, "usage_metadata", None),
                        "candidates_token_count",
                        None,
                    ),
                },
            )

        try:
            raw = json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise StructuredOutputError(
                f"Gemini returned non-JSON content: {exc}",
                context={
                    "provider": self.spec.name,
                    "model": model,
                    "max_tokens": max_tokens,
                    "stop_reason": gemini_finish_reason(response),
                },
            ) from exc
        # Fold Gemini's key/value arrays back into real dicts using the ORIGINAL
        # (untranslated) schema to know which fields were dicts.
        return schema_mod.restore_gemini_dicts(raw, schema)

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        types = require_sdk("google.genai", "gemini").types
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=resolve_max_tokens(model, max_tokens),
            ),
        )
        # Deliberately no MAX_TOKENS check here, unlike generate_structured_json,
        # and the asymmetry is the point. Truncated JSON is *unusable* — a prefix
        # of an object, missing the fields a validator will demand — so failing is
        # the only honest outcome. Truncated prose is merely shorter than asked
        # for: still a well-formed string, often still the answer. Raising would
        # discard a usable result, and it could only raise StructuredOutputError,
        # which is a lie about a call that requested no structure. A caller that
        # must know reaches for `chat()`, which reports StopReason.MAX_TOKENS on
        # the response instead of throwing. Same rule as the Anthropic adapter.
        return response.text or ""

    @staticmethod
    def _encode_tools(types, tools: list[Tool]):
        """Normalized tools -> Gemini ``types.Tool`` (function declarations),
        sharing Gemini's schema dialect. Used by both ``chat`` and ``create_cache``."""
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=schema_mod.to_gemini_schema(t.input_schema),
                    )
                    for t in tools
                ]
            )
        ]

    def create_cache(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[Tool] | None = None,
        ttl: int | None = None,
    ) -> CacheHandle:
        """Create a Gemini ``CachedContent`` resource from a system prompt +
        message prefix (+ optional tools) and return a reusable handle.

        Gemini enforces a per-model minimum token count for a cache; below it the
        create call raises — surfaced as-is (not silently swallowed) so the caller
        learns the prefix is too small to be worth caching.
        """
        types = require_sdk("google.genai", "gemini").types
        system_text, rest = split_system(messages, system)
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_text,
            "contents": encode_gemini_contents(rest),
        }
        if ttl is not None:
            config_kwargs["ttl"] = f"{ttl}s"
        if tools:
            config_kwargs["tools"] = self._encode_tools(types, tools)
        cached = self._client.caches.create(
            model=model, config=types.CreateCachedContentConfig(**config_kwargs)
        )
        return CacheHandle(provider=self.spec.name, name=cached.name, model=model, raw=cached)

    def delete_cache(self, handle: CacheHandle) -> None:
        """Delete the server-side ``CachedContent`` resource this handle names."""
        self._client.caches.delete(name=handle.name)

    def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice = ToolChoice.AUTO,
        force_tool: str | None = None,
        system: str | None = None,
        effort: Effort | None = None,
        cache: CacheHandle | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        types = require_sdk("google.genai", "gemini").types
        system_text, rest = split_system(messages, system)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": resolve_max_tokens(model, max_tokens),
        }
        budget = gemini_thinking_budget(effort)
        if budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)

        tools_by_name: dict[str, Tool] = {}
        if cache is not None:
            # System prompt + tools live IN the cache; resending them alongside
            # cached_content is an error. Only the dynamic tail is sent here.
            config_kwargs["cached_content"] = cache.name
        else:
            config_kwargs["system_instruction"] = system_text
            if tools:
                tools_by_name = {t.name: t for t in tools}
                config_kwargs["tools"] = self._encode_tools(types, tools)
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=self._function_calling_config(
                        types, tool_choice, force_tool
                    )
                )

        raw = self._client.models.generate_content(
            model=model,
            contents=encode_gemini_contents(rest),
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return decode_gemini_response(raw, tools_by_name)

    @staticmethod
    def _function_calling_config(types, tool_choice: ToolChoice, force_tool: str | None):
        if force_tool is not None:
            return types.FunctionCallingConfig(mode="ANY", allowed_function_names=[force_tool])
        mode = {ToolChoice.AUTO: "AUTO", ToolChoice.REQUIRED: "ANY", ToolChoice.NONE: "NONE"}[
            tool_choice
        ]
        return types.FunctionCallingConfig(mode=mode)

    def verify_key(self) -> tuple[bool, str]:
        try:
            # models.list returns a lazy pager; realize one item to force the call.
            next(iter(self._client.models.list()), None)
            return True, "Key is valid."
        except Exception as exc:  # noqa: BLE001
            detail = str(exc).strip() or exc.__class__.__name__
            return False, detail.splitlines()[0][:300]
