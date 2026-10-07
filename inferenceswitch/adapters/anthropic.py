"""Native Anthropic adapter.

Anthropic is not OpenAI-shaped. Schema-constrained JSON comes from structured
outputs (``output_config.format``) on models that have it, and otherwise from a
tool whose ``input_schema`` is the schema; the capability registry picks per
model. Structured outputs take a stricter schema dialect, translated by
:func:`inferenceswitch.schema.to_anthropic_schema`.
"""
from __future__ import annotations

import json
from typing import Any

from .. import schema as schema_mod
from ..capabilities import StructuredMode
from ..errors import StructuredOutputError
from ..schema import to_anthropic_schema
from ..messages import (
    CacheHandle,
    Effort,
    Response,
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

#: Per-model output ceilings for the Claude line. Anthropic requires ``max_tokens``
#: on every request, so inferenceswitch has to pick a default when the caller doesn't —
#: and the right default is *that model's* maximum: anything lower silently
#: truncates a long answer (``stop_reason: "max_tokens"``) for no benefit, since
#: output is billed per token generated, not per token requested.
#:
#: This table is the control surface. It is deliberately a plain mutable dict:
#: add a row for a model inferenceswitch doesn't know yet, or lower one to put a hard
#: ceiling on a model process-wide::
#:
#:     from inferenceswitch import CLAUDE_MAX_OUTPUT_TOKENS
#:     CLAUDE_MAX_OUTPUT_TOKENS["claude-opus-5"] = 32_000   # house limit
#:
#: A per-call ``max_tokens=`` still wins over the table, and is never clamped to
#: it — an explicit number is the caller's decision, and an out-of-range one
#: surfaces Anthropic's own 400 rather than being silently rewritten.
CLAUDE_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "claude-fable-5": 128_000,
    "claude-mythos-5": 128_000,
    "claude-opus-5": 128_000,
    "claude-opus-4-8": 128_000,
    "claude-opus-4-7": 128_000,
    "claude-opus-4-6": 128_000,
    "claude-sonnet-5": 128_000,
    "claude-sonnet-4-6": 128_000,
    "claude-haiku-4-5": 64_000,
}

#: Used when a model ID matches nothing in the table. 64000 is the smallest
#: ceiling in the *current* Claude line (Haiku 4.5), so it is the largest value
#: that is safe to send blind to any current model. Older models cap lower — add
#: an explicit row above before using one.
UNKNOWN_MODEL_MAX_OUTPUT_TOKENS = 64_000


def claude_max_output_tokens(model: str) -> int:
    """This model's maximum output tokens, per :data:`CLAUDE_MAX_OUTPUT_TOKENS`.

    Falls back to substring matching on the longest known ID so dated snapshots
    (``claude-haiku-4-5-20251001``) and provider-prefixed IDs
    (``anthropic.claude-opus-4-8`` on Bedrock) resolve to the same ceiling as the
    bare alias, then to :data:`UNKNOWN_MODEL_MAX_OUTPUT_TOKENS`.
    """
    exact = CLAUDE_MAX_OUTPUT_TOKENS.get(model)
    if exact is not None:
        return exact
    matches = [name for name in CLAUDE_MAX_OUTPUT_TOKENS if name in model]
    if matches:
        return CLAUDE_MAX_OUTPUT_TOKENS[max(matches, key=len)]
    return UNKNOWN_MODEL_MAX_OUTPUT_TOKENS


def resolve_max_tokens(model: str, max_tokens: int | None) -> int:
    """The caller's ``max_tokens`` if given, else the model's own maximum."""
    return max_tokens if max_tokens is not None else claude_max_output_tokens(model)


_ANTHROPIC_EFFORT = {
    Effort.LOW: "low",
    Effort.MEDIUM: "medium",
    Effort.HIGH: "high",
    Effort.MAX: "max",
}


def anthropic_reasoning_kwargs(effort: Effort | None) -> dict:
    """Reasoning via adaptive thinking + ``output_config.effort`` on current
    Anthropic models. ``NONE`` disables thinking. (Effort support is per-model —
    e.g. Haiku 4.5 doesn't take it — so a request with effort on a model that
    lacks it will surface the provider's own 400.)"""
    if effort is None:
        return {}
    if effort is Effort.NONE:
        return {"thinking": {"type": "disabled"}}
    return {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": _ANTHROPIC_EFFORT[effort]},
    }

_STOP_REASON = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "tool_use": StopReason.TOOL_USE,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "refusal": StopReason.REFUSAL,
}


def encode_anthropic_messages(messages: list[Message]) -> list[dict]:
    """Normalized messages -> Anthropic content-block messages."""
    out: list[dict] = []
    for message in messages:
        blocks: list[dict] = []
        for part in message.parts():
            if isinstance(part, Text):
                if part.text:
                    block = {"type": "text", "text": part.text}
                    if part.cache:
                        block["cache_control"] = {"type": "ephemeral"}
                    blocks.append(block)
            elif isinstance(part, ToolUse):
                blocks.append(
                    {"type": "tool_use", "id": part.id, "name": part.name, "input": part.input}
                )
            elif isinstance(part, ToolResult):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": part.tool_call_id,
                        "content": part.content,
                        "is_error": part.is_error,
                    }
                )
        out.append({"role": message.role, "content": blocks})
    return out


def encode_anthropic_tools(tools: list[Tool]) -> list[dict]:
    out: list[dict] = []
    for t in tools:
        entry: dict = {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        if t.cache:
            # Caches the tools + system prefix up to and including this tool.
            entry["cache_control"] = {"type": "ephemeral"}
        out.append(entry)
    return out


def encode_anthropic_tool_choice(tool_choice: ToolChoice, force_tool: str | None):
    if force_tool is not None:
        return {"type": "tool", "name": force_tool}
    return {
        ToolChoice.AUTO: {"type": "auto"},
        ToolChoice.REQUIRED: {"type": "any"},
        ToolChoice.NONE: {"type": "none"},
    }[tool_choice]


def decode_anthropic_response(raw) -> Response:
    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in raw.content:
        if block.type == "text":
            text_chunks.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

    usage = Usage()
    if getattr(raw, "usage", None) is not None:
        usage.input_tokens = getattr(raw.usage, "input_tokens", None)
        usage.output_tokens = getattr(raw.usage, "output_tokens", None)
        usage.cache_read_tokens = getattr(raw.usage, "cache_read_input_tokens", None)
        usage.cache_write_tokens = getattr(raw.usage, "cache_creation_input_tokens", None)

    return Response(
        text="".join(text_chunks),
        tool_calls=tool_calls,
        stop_reason=_STOP_REASON.get(raw.stop_reason, StopReason.OTHER),
        usage=usage,
        raw=raw,
    )


def _unwrap_stray_key(result: Any, schema: dict) -> Any:
    """The value under ``result``'s single key, if that key is not a schema
    property and the value conforms to the schema; else ``None``."""
    if not isinstance(result, dict) or len(result) != 1:
        return None
    (key, inner), = result.items()
    properties = schema_mod._resolve_ref(schema, schema).get("properties", {})
    if key in properties or schema_mod.schema_errors(inner, schema):
        return None
    return inner


class AnthropicAdapter(Adapter):
    def _build_client(self, api_key: str) -> Any:
        anthropic = require_sdk("anthropic", "anthropic")
        return anthropic.Anthropic(api_key=api_key)

    def _send(self, **create_kwargs: Any) -> Any:
        """Issue one Messages request over a stream, returning the accumulated
        final message.

        Streaming rather than `messages.create` because the default cap is the
        model's own maximum (see `CLAUDE_MAX_OUTPUT_TOKENS`), far above the SDK's
        non-streaming ceiling: it derives an expected wall-clock from `max_tokens`
        and raises `ValueError` — before any network call — for a request that
        could outrun its 10-minute timeout. Streaming is the provider's prescribed
        answer for large outputs; it also keeps the connection from idling out on
        a genuinely long generation.

        `get_final_message()` returns the same `Message` object a non-streaming
        call would, so every caller and `decode_anthropic_response` are unchanged.
        """
        with self._client.messages.stream(**create_kwargs) as stream:
            return stream.get_final_message()

    def generate_structured_json(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict,
        system: str | None = None,
        tool_name: str = "generate_json",
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Any:
        """Schema-constrained JSON, by the mechanism the capability registry names
        for ``model`` (see :meth:`Capabilities.structured_output_for`):

        * ``OUTPUT_CONFIG_JSON_SCHEMA`` — structured outputs: the schema, in
          Anthropic's dialect, as ``output_config.format``; the reply's text is
          the JSON.
        * ``STRICT_TOOL_USE`` / ``TOOL_USE`` — the schema as a tool's
          ``input_schema`` (``strict: true`` for the former), offered with
          ``tool_choice: auto`` and an instruction to call it. Forced
          ``tool_choice`` is not used: the newest models reject it with a 400.

        Whatever the mechanism, the result is checked against the caller's
        original schema, including the constraints the Anthropic dialect drops,
        and a mismatch raises :class:`StructuredOutputError`.
        """
        max_tokens = resolve_max_tokens(model, max_tokens)
        # NOTE: temperature/top_p/top_k are deliberately NOT forwarded. Newer
        # Anthropic models (Opus 4.7/4.8, Fable 5) reject sampling params with a
        # 400; older ones (Sonnet 4.6, Haiku 4.5) accept them. Support is
        # per-MODEL, not per-provider, so a provider-level capability flag can't
        # express it — see the sampling-params discussion. Omitting them is safe
        # on every model.
        _ = temperature
        mode = self.capabilities.structured_output_for(model)
        context: dict[str, Any] = {
            "provider": self.spec.name,
            "model": model,
            "mode": mode.value,
        }
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            create_kwargs["system"] = system

        if mode is StructuredMode.OUTPUT_CONFIG_JSON_SCHEMA:
            create_kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": to_anthropic_schema(schema)}
            }
        elif mode in (StructuredMode.STRICT_TOOL_USE, StructuredMode.TOOL_USE):
            context["tool_name"] = tool_name
            strict = mode is StructuredMode.STRICT_TOOL_USE
            tool: dict[str, Any] = {
                "name": tool_name,
                "description": f"Return the {tool_name} result as structured JSON.",
                # Non-strict tools take the caller's schema as-is: the dialect is
                # permissive there, and the result is validated below anyway.
                "input_schema": to_anthropic_schema(schema) if strict else schema,
            }
            if strict:
                tool["strict"] = True
            create_kwargs["tools"] = [tool]
            create_kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
            instruction = (
                f"Respond only by calling the `{tool_name}` tool, exactly once. "
                "Its input is your answer."
            )
            create_kwargs["system"] = f"{system}\n\n{instruction}" if system else instruction
        else:
            raise StructuredOutputError(
                f"Anthropic adapter cannot honor structured mode {mode!r}.",
                context=context,
            )

        response = self._send(**create_kwargs)
        context["stop_reason"] = response.stop_reason
        self._check_structured_stop(response, model, max_tokens, context)

        if mode is StructuredMode.OUTPUT_CONFIG_JSON_SCHEMA:
            text = next((b.text for b in response.content if b.type == "text"), None)
            if text is None:
                raise StructuredOutputError(
                    "Claude returned no text block for a structured-output request.",
                    context=context,
                )
            try:
                result = json.loads(text)
            except json.JSONDecodeError as exc:
                raise StructuredOutputError(
                    f"Claude's structured output is not valid JSON: {exc}",
                    context=context,
                ) from exc
        else:
            tool_use = next(
                (b for b in response.content if b.type == "tool_use" and b.name == tool_name),
                None,
            )
            if tool_use is None:
                raise StructuredOutputError(
                    f"Claude did not call the '{tool_name}' tool, so there is no "
                    "structured result.",
                    context=context,
                )
            # The Anthropic SDK parses tool inputs into native dicts already.
            result = tool_use.input

        result = schema_mod.restore_gemini_dicts(result, schema)
        errors = schema_mod.schema_errors(result, schema)
        if errors and mode is StructuredMode.TOOL_USE:
            # A non-strict tool call sometimes nests a valid answer under one
            # stray key, e.g. {"$PARAMETER_NAME": {...}}. Unwrap only when that key
            # is not a schema property and the inner value conforms.
            unwrapped = _unwrap_stray_key(result, schema)
            if unwrapped is not None:
                return unwrapped
        if errors:
            raise StructuredOutputError(
                "Claude's structured output does not match the schema: "
                + "; ".join(errors[:5])
                + (f" (and {len(errors) - 5} more)" if len(errors) > 5 else ""),
                context={**context, "schema_errors": errors},
            )
        return result

    def _check_structured_stop(self, response, model: str, max_tokens: int, context: dict) -> None:
        """Raise for the stop reasons that leave no trustworthy structured result."""
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            explanation = getattr(details, "explanation", None)
            raise StructuredOutputError(
                "Claude declined the request (stop_reason 'refusal'), so the output "
                "may not match the schema"
                + (f" [category: {category}]" if category else "")
                + (f": {explanation}" if explanation else "."),
                context={**context, "refusal_category": category},
            )
        # A max_tokens stop means the JSON was cut off mid-object; the partial
        # value would be missing trailing (often required) fields. Fail loudly
        # here rather than letting a validator report a misleading "field
        # required".
        if response.stop_reason == "max_tokens":
            model_max = claude_max_output_tokens(model)
            remedy = (
                " — raise max_tokens toward it."
                if max_tokens < model_max
                else ", so the schema is too large for one response: split the "
                "request or narrow the schema."
            )
            raise StructuredOutputError(
                f"Claude hit the {max_tokens}-token output cap before finishing the "
                f"structured output, so the JSON is truncated. This model's "
                f"maximum is {model_max}{remedy}",
                context={
                    **context,
                    "max_tokens": max_tokens,
                    "model_max_tokens": model_max,
                    "output_tokens": getattr(response.usage, "output_tokens", None),
                },
            )

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        _ = temperature  # see generate_structured_json — not forwarded on Anthropic
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": resolve_max_tokens(model, max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            create_kwargs["system"] = system
        response = self._send(**create_kwargs)
        return "".join(b.text for b in response.content if b.type == "text")

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
        temperature: float = 0.1,  # not forwarded — see structured-output note
        max_tokens: int | None = None,
    ) -> Response:
        _ = temperature
        if cache is not None:
            return self._chat_cached(
                model=model, cache=cache, tail=messages, effort=effort, max_tokens=max_tokens
            )
        system_text, rest = split_system(messages, system)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": resolve_max_tokens(model, max_tokens),
            "messages": encode_anthropic_messages(rest),
        }
        if system_text:
            create_kwargs["system"] = system_text
        if tools:
            create_kwargs["tools"] = encode_anthropic_tools(tools)
            create_kwargs["tool_choice"] = encode_anthropic_tool_choice(
                tool_choice, force_tool
            )
        create_kwargs.update(anthropic_reasoning_kwargs(effort))
        raw = self._send(**create_kwargs)
        return decode_anthropic_response(raw)

    def create_cache(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[Tool] | None = None,
        ttl: int | None = None,
    ) -> CacheHandle:
        """Anthropic caching is inline and stateless, so there's no server
        resource to create — the handle just captures the prefix to replay with a
        ``cache_control`` breakpoint on each use. Pure (no network / no key needed).

        A prefix below Anthropic's ~1024-token minimum is simply not cached by the
        server (no error), matching the provider's own behavior.
        """
        return CacheHandle(
            provider=self.spec.name,
            model=model,
            system=system,
            messages=tuple(messages),
            tools=tuple(tools or ()),
            ttl=ttl,
        )

    def _chat_cached(
        self, *, model: str, cache: CacheHandle, tail: list[Message], effort, max_tokens
    ) -> Response:
        """Replay the cached prefix (system + messages + tools) with a single
        ``cache_control`` breakpoint at its deepest segment — which caches the
        whole contiguous prefix above it — then append the dynamic ``tail``."""
        tail_system, tail_rest = split_system(tail, None)
        prefix_messages = list(cache.messages)
        prefix_tools = list(cache.tools)
        # Wire order is tools -> system -> messages; the breakpoint goes on the
        # deepest present segment so everything above it is cached.
        deepest = (
            "messages" if prefix_messages
            else "system" if cache.system
            else "tools" if prefix_tools
            else None
        )

        encoded_prefix = encode_anthropic_messages(prefix_messages)
        if deepest == "messages" and encoded_prefix and encoded_prefix[-1]["content"]:
            encoded_prefix[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": resolve_max_tokens(model, max_tokens),
            "messages": encoded_prefix + encode_anthropic_messages(tail_rest),
        }

        system_blocks: list[dict] = []
        if cache.system:
            block: dict[str, Any] = {"type": "text", "text": cache.system}
            if deepest == "system":
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(block)
        if tail_system:  # any system on the tail rides along uncached, after the prefix
            system_blocks.append({"type": "text", "text": tail_system})
        if system_blocks:
            create_kwargs["system"] = system_blocks

        if prefix_tools:
            encoded_tools = encode_anthropic_tools(prefix_tools)
            if deepest == "tools":
                encoded_tools[-1]["cache_control"] = {"type": "ephemeral"}
            create_kwargs["tools"] = encoded_tools

        create_kwargs.update(anthropic_reasoning_kwargs(effort))
        raw = self._send(**create_kwargs)
        return decode_anthropic_response(raw)

    def verify_key(self) -> tuple[bool, str]:
        try:
            self._client.models.list(limit=1)
            return True, "Key is valid."
        except Exception as exc:  # noqa: BLE001
            detail = str(exc).strip() or exc.__class__.__name__
            return False, detail.splitlines()[0][:300]
