"""OpenAI-compatible adapter — one class for many providers.

Covers OpenAI itself and every server that speaks the OpenAI Chat Completions
wire format: Mistral, DeepSeek, Groq, and local Ollama / LM Studio. The only
differences between them are ``base_url``, key env, and (per capability) how
strictly they enforce a JSON schema — all of which live in the registry, so
adding another OpenAI clone is a data change, not a code change.
"""
from __future__ import annotations

import json
from typing import Any

from collections.abc import Mapping

from .. import schema as schema_mod
from ..capabilities import Capability, StructuredMode, _for_model
from ..errors import StructuredOutputError, UnsupportedSchemaError
from ..messages import (
    Effort,
    Response,
    Message,
    StopReason,
    StreamCallback,
    StreamDelta,
    StructuredResult,
    Text,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolUse,
    Usage,
)
from .base import Adapter, require_sdk, split_system

# OpenAI reasoning_effort levels. NONE and MAX are per model when the provider
# declares "reasoning_none" / "reasoning_max" tables; without them NONE is
# "minimal" and MAX clamps to "high", which the oldest reasoning models accept.
_OPENAI_EFFORT = {
    Effort.NONE: "minimal",
    Effort.LOW: "low",
    Effort.MEDIUM: "medium",
    Effort.HIGH: "high",
    Effort.MAX: "high",
}


def openai_reasoning_kwargs(
    effort: Effort | None, config: Mapping, model: str | None = None
) -> dict:
    """Request kwargs for reasoning effort — empty unless the provider declares a
    ``reasoning_param`` (only OpenAI does; others reason via model choice).

    :attr:`Effort.NONE` and :attr:`Effort.MAX` mean the least and the most
    reasoning the model accepts, which differ by model (``"none"`` /
    ``"minimal"`` / ``"low"``; ``"max"`` / ``"xhigh"`` / ``"high"``). A
    ``config["reasoning_none"]`` or ``config["reasoning_max"]`` table, matched
    by model substring (longest wins, as for the per-model capability tables)
    with a ``"*"`` default, chooses it; without a table, NONE is ``"minimal"``
    and MAX is ``"high"``.
    """
    if effort is None:
        return {}
    param = config.get("reasoning_param")
    if not param:
        return {}
    value = _OPENAI_EFFORT[effort]
    if effort in (Effort.NONE, Effort.MAX) and model is not None:
        value = _config_for_model(config, f"reasoning_{effort.value}", model, value)
    return {param: value}


def openai_strict_for(config: Mapping, model: str) -> bool:
    """Whether ``model`` gets ``strict: true`` on its ``json_schema``.

    ``config["strict_json_schema"]`` decides per model (substring, longest
    wins, ``"*"`` default); without it, ``config["openai_strict"]`` decides
    for every model.
    """
    return bool(
        _config_for_model(config, "strict_json_schema", model, config.get("openai_strict", False))
    )


def _config_for_model(config: Mapping, key: str, model: str, default: Any) -> Any:
    """``config[key]``'s entry for ``model``: a table matched like the
    per-model capability tables, with ``"*"`` as its own default."""
    table = config.get(key)
    if not table:
        return default
    rows = {k: v for k, v in table.items() if k != "*"}
    return _for_model(rows, model, table.get("*", default))


_FINISH_REASON = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "content_filter": StopReason.REFUSAL,
}


def encode_openai_messages(messages: list[Message]) -> list[dict]:
    """Normalized messages -> OpenAI Chat Completions message array."""
    out: list[dict] = []
    for message in messages:
        if message.role == "system":
            out.append({"role": "system", "content": _join_text(message)})
            continue
        if message.role == "assistant":
            tool_calls = [
                {
                    "id": p.id,
                    "type": "function",
                    "function": {"name": p.name, "arguments": json.dumps(p.input)},
                }
                for p in message.parts()
                if isinstance(p, ToolUse)
            ]
            entry: dict = {"role": "assistant", "content": _join_text(message) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue
        # user role: split into tool results (their own "tool" messages) + text
        results = [p for p in message.parts() if isinstance(p, ToolResult)]
        for r in results:
            out.append(
                {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
            )
        text = _join_text(message)
        if text or not results:
            out.append({"role": "user", "content": text})
    return out


def encode_openai_tools(tools: list[Tool]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def encode_openai_tool_choice(tool_choice: ToolChoice, force_tool: str | None):
    if force_tool is not None:
        return {"type": "function", "function": {"name": force_tool}}
    return tool_choice.value  # "auto" | "required" | "none"


def openai_usage(raw) -> Usage:
    """The response's token counts; ``Usage()`` when it reports none (some
    local servers omit ``usage`` entirely)."""
    usage = Usage()
    if getattr(raw, "usage", None) is not None:
        usage.input_tokens = getattr(raw.usage, "prompt_tokens", None)
        usage.output_tokens = getattr(raw.usage, "completion_tokens", None)
        details = getattr(raw.usage, "prompt_tokens_details", None)
        if details is not None:
            usage.cache_read_tokens = getattr(details, "cached_tokens", None)
        # ``prompt_tokens`` counts the cached tokens too; ``input_tokens`` is the
        # uncached rest, as on Anthropic, so each count has one price.
        if usage.input_tokens is not None and usage.cache_read_tokens:
            usage.input_tokens -= usage.cache_read_tokens
    return usage


def decode_openai_response(raw) -> Response:
    """OpenAI response object -> normalized Response."""
    choice = raw.choices[0]
    message = choice.message
    tool_calls: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

    return Response(
        text=message.content or "",
        tool_calls=tool_calls,
        stop_reason=_FINISH_REASON.get(choice.finish_reason, StopReason.OTHER),
        usage=openai_usage(raw),
        raw=raw,
    )


#: Where OpenAI-compatible servers put streamed reasoning, by the name each
#: uses: ``reasoning_content`` (DeepSeek, LM Studio, vLLM, xAI) and
#: ``reasoning`` (Groq with ``reasoning_format: "parsed"``, Ollama,
#: OpenRouter). OpenAI's own Chat Completions streams no reasoning.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def openai_stream_deltas(chunk) -> list[StreamDelta]:
    """The thinking and text a ``ChatCompletionChunk`` adds to the first choice."""
    out: list[StreamDelta] = []
    for choice in getattr(chunk, "choices", None) or []:
        if getattr(choice, "index", 0) != 0:
            continue
        delta = choice.delta
        for name in _REASONING_FIELDS:
            reasoning = getattr(delta, name, None)
            if isinstance(reasoning, str) and reasoning:
                out.append(StreamDelta("thinking", reasoning))
                break
        if delta.content:
            out.append(StreamDelta("text", delta.content))
    return out


def _join_text(message: Message) -> str:
    return "".join(p.text for p in message.parts() if isinstance(p, Text))


class OpenAICompatibleAdapter(Adapter):
    def _build_client(self, api_key: str) -> Any:
        openai = require_sdk("openai", "openai")
        # base_url None -> the SDK's default (api.openai.com).
        return openai.OpenAI(api_key=api_key, base_url=self.spec.base_url)

    def _create(self, on_stream: StreamCallback | None, **kwargs: Any) -> Any:
        """One ``chat.completions.create`` call. With ``on_stream`` it streams
        instead, hands each chunk's thinking and text to ``on_stream``, and
        returns the chunks accumulated into one ``ChatCompletion`` by the SDK's
        own ``ChatCompletionStreamState``, so the caller reads it as before.

        Usage is asked for with ``stream_options.include_usage``; a provider
        that rejects it sets ``config["stream_include_usage"] = False``."""
        if on_stream is None:
            return self._client.chat.completions.create(**kwargs)
        streaming = require_sdk("openai.lib.streaming.chat", "openai")
        state = streaming.ChatCompletionStreamState()
        if self.capabilities.config.get("stream_include_usage", True):
            kwargs["stream_options"] = {"include_usage": True}
        for chunk in self._client.chat.completions.create(**kwargs, stream=True):
            state.handle_chunk(chunk)
            for delta in openai_stream_deltas(chunk):
                on_stream(delta)
        return state.get_final_completion()

    def _sampling_kwargs(self, temperature: float | None, max_tokens: int | None) -> dict:
        """``temperature`` (omitted when ``None``: the client passes ``None``
        for models that reject it) and the output cap under the provider's
        parameter name (``max_tokens`` unless ``config["max_tokens_param"]``)."""
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs[self.capabilities.config.get("max_tokens_param", "max_tokens")] = max_tokens
        return kwargs

    def _messages(self, prompt: str, system: str | None) -> list[dict]:
        messages: list[dict] = []
        if system and self.capabilities.has(Capability.SYSTEM_PROMPT):
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict,
        system: str | None = None,
        tool_name: str = "generate_json",
        temperature: float | None = 0.1,
        max_tokens: int | None = None,
        cache_prefix: str | None = None,
        on_stream: StreamCallback | None = None,
    ) -> StructuredResult:
        mode = self.capabilities.structured_output
        if cache_prefix:
            # One string, prefix first: some compatible servers take only string
            # content, and a stable leading prefix is what implicit caching keys on.
            prompt = f"{cache_prefix}\n\n{prompt}" if prompt else cache_prefix
        kwargs: dict[str, Any] = {"model": model}
        kwargs.update(self._sampling_kwargs(temperature, max_tokens))

        strict = False
        if mode is StructuredMode.RESPONSE_FORMAT_JSON_SCHEMA:
            sent = schema
            if openai_strict_for(self.capabilities.config, model):
                try:
                    sent, strict = schema_mod.to_openai_strict_schema(schema), True
                except UnsupportedSchemaError:
                    # Strict mode cannot express this schema (a free-form
                    # object, a multi-entry allOf, ...): send it as guidance.
                    sent = schema
            json_schema: dict[str, Any] = {"name": tool_name, "schema": sent}
            if strict:
                json_schema["strict"] = True
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
            messages = self._messages(prompt, system)
        elif mode is StructuredMode.JSON_OBJECT_BEST_EFFORT:
            # Servers that take the OpenAI format but not per-field enforcement:
            # ask for a JSON object and put the schema in the prompt.
            kwargs["response_format"] = {"type": "json_object"}
            augmented = (
                f"{prompt}\n\n"
                "Return ONLY a single JSON object that conforms to this JSON Schema "
                "(no prose, no markdown fences):\n"
                f"{json.dumps(schema)}"
            )
            messages = self._messages(augmented, system)
        else:  # pragma: no cover - registry never assigns another mode to this kind
            raise StructuredOutputError(
                f"OpenAI-compatible adapter cannot honor structured mode {mode!r}.",
                context={"provider": self.spec.name, "model": model},
            )

        kwargs["messages"] = messages
        response = self._create(on_stream, **kwargs)
        content = response.choices[0].message.content
        if not content:
            raise StructuredOutputError(
                "Provider returned an empty completion for a structured request.",
                context={
                    "provider": self.spec.name,
                    "model": model,
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                },
            )
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"Provider returned non-JSON content: {exc}",
                context={"provider": self.spec.name, "model": model},
            ) from exc
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if strict:
            # Undo the strict rewrite (nulls for optional fields, maps as
            # key/value arrays), then check what strict mode could not enforce
            # (dropped keywords such as minLength) against the caller's schema.
            value = schema_mod.restore_openai_strict(value, schema)
            errors = schema_mod.schema_errors(value, schema)
            if errors:
                raise StructuredOutputError(
                    "Structured output does not match the schema: "
                    + "; ".join(errors[:5])
                    + (f" (and {len(errors) - 5} more)" if len(errors) > 5 else ""),
                    context={
                        "provider": self.spec.name,
                        "model": model,
                        "finish_reason": finish_reason,
                        "schema_errors": errors,
                    },
                )
        return StructuredResult(
            value=value,
            usage=openai_usage(response),
            stop_reason=(
                None if finish_reason is None
                else _FINISH_REASON.get(finish_reason, StopReason.OTHER)
            ),
            raw=response,
        )

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = 0.1,
        max_tokens: int | None = None,
        on_stream: StreamCallback | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._messages(prompt, system),
        }
        kwargs.update(self._sampling_kwargs(temperature, max_tokens))
        response = self._create(on_stream, **kwargs)
        return response.choices[0].message.content or ""

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
        cache=None,  # reusable-cache handle: not an OpenAI concept; never non-None here
        temperature: float | None = 0.1,
        max_tokens: int | None = None,
        on_stream: StreamCallback | None = None,
    ) -> Response:
        _ = cache
        system_text, rest = split_system(messages, system)
        encoded = encode_openai_messages(rest)
        if system_text:
            encoded.insert(0, {"role": "system", "content": system_text})

        kwargs: dict[str, Any] = {"model": model, "messages": encoded}
        kwargs.update(self._sampling_kwargs(temperature, max_tokens))
        if tools:
            kwargs["tools"] = encode_openai_tools(tools)
            kwargs["tool_choice"] = encode_openai_tool_choice(tool_choice, force_tool)
        kwargs.update(openai_reasoning_kwargs(effort, self.capabilities.config, model))

        raw = self._create(on_stream, **kwargs)
        return decode_openai_response(raw)

    def verify_key(self) -> tuple[bool, str]:
        try:
            # models.list is the cheapest authenticated call across OpenAI clones.
            self._client.models.list()
            return True, "Key is valid."
        except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
            detail = str(exc).strip() or exc.__class__.__name__
            return False, detail.splitlines()[0][:300]
