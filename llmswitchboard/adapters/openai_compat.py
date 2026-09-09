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

from ..capabilities import Capability, StructuredMode
from ..errors import StructuredOutputError
from ..messages import (
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

# OpenAI reasoning_effort accepts minimal/low/medium/high; MAX clamps to high.
_OPENAI_EFFORT = {
    Effort.NONE: "minimal",
    Effort.LOW: "low",
    Effort.MEDIUM: "medium",
    Effort.HIGH: "high",
    Effort.MAX: "high",
}


def openai_reasoning_kwargs(effort: Effort | None, config: Mapping) -> dict:
    """Request kwargs for reasoning effort — empty unless the provider declares a
    ``reasoning_param`` (only OpenAI does; others reason via model choice)."""
    if effort is None:
        return {}
    param = config.get("reasoning_param")
    if not param:
        return {}
    return {param: _OPENAI_EFFORT[effort]}


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


def decode_openai_response(raw) -> LLMResponse:
    """OpenAI response object -> normalized LLMResponse."""
    choice = raw.choices[0]
    message = choice.message
    tool_calls: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

    usage = Usage()
    if getattr(raw, "usage", None) is not None:
        usage.input_tokens = getattr(raw.usage, "prompt_tokens", None)
        usage.output_tokens = getattr(raw.usage, "completion_tokens", None)
        details = getattr(raw.usage, "prompt_tokens_details", None)
        if details is not None:
            usage.cache_read_tokens = getattr(details, "cached_tokens", None)

    return LLMResponse(
        text=message.content or "",
        tool_calls=tool_calls,
        stop_reason=_FINISH_REASON.get(choice.finish_reason, StopReason.OTHER),
        usage=usage,
        raw=raw,
    )


def _join_text(message: Message) -> str:
    return "".join(p.text for p in message.parts() if isinstance(p, Text))


class OpenAICompatibleAdapter(Adapter):
    def _build_client(self, api_key: str) -> Any:
        openai = require_sdk("openai", "openai")
        # base_url None -> the SDK's default (api.openai.com).
        return openai.OpenAI(api_key=api_key, base_url=self.spec.base_url)

    def _messages(self, prompt: str, system: str | None) -> list[dict]:
        messages: list[dict] = []
        if system and self.capabilities.has(Capability.SYSTEM_PROMPT):
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

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
        mode = self.capabilities.structured_output
        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if mode is StructuredMode.RESPONSE_FORMAT_JSON_SCHEMA:
            json_schema: dict[str, Any] = {"name": tool_name, "schema": schema}
            if self.capabilities.config.get("openai_strict", False):
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
        response = self._client.chat.completions.create(**kwargs)
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
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"Provider returned non-JSON content: {exc}",
                context={"provider": self.spec.name, "model": model},
            ) from exc

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": self._messages(prompt, system),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self._client.chat.completions.create(**kwargs)
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
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        _ = cache
        system_text, rest = split_system(messages, system)
        encoded = encode_openai_messages(rest)
        if system_text:
            encoded.insert(0, {"role": "system", "content": system_text})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": encoded,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = encode_openai_tools(tools)
            kwargs["tool_choice"] = encode_openai_tool_choice(tool_choice, force_tool)
        kwargs.update(openai_reasoning_kwargs(effort, self.capabilities.config))

        raw = self._client.chat.completions.create(**kwargs)
        return decode_openai_response(raw)

    def verify_key(self) -> tuple[bool, str]:
        try:
            # models.list is the cheapest authenticated call across OpenAI clones.
            self._client.models.list()
            return True, "Key is valid."
        except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
            detail = str(exc).strip() or exc.__class__.__name__
            return False, detail.splitlines()[0][:300]
