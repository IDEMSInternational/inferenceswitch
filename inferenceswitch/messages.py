"""Provider-neutral message / tool / response model.

This is the normalized vocabulary the ``chat`` surface speaks, so a workflow
writes one shape and the adapters translate it into each provider's native form
(OpenAI ``tool_calls`` arrays, Anthropic ``tool_use``/``tool_result`` blocks,
Gemini ``functionCall``/``functionResponse`` parts). It is modeled on Anthropic's
content-block layout because that is the superset — the other two shapes derive
from it cleanly.

Ergonomic constructors (:func:`user`, :func:`assistant`, :func:`tool_result`)
cover the common path; the dataclasses are there when you need to build content
by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

# ── content parts ─────────────────────────────────────────────────────────────


@dataclass
class Text:
    text: str
    #: Mark this block as an explicit prompt-cache breakpoint. Honored only by
    #: providers with EXPLICIT_PROMPT_CACHING (Anthropic ``cache_control``); the
    #: client raises :class:`UnsupportedCapabilityError` if the routed provider
    #: lacks it, so the marker never degrades to a silent no-op.
    cache: bool = False


@dataclass
class ToolUse:
    """An assistant's request to call a tool (appears in an assistant message)."""

    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    """The result of a tool call (appears in a user message).

    ``name`` is carried alongside ``tool_call_id`` because providers disagree on
    the join key — OpenAI/Anthropic match results to calls by id, Gemini matches
    by function *name*. Keeping both lets one normalized result serve all three.
    """

    tool_call_id: str
    content: str
    is_error: bool = False
    name: str | None = None


ContentPart = Union[Text, ToolUse, ToolResult]


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: Union[str, list[ContentPart]]

    def parts(self) -> list[ContentPart]:
        """Content as a part list (wrapping a bare string in a single Text)."""
        if isinstance(self.content, str):
            return [Text(self.content)]
        return self.content


# ── tools ─────────────────────────────────────────────────────────────────────


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    #: Place an explicit cache breakpoint on this tool (Anthropic caches the tool
    #: set up to and including it). Same capability gate as :attr:`Text.cache`.
    cache: bool = False


@dataclass
class ToolCall:
    """A tool call the model wants made (parsed out of a response)."""

    id: str
    name: str
    input: dict


class ToolChoice(str, Enum):
    """How eagerly the model may call tools. To force one specific tool, pass its
    name as a string to ``chat(force_tool=...)`` instead."""

    AUTO = "auto"          # model decides
    REQUIRED = "required"  # must call at least one tool
    NONE = "none"          # tools visible but must not be called


class Effort(str, Enum):
    """Normalized reasoning effort, mapped to each provider's own knob:
    Anthropic ``output_config.effort`` + adaptive thinking, OpenAI
    ``reasoning_effort``, Gemini thinking budget. ``NONE`` disables reasoning
    where the provider allows it."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


# ── response ──────────────────────────────────────────────────────────────────


class StopReason(str, Enum):
    """Normalized reason a turn ended, unified across providers' own vocabularies."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    STOP_SEQUENCE = "stop_sequence"
    OTHER = "other"


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class CacheHandle:
    """A reusable prompt cache created once via :meth:`LLMClient.create_cache`
    and passed to :meth:`~LLMClient.chat` as ``cache=`` to avoid resending — and
    re-billing — a large static prefix. One handle type, two backings:

    * **Server-backed** (Gemini ``CachedContent``): ``name`` references a
      provider-side resource; the prefix is not resent over the wire.
    * **Replay-backed** (Anthropic): there is no server resource — the handle
      *holds* the prefix (:attr:`system`, :attr:`messages`, :attr:`tools`), which
      the adapter resends with a ``cache_control`` breakpoint each call, letting
      Anthropic's stateless prefix cache do the work.

    Either way the caller holds the handle, so reuse (and its cost) is explicit.
    A handle is not portable across providers. Requires the
    :class:`~inferenceswitch.Capability` REUSABLE_PROMPT_CACHE.
    """

    provider: str          # the provider that created it — a handle is not portable
    model: str             # the model the cache is pinned to
    #: Server-backed id (e.g. ``"cachedContents/abc123"``); ``None`` when replay-backed.
    name: str | None = None
    #: Replay-backed prefix (Anthropic). Empty for server-backed handles.
    system: str | None = None
    messages: tuple = ()   # tuple[Message, ...]
    tools: tuple = ()      # tuple[Tool, ...]
    ttl: int | None = None  # requested TTL in seconds
    #: The untouched provider SDK cache object, if any (server-backed only).
    raw: Any = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    usage: Usage = field(default_factory=Usage)
    #: The untouched provider SDK response, for anything not normalized here.
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ── ergonomic constructors ────────────────────────────────────────────────────


def user(text: str) -> Message:
    return Message(role="user", content=text)


def assistant(text: str) -> Message:
    return Message(role="assistant", content=text)


def system(text: str) -> Message:
    return Message(role="system", content=text)


def tool_result(call: ToolCall, content: str, *, is_error: bool = False) -> Message:
    """A user message carrying one tool result, keyed back to ``call`` (so the
    provider-specific id/name join is handled for you)."""
    return Message(
        role="user",
        content=[
            ToolResult(
                tool_call_id=call.id,
                content=content,
                is_error=is_error,
                name=call.name,
            )
        ],
    )
