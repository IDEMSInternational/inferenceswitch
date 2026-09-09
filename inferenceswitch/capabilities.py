"""Capability model — the abstraction that replaces provider-name branching.

Two kinds of thing live here:

* **Discriminators** — single-valued mechanism selectors the adapters dispatch
  on: how a provider is forced to emit JSON (:class:`StructuredMode`) and which
  schema dialect it accepts (:class:`SchemaDialect`).
* **Feature flags** — a :class:`frozenset` of :class:`Capability` tokens
  describing what the *provider* can do (tool calling, vision, explicit prompt
  caching, ...), plus an open ``config`` map for per-capability tuning.

Feature flags describe the **provider**, independent of whether inferenceswitch yet
exposes a first-class method for that feature. A workflow uses them to branch or
degrade deliberately (``client.supports(...)`` / ``client.require(...)``); for a
capability the library doesn't wrap yet, drop to ``client.raw_client(...)``. New
capabilities are additive — a new token in the enum and the relevant provider
sets, no change to every ProviderSpec.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StructuredMode(str, Enum):
    """How a provider is made to return schema-valid JSON (a discriminator)."""

    #: OpenAI Chat Completions ``response_format={"type":"json_schema",...}``.
    RESPONSE_FORMAT_JSON_SCHEMA = "response_format_json_schema"
    #: ``response_format={"type":"json_object"}`` + schema injected into the
    #: prompt. Fallback for local/older servers without per-field enforcement.
    JSON_OBJECT_BEST_EFFORT = "json_object_best_effort"
    #: Anthropic: schema as a tool ``input_schema`` + forced ``tool_choice``.
    FORCED_TOOL_USE = "forced_tool_use"
    #: Gemini: ``response_mime_type`` + ``response_schema`` (needs dialect xlate).
    RESPONSE_SCHEMA = "response_schema"


class SchemaDialect(str, Enum):
    """Which JSON-Schema variant a provider accepts (a discriminator)."""

    STANDARD = "standard"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    #: Gemini's OpenAPI subset: rejects ``additionalProperties`` — the divergence
    #: a plain OpenAI shim cannot handle. See :mod:`inferenceswitch.schema`.
    GEMINI = "gemini"


class Capability(str, Enum):
    """A provider feature a workflow can discover and gate on.

    Presence means *the provider supports it*. Whether inferenceswitch exposes a
    first-class method for it is separate — see the module docstring.
    """

    # ── conversation shape ────────────────────────────────────────────────
    MULTI_TURN = "multi_turn"                    # accepts a message history
    SYSTEM_PROMPT = "system_prompt"              # honors a system instruction
    SAMPLING_PARAMS = "sampling_params"          # temperature / top_p / top_k
    STREAMING = "streaming"                      # incremental token stream

    # ── structured output & tools ─────────────────────────────────────────
    STRICT_SCHEMA_OUTPUT = "strict_schema_output"  # enforced (not best-effort) JSON schema
    TOOL_CALLING = "tool_calling"                # general function/tool calling
    PARALLEL_TOOL_USE = "parallel_tool_use"      # multiple tool calls per turn

    # ── multimodal input ──────────────────────────────────────────────────
    VISION = "vision"                            # image input
    PDF_INPUT = "pdf_input"                      # native PDF/document input
    AUDIO_INPUT = "audio_input"                  # audio input

    # ── advanced / cost levers ────────────────────────────────────────────
    REASONING_EFFORT = "reasoning_effort"        # thinking / reasoning-effort control
    # Two distinct caching contracts — different mechanisms, different providers:
    EXPLICIT_PROMPT_CACHING = "explicit_prompt_caching"  # inline caller-placed breakpoints (Anthropic cache_control)
    IMPLICIT_PROMPT_CACHING = "implicit_prompt_caching"  # automatic prefix caching, no caller control (OpenAI/DeepSeek/Gemini)
    REUSABLE_PROMPT_CACHE = "reusable_prompt_cache"  # create-once/reference-many named cache handle (Gemini CachedContent)
    TOKEN_COUNTING = "token_counting"            # dedicated count-tokens endpoint
    BATCH = "batch"                              # async batch API (usually ~50% cost)


@dataclass(frozen=True)
class Capabilities:
    """What a provider can do, declared once on its registry entry."""

    structured_output: StructuredMode
    schema_dialect: SchemaDialect
    features: frozenset[Capability] = frozenset()
    #: Open per-capability tuning, e.g. ``{"openai_strict": True}`` or
    #: ``{"reasoning_effort_levels": ["low", "high"]}``. String-keyed so new
    #: capabilities can attach config without touching this class.
    config: Mapping[str, Any] = field(default_factory=dict)

    def has(self, capability: Capability) -> bool:
        return capability in self.features
