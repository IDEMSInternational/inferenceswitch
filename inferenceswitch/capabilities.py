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
    #: Anthropic structured outputs: ``output_config={"format": {"type":
    #: "json_schema", ...}}``. Constrained decoding; needs the Anthropic dialect.
    OUTPUT_CONFIG_JSON_SCHEMA = "output_config_json_schema"
    #: Anthropic: schema as a ``strict: true`` tool's ``input_schema``, offered
    #: with ``tool_choice: auto`` and a prompt instruction to call it. Strict
    #: tools are constrained like structured outputs.
    STRICT_TOOL_USE = "strict_tool_use"
    #: Anthropic, models without structured outputs: the same tool, non-strict,
    #: so the schema is guidance only and the result is validated client-side.
    TOOL_USE = "tool_use"
    #: Deprecated alias of :attr:`TOOL_USE`. The tool is no longer forced:
    #: forced ``tool_choice`` returns a 400 on the newest Claude models.
    FORCED_TOOL_USE = "tool_use"
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
    #: Per-model overrides of :attr:`structured_output`, for providers whose
    #: models differ in mechanism. Keys match a model ID exactly or as a
    #: substring (longest wins), so dated snapshots and provider-prefixed IDs
    #: resolve like the bare alias. See :meth:`structured_output_for`.
    model_structured_output: Mapping[str, StructuredMode] = field(default_factory=dict)
    #: Whether the provider honors a *forced* tool choice — one named tool
    #: (``chat(force_tool=...)``) or "some tool" (``ToolChoice.REQUIRED``). Where
    #: it does not, the adapter offers the tools with an automatic choice plus a
    #: prompt instruction, and raises :class:`~inferenceswitch.ToolChoiceError`
    #: on a turn that ignored it.
    forced_tool_choice: bool = True
    #: Per-model overrides of :attr:`forced_tool_choice`, matched like
    #: :attr:`model_structured_output`. See :meth:`forced_tool_choice_for`.
    model_forced_tool_choice: Mapping[str, bool] = field(default_factory=dict)

    def has(self, capability: Capability) -> bool:
        return capability in self.features

    def structured_output_for(self, model: str) -> StructuredMode:
        """The structured-output mechanism for ``model`` on this provider."""
        return _for_model(self.model_structured_output, model, self.structured_output)

    def forced_tool_choice_for(self, model: str) -> bool:
        """Whether ``model`` on this provider honors a forced tool choice."""
        return _for_model(self.model_forced_tool_choice, model, self.forced_tool_choice)


def _for_model(table: Mapping[str, Any], model: str, default: Any) -> Any:
    """``table``'s entry for ``model``: an exact key, else the longest key that is
    a substring of ``model``, else ``default``."""
    exact = table.get(model)
    if exact is not None:
        return exact
    matches = [name for name in table if name in model]
    if matches:
        return table[max(matches, key=len)]
    return default
