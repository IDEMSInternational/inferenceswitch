# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-10-11

Adds model cards and selection by rung
([#7](https://github.com/IDEMSInternational/inferenceswitch/issues/7)). Callers
pinned to `<0.3` need to raise the pin to use them; nothing existing is removed.

### Added

- **Model cards**: a versioned data file shipped in the package
  (`inferenceswitch/model_cards.json`, read with `json`) with one card per
  model: a vendor-neutral rung (`nano` / `small` / `mid` / `large` /
  `frontier`, provisional), a structured-output grade (`reliable` /
  `best_effort` / `none`), context window, list price per provider route,
  per-model capability overrides, and family, parameter count and quantization
  for open-weight models. Seeded for the Anthropic and Gemini models in the
  registry and for Qwen3 4B / 8B / 32B, Llama 3.1 8B and Llama 3.3 70B on
  Ollama and LM Studio. `ModelCard`, `ModelCards`, `default_cards()`; consumers
  add or override cards with `default_cards().merged(...)` and
  `Client(cards=...)`. Placement rubric in `docs/model-cards.md`.
- **`Client.select_model(rung, needs=..., structured_output=..., min_context=...,
  local_only=..., prefer="local" | "cheapest" | "fastest")`** picks an
  available model: providers with a key, and local servers that answer
  `GET /models` (only the models they list count). The rung is a floor (this
  rung or above; `allow_higher=False` for exactly this rung). Returns a
  `Selection` with the routable `model`, the `reason` it won, every candidate
  with why it was excluded, and `to_dict()` for run records. Raises
  `NoModelAvailableError` naming the unavailable providers and the
  requirements that excluded the rest.
- `Client.availability()`, `Client.card(model)`, `ProviderAvailability`,
  `static_availability()` and `Client(availability=...)` to declare
  availability instead of checking it (tests, environments that must not probe).

### Changed

- **Capabilities are per model where a card says so.** `supports`, `require`
  and `capabilities` given `model=` apply that model's card on top of the
  provider defaults, and `chat`'s capability gates use them. Today this only
  adds: `SAMPLING_PARAMS` is reported for Claude Haiku 4.5 and Sonnet 4.6. A
  card that removes a feature makes `chat` raise `UnsupportedCapabilityError`
  for that model. Provider-level calls (`supports(cap, "anthropic")`,
  `providers_for`, `models_for`) are unchanged.

## [0.2.0] — 2026-10-11

0.2 rather than 0.1.1: besides the fixes, this adds API and narrows what the
Anthropic adapter accepts (see *Changed*). Callers pinned to `<0.2` need to
raise the pin.

### Fixed

- **Tool-call markup leaking into a string field is repaired** on the Anthropic
  tool paths (`TOOL_USE`, `STRICT_TOOL_USE`). A tool call sometimes ran one
  string argument on into the next, e.g. a `caveats` value ending
  `...</caveats>\n<parameter name="next_step">...` with `next_step` missing
  ([terraso-planning#369](https://github.com/IDEMSInternational/terraso-planning/pull/369),
  [#332](https://github.com/IDEMSInternational/terraso-planning/issues/332)). Trailing closing tags (the field's own,
  `</parameter>`, `</invoke>`, `</function_calls>`) are now stripped from
  top-level string fields, and each `<parameter name="X">value` segment is moved
  back out as field `X` (JSON-decoded when it parses), never overwriting a field
  already present. Models on structured outputs, which is every Claude 5.x model
  including Sonnet 5 and Opus 5, do not go through a tool and are unaffected.
- **Anthropic structured JSON now uses structured outputs** (`output_config.format`,
  constrained decoding) instead of a forced, non-strict tool
  ([#1](https://github.com/IDEMSInternational/inferenceswitch/issues/1)).
  `generate_structured_json` works on Claude Opus 5.5, Sonnet 5.5, Fable 5.1 and
  Mythos 5.1, which reject forced `tool_choice` with a 400, and the result can no
  longer arrive nested under a stray key such as `{"$PARAMETER_NAME": {...}}`.
- **`chat(force_tool=...)` and `ToolChoice.REQUIRED` work on Claude Opus 5.5,
  Sonnet 5.5, Fable 5.1 and Mythos 5.1**, which reject `tool_choice` `"tool"`
  and `"any"` with a 400. On those models the request goes out with
  `tool_choice: auto` and a system-prompt line asking for the tool (appended
  after any system prompt you passed). Since the model can still answer without
  calling it, a turn with no call to the forced tool (or, for `REQUIRED`, no
  tool call at all) raises `ToolChoiceError`. Older Claude models are unchanged:
  they still get a forced `tool_choice`, and a turn without the call is returned
  as before.
- Gemini schema translation no longer rewrites an object carrying
  `additionalProperties: false` (as Pydantic's `extra="forbid"` emits) into a
  key/value array. The boolean is dropped instead, so callers no longer need to
  strip it before calling.

### Added

- `Client.generate_structured`, which returns a `StructuredResult` with the
  parsed `value`, the call's token `usage`, its `stop_reason` and the `raw`
  provider response
  ([#4](https://github.com/IDEMSInternational/inferenceswitch/issues/4)). Each
  adapter fills `Usage` with the same mapping as its `chat` path; fields a
  provider doesn't report are `None`. `generate_structured_json` is unchanged and
  returns `generate_structured(...).value`.
- `inferenceswitch.schema.to_anthropic_schema`: translates a schema to the
  dialect Anthropic structured outputs accept. It closes every object, rewrites
  maps to key/value arrays (folded back on the way out, as for Gemini), and
  removes unsupported constraints (`minimum`, `maximum`, `multipleOf`,
  `minLength`, `maxLength`, `pattern`, ...). Recursive schemas, external `$ref`
  and free-form objects raise the new `UnsupportedSchemaError` before any request
  is sent. `required` is kept as the caller wrote it.
- `inferenceswitch.schema.schema_errors`: a small dependency-free validator. The
  Anthropic adapter checks every structured result against the caller's original
  schema, including the constraints removed for the request, and raises
  `StructuredOutputError` (with `context["schema_errors"]`) on a mismatch.
- Per-model structured-output mechanism: `Capabilities.model_structured_output`
  and `Capabilities.structured_output_for(model)`. New `StructuredMode` values
  `OUTPUT_CONFIG_JSON_SCHEMA`, `STRICT_TOOL_USE` and `TOOL_USE`. Anthropic
  defaults to structured outputs, and models that predate them (Claude 3.x, Opus
  4 / 4.1, Sonnet 4) fall back to a tool offered with `tool_choice: auto` and an
  instruction to call it. On that non-strict path, an answer wrapped in a single
  stray key is unwrapped only when the key is not a schema property and the
  inner value validates.
- Per-model forced tool choice: `Capabilities.forced_tool_choice`,
  `Capabilities.model_forced_tool_choice` and
  `Capabilities.forced_tool_choice_for(model)`, matched like
  `model_structured_output`. Anthropic defaults to no forced choice, so a model
  newer than the table is not sent a request it would reject. Claude 3.x, 4.x,
  Opus 5, Sonnet 5, Fable 5 and Mythos 5 keep forced choice.
- `ToolChoiceError`, carrying the turn that came back as `.response` and the
  provider, model, requested choice, stop reason and tools called in `.context`.
- `StructuredOutputError` for an Anthropic refusal (`stop_reason: "refusal"`,
  with the category in `context["refusal_category"]`) and for a fallback
  response that did not call the tool.

### Changed

- `StructuredMode.FORCED_TOOL_USE` is a deprecated alias of `TOOL_USE`; the tool
  is no longer forced.
- The `anthropic` extra now requires `anthropic>=0.77`, the first SDK release
  with `output_config` on the Messages API.
- A free-form `{"type": "object"}` schema is rejected on Anthropic models that
  use structured outputs, because the dialect cannot express it.

## [0.1.0] — first public release

Initial release. Developed privately as `llmswitch` and renamed for
publication: that name was crowded on both PyPI and GitHub, and `llm` was too
narrow for a library that already dispatches to local small-model servers
(`ollama`, `lmstudio`) alongside the hosted providers.

- Static provider registry with prefix, `provider/model`, and explicit routing.
  Unroutable or ambiguous model strings raise `ProviderResolutionError` — there
  is no silent default provider.
- Uniform structured-JSON and plain-text generation across Anthropic, Gemini,
  OpenAI, and OpenAI-compatible providers (Mistral, DeepSeek, Groq, Ollama,
  LM Studio).
- Schema-dialect translation for Gemini's `response_schema`, which rejects
  `additionalProperties`: dict-typed nodes are rewritten to key/value arrays and
  folded back on the way out. Driven by the caller's own schema, including
  `$defs` / `$ref` and `anyOf`-wrapped optional dicts.
- Per-model output caps, with a process-wide table as the control surface.
- Portable prompt-caching surface across providers.
- Capability introspection (`models_for`, effort control, assertions).
- Zero required dependencies; provider SDKs are optional extras, lazily imported.

The entry points are `Client` and `Response`, unprefixed like every other
export (`Message`, `Tool`, `Usage`, `Capability`). They were `LLMClient` and
`LLMResponse` during private development; the prefix was dropped for the same
reason `llm` left the package name, and because the package already qualifies
them.
