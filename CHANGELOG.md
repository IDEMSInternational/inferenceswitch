# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] — 2026-10-12

### Fixed

- **`select_model` no longer offers Gemini 2.5 models.** Google now answers
  `gemini-2.5-pro` and `gemini-2.5-flash-lite` with 404 "no longer available
  to new users", and limits the whole 2.5 line to accounts that used it before.
  With both a Gemini and an Anthropic key, `select_model("large",
  structured_output="reliable")` picked `gemini/gemini-2.5-pro` (the cheapest
  `large` card), so the first call failed. The `gemini-2.5-pro`,
  `gemini-2.5-flash` and `gemini-2.5-flash-lite` cards are removed, and
  `gemini-2.5-flash` leaves the Gemini `known_models`. Passing these ids to
  `chat()` still works for accounts that have access, and those accounts can
  merge the cards back in with `default_cards().merged(...)`. The remaining
  Gemini cards were checked against the live API on 2026-10-12.
  `gemini-3.1-pro-preview` is still the Gemini `large` card. Google offers it
  only on paid tiers, so a free-tier key gets a 429 on it
  ([#12](https://github.com/IDEMSInternational/inferenceswitch/issues/12)).
- A test now fails if any card routes to a model listed as retired
  (`RETIRED_MODELS` in `tests/test_model_cards.py`). `docs/model-cards.md`
  says when to add to that list.

## [0.4.0] — 2026-10-12

Adds a provider allow-list to model selection
([#9](https://github.com/IDEMSInternational/inferenceswitch/issues/9)).
Callers pinned to `<0.4` need to raise the pin to use it; nothing existing is
removed.

### Added

- **Provider allow-list for `select_model`**
  ([#9](https://github.com/IDEMSInternational/inferenceswitch/issues/9)).
  `providers=["anthropic", ...]` restricts selection to those registry
  providers, for reasons the ranking can't see (credits, contracts, data
  agreements). The filter applies before ranking, so the explanation still
  covers the choice. Routes on other providers are listed as excluded by the
  restriction and their availability is not checked. The reason says the
  candidates were restricted, `to_dict()` records the list under
  `request.providers` (`None` when unrestricted), and `NoModelAvailableError`
  leads with the restriction. Unregistered names, or an empty list, raise
  `ValueError`. Callers that don't pass `providers` see no change, except the
  new `providers` key in the request record.

## [0.3.0] — 2026-10-11

Adds model cards and selection by rung
([#7](https://github.com/IDEMSInternational/inferenceswitch/issues/7)). Callers
pinned to `<0.3` need to raise the pin to use them; nothing existing is removed.

### Fixed

- **OpenAI reasoning models no longer 400 on every call.** The OpenAI-compatible
  adapter always sent `temperature` (default `0.1`) and the output cap as
  `max_tokens`. GPT-5 and later, and the o-series, reject both. For the
  `openai` provider the cap is now sent as `max_completion_tokens`, which every
  OpenAI chat model accepts. `temperature` is sent only to models that accept
  sampling parameters: per model, from a new
  `Capabilities.model_sampling_params` table (on OpenAI, the `gpt-4*` and
  `gpt-3.5` lines) and any model card. Other OpenAI-compatible servers
  (Mistral, DeepSeek, Groq, Ollama, LM Studio) are unchanged.
- **`Effort.NONE` on OpenAI sends a level each model accepts.** It was always
  `"minimal"`, which GPT-5.1 and later reject. It is now `"none"` on GPT-5.1
  and later (including GPT-5.4, 5.6 and GPT-6 Sol / Luna), `"minimal"` on the
  original GPT-5 models, and `"low"` on models with neither (GPT-6 Astra,
  6.1 Sol, the o-series, and unknown models). The table is
  `config["reasoning_none"]` on the `openai` registry entry.
- **`Effort.MAX` on OpenAI asks for the most each model accepts.** It was
  always clamped to `"high"`. It is now `"max"` on GPT-5.6 and GPT-6
  (including 6.1), `"xhigh"` on GPT-5.4, and still `"high"` elsewhere (the
  original GPT-5 line, 5.1, the o-series, unknown models). The table is
  `config["reasoning_max"]`.
- **Claude Haiku 5.5's default output cap is 128K**, not the 64K fallback for
  unknown models: `claude-haiku-5` joins `CLAUDE_MAX_OUTPUT_TOKENS`.
- **The Gemini catalog uses Google's current ids.** `known_models` was
  `gemini-3-flash` (not an id Google documents; the model is
  `gemini-3-flash-preview`) and `gemini-3.5-flash` (now served by 3.6 Flash).
  It is now `gemini-2.5-flash`, `gemini-3-flash-preview`, `gemini-3.6-flash`
  and `gemini-3.8-flash`. Bare `gemini-*` names still route by prefix, so code
  passing the old names is unaffected; only `known_models()` and `models_for`
  list different entries. `gemini-3.8-flash` joins the output-cap table.
- **OpenAI structured output is enforced** (`strict: true`). The schema was
  sent as non-strict guidance. It is now translated to the strict subset by the
  new `inferenceswitch.schema.to_openai_strict_schema`: every object is closed
  and lists every property as required, optional properties become nullable,
  maps become key/value arrays, `oneOf` becomes `anyOf`, and keywords strict
  mode lacks (`minLength`, `maxLength`, ...) move into the description. On the
  way back, `restore_openai_strict` drops the nulls for optional fields and
  folds maps back, and the result is checked against your original schema
  (`StructuredOutputError` on a mismatch). A schema strict mode cannot express
  (a free-form object, a multi-entry `allOf`, a non-object root) is sent
  non-strict, as before. OpenAI model cards are graded `reliable` accordingly.
  Other OpenAI-compatible providers are unchanged.
- **DeepSeek structured output** now asks for `json_object` with the schema in
  the prompt, validated client-side. DeepSeek documents no `json_schema`
  response format, which is what was sent before. `STRICT_SCHEMA_OUTPUT` is no
  longer advertised for `deepseek`.

### Added

- **Model cards**: a versioned data file shipped in the package
  (`inferenceswitch/model_cards.json`, read with `json`) with one card per
  model: a vendor-neutral rung (`nano` / `small` / `mid` / `large` /
  `frontier`, provisional), a structured-output grade (`reliable` /
  `best_effort` / `none`), context window, list price per provider route,
  per-model capability overrides, and family, parameter count and quantization
  for open-weight models. 50 seed cards: Anthropic (Claude 4.x and 5.x),
  Gemini (2.5 through 3.8), OpenAI (GPT-6, 5.6, 5.4 and the 4.x non-reasoning
  models), Mistral (Large 4/3, Medium 3.5, Small 4, Ministral 3), DeepSeek
  (Flash, V4 Pro), Groq (gpt-oss, Qwen3.8 27B), and Qwen3, Llama and gpt-oss
  on Ollama and LM Studio. Prices are list prices checked on 2026-10-11. `ModelCard`, `ModelCards`, `default_cards()`; consumers
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

- `known_models` for OpenAI (GPT-6, 5.6, 5.4 and 4.x), Mistral (Large 4/3,
  Medium 3.5, Small 4, Ministral 3) and DeepSeek (Flash, V4 Pro), each with a
  card. They route by prefix as before; they now appear in `known_models()`
  and `models_for` as routable ids instead of a bare provider name.
- The Anthropic catalog (`known_models`) adds the Claude 5.x models: Haiku 5.5,
  Sonnet 5 and 5.5, Opus 5 and 5.5, Fable 5 and 5.1. They were already
  routable by prefix; they now appear in `known_models()` and `models_for`.

- **Capabilities are per model where a card says so.** `supports`, `require`
  and `capabilities` given `model=` apply that model's card on top of the
  provider defaults, and `chat`'s capability gates use them. Today this only
  adds: `SAMPLING_PARAMS` is reported for Claude Haiku 4.5 and Sonnet 4.6. A
  card that removes a feature makes `chat` raise `UnsupportedCapabilityError`
  for that model. Provider-level calls (`supports(cap, "anthropic")`,
  `providers_for`, `models_for`) are unchanged.
- The `openai` provider no longer advertises `SAMPLING_PARAMS` at the provider
  level (support is per model, as on Anthropic):
  `supports(Capability.SAMPLING_PARAMS, "openai")` is now `False`, and `True`
  for `model="openai/gpt-4o-mini"` and the other models that accept it.

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
