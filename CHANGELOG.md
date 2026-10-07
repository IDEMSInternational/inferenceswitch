# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Anthropic structured JSON now uses structured outputs** (`output_config.format`,
  constrained decoding) instead of a forced, non-strict tool
  ([#1](https://github.com/IDEMSInternational/inferenceswitch/issues/1)).
  `generate_structured_json` works on Claude Opus 5.5, Sonnet 5.5, Fable 5.1 and
  Mythos 5.1, which reject forced `tool_choice` with a 400, and the result can no
  longer arrive nested under a stray key such as `{"$PARAMETER_NAME": {...}}`.
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
