# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
