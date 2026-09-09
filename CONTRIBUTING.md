# Contributing

Thanks for taking a look.

## Getting set up

```bash
pip install -e ".[dev]"
pytest -q
```

The suite makes **no network calls** — adapters are exercised against fakes — so
it runs in about a second and needs no API keys.

## What this library is, and is not

Scope discipline is the main design constraint, so it is worth stating before
you open a PR:

- It is an **import-only** library. No proxy server, no plugin system, no
  dynamic provider loading.
- It does **not** own key storage or encryption. Applications with per-user keys
  pass a `key_provider`.
- It contains **no logging**. Please keep it that way — it is what makes "keys
  are never logged" a structural guarantee rather than a promise.
- Base URLs come from the static registry and are not caller-injectable.

## Adding a provider

See *Adding a provider* in the README. Most OpenAI-compatible providers need a
registry entry and nothing else; a provider with its own structured-output
mechanism needs an adapter.

## Pull requests

- Add tests. New provider behaviour should be covered against a fake rather than
  a live endpoint.
- Keep commit subjects in the imperative, prefixed with an area:
  `gemini: request an output ceiling`.
- Note any user-visible change in `CHANGELOG.md` under `[Unreleased]`.
