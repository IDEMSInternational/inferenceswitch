# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's
[Report a vulnerability](https://github.com/IDEMSInternational/inferenceswitch/security/advisories/new)
form rather than opening a public issue.

We aim to acknowledge a report within five working days.

## What this library does with your API keys

`inferenceswitch` handles provider API keys, so it is worth being explicit about
the boundaries:

- **Keys are never logged.** The package contains no logging calls of any kind —
  no `print`, no `logging` handler. This is structural rather than a policy that
  could be forgotten in a later change.
- **Keys are never written to disk** and are never placed in an exception
  message. `MissingAPIKeyError` names the environment *variable*, never a value.
- **Keys are read from the environment by default**, or supplied by a
  caller-provided `key_provider` / per-call `api_key`.
- **This library does not store or encrypt keys.** An application with per-user
  keys is expected to own that, and to pass a `key_provider` that decrypts on
  its side.
- **No network destinations are caller-injectable.** Base URLs come from the
  static registry, so a malicious model string cannot redirect a request to an
  attacker-controlled host.

## Scope

This is an import-only library. It runs no server and opens no listening socket,
so deployment- and proxy-level concerns do not apply. Vulnerabilities in the
underlying provider SDKs should be reported to those projects.
