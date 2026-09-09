"""Prompt caching — one example per *capability*, each model-agnostic.

Each example takes a model and runs the same code for any model whose provider
supports that capability. Pick the model; the library picks the mechanism.

    python -m examples.prompt_caching reusable  claude-opus-4-8
    python -m examples.prompt_caching reusable  gemini-2.5-flash
    python -m examples.prompt_caching explicit  claude-opus-4-8
    python -m examples.prompt_caching implicit  openai/gpt-4o

Needs the matching provider key + extra for whichever model you choose
(ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY).

The shape that makes caching worthwhile is always the same: a large STATIC prefix
(system prompt, reference document, few-shot block) reused across many small,
varying queries.
"""
from __future__ import annotations

import sys

from llmswitchboard import Capability, LLMClient, Message, Text, user

# A big static prefix — the thing we want to pay for once, not once per query.
# (Cache minimums: Anthropic ~1024 tokens; Gemini ~32k. This is sized to clear
# Gemini's bar; shrink it and Gemini's create_cache will reject it.)
REFERENCE_DOC = "ACME employee handbook.\n" + ("Policy clause text. " * 12000)
QUESTIONS = [
    "How many vacation days do I accrue per year?",
    "What is the parental leave policy?",
    "Can I carry unused leave into next year?",
]


def _show(question: str, resp) -> None:
    print(f"Q: {question}\n  {resp.text[:80]}\n  cache_read={resp.usage.cache_read_tokens}\n")


def reusable(model: str) -> None:
    """REUSABLE_PROMPT_CACHE — create once, reference many.

    One code path for any supported model: the library creates a Gemini
    ``CachedContent`` resource or an Anthropic replay handle as appropriate. The
    cached prefix (system + messages) is not resent — only the varying tail is.
    """
    client = LLMClient()
    client.require(Capability.REUSABLE_PROMPT_CACHE, model=model)  # clear error if unsupported

    handle = client.create_cache(
        model=model,
        system=REFERENCE_DOC,
        messages=[Message("user", [Text("Answer only from the handbook above.")])],
        ttl=600,  # seconds (exact on Gemini; ~5 min ephemeral on Anthropic today)
    )
    try:
        for q in QUESTIONS:
            _show(q, client.chat(model=model, cache=handle, messages=[user(q)]))
    finally:
        client.delete_cache(handle)  # frees the Gemini resource; no-op for a replay handle


def explicit(model: str) -> None:
    """EXPLICIT_PROMPT_CACHING — inline breakpoint, resent each call.

    Mark the static block ``cache=True`` and resend it; the provider hashes the
    prefix and bills cache-read price on a hit. Stateless — nothing to create or
    release. Supported where the provider takes inline breakpoints (Anthropic).
    """
    client = LLMClient()
    client.require(Capability.EXPLICIT_PROMPT_CACHING, model=model)

    for q in QUESTIONS:
        _show(
            q,
            client.chat(
                model=model,
                messages=[Message("user", [Text(REFERENCE_DOC, cache=True), Text(q)])],
            ),
        )


def implicit(model: str) -> None:
    """IMPLICIT_PROMPT_CACHING — automatic; nothing to place.

    The provider caches long prefixes server-side on its own. Just resend the
    prefix and observe the savings. Supported where the provider auto-caches
    (OpenAI, DeepSeek, Gemini).
    """
    client = LLMClient()
    client.require(Capability.IMPLICIT_PROMPT_CACHING, model=model)

    for q in QUESTIONS:
        _show(
            q,
            client.chat(
                model=model,
                messages=[Message("user", [Text(REFERENCE_DOC), Text(q)])],
            ),
        )


_EXAMPLES = {"reusable": reusable, "explicit": explicit, "implicit": implicit}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in _EXAMPLES:
        sys.exit(f"usage: python -m examples.prompt_caching {{{'|'.join(_EXAMPLES)}}} <model>")
    _EXAMPLES[sys.argv[1]](sys.argv[2])
