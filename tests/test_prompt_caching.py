"""Prompt caching — explicit (inline breakpoints) vs implicit (automatic).

All offline: encode functions are pure and the capability gate fires before any
adapter/SDK is built, so no SDK or network is touched.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from llmswitchboard import (
    CacheHandle,
    Capability,
    LLMClient,
    LLMResponse,
    LLMSwitchboardError,
    Message,
    Text,
    Tool,
    UnsupportedCapabilityError,
    user,
)
from llmswitchboard.adapters.anthropic import encode_anthropic_messages, encode_anthropic_tools


@pytest.fixture
def client():
    return LLMClient()


# ── the two capabilities have distinct, deliberate provider matrices ───────────


def test_explicit_is_anthropic_only(client):
    # Inline caller-placed breakpoints (cache_control) — Anthropic's mechanism.
    assert client.supports(Capability.EXPLICIT_PROMPT_CACHING, "anthropic")
    # Gemini's caching is out-of-band CachedContent, NOT inline, so it must not
    # advertise the inline capability (else the dropdown would promise a call it
    # can't serve inline).
    assert not client.supports(Capability.EXPLICIT_PROMPT_CACHING, "gemini")
    for p in ("openai", "deepseek", "groq", "mistral", "ollama", "lmstudio"):
        assert not client.supports(Capability.EXPLICIT_PROMPT_CACHING, p)


def test_implicit_is_the_automatic_cachers(client):
    for p in ("openai", "deepseek", "gemini"):
        assert client.supports(Capability.IMPLICIT_PROMPT_CACHING, p)
    # Anthropic never caches without an explicit breakpoint -> not implicit.
    assert not client.supports(Capability.IMPLICIT_PROMPT_CACHING, "anthropic")
    # Groq/Mistral don't do automatic prefix caching.
    assert not client.supports(Capability.IMPLICIT_PROMPT_CACHING, "groq")
    assert not client.supports(Capability.IMPLICIT_PROMPT_CACHING, "mistral")


def test_dropdown_for_inline_caching_lists_only_anthropic(client):
    provs = client.providers_for(Capability.EXPLICIT_PROMPT_CACHING)
    assert provs == ["anthropic"]
    models = client.models_for(Capability.EXPLICIT_PROMPT_CACHING)
    assert models and all(m.startswith("anthropic/") for m in models)


# ── the marker translates to cache_control on Anthropic ────────────────────────


def test_text_marker_emits_cache_control():
    out = encode_anthropic_messages([Message("user", [Text("big context", cache=True)])])
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_unmarked_text_has_no_cache_control():
    out = encode_anthropic_messages([user("hello")])
    assert "cache_control" not in out[0]["content"][0]


def test_tool_marker_emits_cache_control():
    out = encode_anthropic_tools(
        [
            Tool("a", "d", {"type": "object"}),
            Tool("b", "d", {"type": "object"}, cache=True),
        ]
    )
    assert "cache_control" not in out[0]
    assert out[1]["cache_control"] == {"type": "ephemeral"}


# ── derive -> fail-closed guard (before any SDK is built) ──────────────────────


def test_cache_marker_on_non_inline_provider_raises(client):
    # A cache=True marker derives an EXPLICIT_PROMPT_CACHING requirement; routing
    # to a provider without it must fail loudly, not drop the marker silently.
    with pytest.raises(UnsupportedCapabilityError) as exc:
        client.chat(
            model="x",
            provider="gemini",
            messages=[Message("user", [Text("ctx", cache=True)])],
        )
    assert "explicit_prompt_caching" in str(exc.value)


def test_tool_cache_marker_also_gates(client):
    with pytest.raises(UnsupportedCapabilityError):
        client.chat(
            model="x",
            provider="openai",
            messages=[user("hi")],
            tools=[Tool("t", "d", {"type": "object"}, cache=True)],
        )


# ── REUSABLE_PROMPT_CACHE: out-of-band handle (Gemini CachedContent) ───────────


def test_reusable_cache_capability_spans_both_backings(client):
    # Unified surface: Anthropic (replay-backed) and Gemini (server-backed) both
    # advertise it; the OpenAI-family/local providers don't.
    assert client.supports(Capability.REUSABLE_PROMPT_CACHE, "anthropic")
    assert client.supports(Capability.REUSABLE_PROMPT_CACHE, "gemini")
    for p in ("openai", "deepseek", "groq", "mistral", "ollama"):
        assert not client.supports(Capability.REUSABLE_PROMPT_CACHE, p)
    assert set(client.providers_for(Capability.REUSABLE_PROMPT_CACHE)) == {"anthropic", "gemini"}


def test_create_cache_on_unsupported_provider_raises(client):
    # Gated before any adapter/SDK is built. OpenAI has no reusable cache handle.
    with pytest.raises(UnsupportedCapabilityError):
        client.create_cache(model="openai/gpt-4o", messages=[user("x")])


def test_chat_with_cache_rejects_system_or_tools(client):
    # The cached prefix owns system/tools; passing them again is a uniform error.
    handle = CacheHandle(provider="gemini", name="cachedContents/x", model="m")
    with pytest.raises(LLMSwitchboardError):
        client.chat(model="x", provider="gemini", messages=[user("q")],
                    cache=handle, system="sys")
    with pytest.raises(LLMSwitchboardError):
        client.chat(model="x", provider="gemini", messages=[user("q")],
                    cache=handle, tools=[Tool("t", "d", {"type": "object"})])


def test_chat_rejects_a_foreign_cache_handle(client):
    # Handle was minted for another provider than the request routes to.
    handle = CacheHandle(provider="openai", name="cachedContents/x", model="m")
    with pytest.raises(LLMSwitchboardError):
        client.chat(model="x", provider="gemini", messages=[user("tail")], cache=handle)


class _FakeGemini:
    """A stand-in google.genai Client: records the config objects it's handed."""

    def __init__(self):
        self.created: dict = {}
        self.generated: dict = {}
        self.deleted: str | None = None
        self.caches = NS(create=self._cache_create, delete=self._cache_delete)
        self.models = NS(generate_content=self._generate_content)

    def _cache_create(self, *, model, config):
        self.created = {"model": model, "config": config}
        return NS(name="cachedContents/abc", model=model)

    def _cache_delete(self, *, name):
        self.deleted = name
        return None

    def _generate_content(self, *, model, contents, config):
        self.generated = {"model": model, "contents": contents, "config": config}
        return NS(
            candidates=[
                NS(
                    content=NS(parts=[NS(text="hi", function_call=None)]),
                    finish_reason=NS(name="STOP"),
                )
            ],
            usage_metadata=NS(
                prompt_token_count=1,
                candidates_token_count=1,
                cached_content_token_count=5,  # the cache hit
                thoughts_token_count=0,
            ),
        )


def test_gemini_create_cache_then_reference_it():
    # Real google.genai types, fake transport — exercises the config plumbing
    # end to end without a network call.
    pytest.importorskip("google.genai")
    from llmswitchboard.adapters.gemini import GeminiAdapter
    from llmswitchboard.registry import default_registry

    spec = default_registry().get("gemini")
    fake = _FakeGemini()
    adapter = GeminiAdapter(spec, "", raw_client=fake)

    handle = adapter.create_cache(
        model="gemini-2.5-flash", system="BIG SYSTEM", messages=[user("context")], ttl=3600
    )
    assert isinstance(handle, CacheHandle)
    assert handle.provider == "gemini"
    assert handle.name == "cachedContents/abc"
    # The system prompt and TTL went into the cache create config.
    create_cfg = fake.created["config"]
    assert create_cfg.system_instruction == "BIG SYSTEM"
    assert create_cfg.ttl == "3600s"

    # Referencing the handle: cached_content is set and the prefix is NOT resent.
    resp = adapter.chat(model="gemini-2.5-flash", messages=[user("just the tail")], cache=handle)
    gen_cfg = fake.generated["config"]
    assert gen_cfg.cached_content == "cachedContents/abc"
    assert gen_cfg.system_instruction is None  # lives in the cache, not resent
    assert resp.text == "hi"
    assert resp.usage.cache_read_tokens == 5

    # delete_cache frees the server resource by name.
    adapter.delete_cache(handle)
    assert fake.deleted == "cachedContents/abc"


def test_delete_cache_is_a_noop_for_replay_handles():
    # A replay-backed (Anthropic) handle names no server resource, so the client
    # short-circuits — no adapter is built and no key is needed.
    client = LLMClient()
    handle = CacheHandle(provider="anthropic", model="claude-opus-4-8", system="SYS")
    assert handle.name is None
    client.delete_cache(handle)  # must not raise (no ANTHROPIC_API_KEY required)


class _FakeAnthropic:
    """A stand-in anthropic.Anthropic: records every request payload.

    The adapter streams (see `AnthropicAdapter._send`), so the double exposes
    `messages.stream` as a context manager whose `get_final_message()` yields the
    same object a non-streaming `create` would have returned.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.messages = NS(stream=self._stream)

    def _stream(self, **kwargs):
        message = self._create(**kwargs)

        class _Stream:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def get_final_message(self_inner):
                return message

        return _Stream()

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return NS(
            content=[NS(type="text", text="answer")],
            stop_reason="end_turn",
            usage=NS(
                input_tokens=3,
                output_tokens=2,
                cache_read_input_tokens=100,  # cache hit on the replayed prefix
                cache_creation_input_tokens=0,
            ),
        )


def test_anthropic_create_cache_is_pure_and_replays_with_breakpoint():
    from llmswitchboard.adapters.anthropic import AnthropicAdapter
    from llmswitchboard.registry import default_registry

    spec = default_registry().get("anthropic")
    fake = _FakeAnthropic()
    adapter = AnthropicAdapter(spec, "", raw_client=fake)

    handle = adapter.create_cache(
        model="claude-opus-4-8", system="BIG SYSTEM", messages=[user("reference doc")]
    )
    assert fake.calls == []            # create is pure — no network
    assert handle.name is None         # replay-backed, not server-backed
    assert handle.provider == "anthropic"

    resp = adapter.chat(model="claude-opus-4-8", messages=[user("q1")], cache=handle)
    kw = fake.calls[0]
    # Prefix message replayed with the breakpoint on its (deepest) last block...
    assert kw["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # ...system carried along (cached implicitly, above the breakpoint)...
    assert kw["system"][0]["text"] == "BIG SYSTEM"
    # ...and the dynamic tail appended after the cached prefix.
    assert kw["messages"][-1]["content"][0]["text"] == "q1"
    assert resp.text == "answer"
    assert resp.usage.cache_read_tokens == 100


def test_same_caller_code_caches_on_both_providers(monkeypatch):
    """The whole point: create_cache + chat(cache=) is written once and works on
    an Anthropic model or a Gemini model, no provider branching in the caller."""
    client = LLMClient()

    class _FakeAdapter:
        def __init__(self, name):
            self.spec = NS(name=name)

        def create_cache(self, *, model, messages, system=None, tools=None, ttl=None):
            # server-backed for gemini, replay-backed for anthropic — caller doesn't care
            return CacheHandle(
                provider=self.spec.name,
                model=model,
                name="cachedContents/x" if self.spec.name == "gemini" else None,
                system=system,
                messages=tuple(messages),
            )

        def chat(self, *, model, messages, cache=None, **kw):
            assert cache is not None and cache.provider == self.spec.name
            return LLMResponse(text=f"{self.spec.name}:ok")

    monkeypatch.setattr(client, "_adapter", lambda spec, api_key: _FakeAdapter(spec.name))

    def cached_tool(model: str) -> str:  # identical body for either provider
        handle = client.create_cache(
            model=model, system="SYS", messages=[Message("user", [Text("DOC")])]
        )
        return client.chat(model=model, cache=handle, messages=[user("q")]).text

    assert cached_tool("claude-opus-4-8") == "anthropic:ok"
    assert cached_tool("gemini-2.5-flash") == "gemini:ok"
