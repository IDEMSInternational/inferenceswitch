"""Anthropic output caps: default to the model's own maximum, caller-controllable.

Offline — the adapter is driven with a fake SDK client that records the request
payload, so every assertion is about what inferenceswitch *sends*.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from inferenceswitch import (
    CLAUDE_MAX_OUTPUT_TOKENS,
    StructuredOutputError,
    claude_max_output_tokens,
    user,
)
from inferenceswitch.adapters.anthropic import (
    UNKNOWN_MODEL_MAX_OUTPUT_TOKENS,
    AnthropicAdapter,
    resolve_max_tokens,
)
from inferenceswitch.registry import default_registry


class _FakeAnthropic:
    """A stand-in anthropic.Anthropic that records every request payload.

    The adapter streams (see `AnthropicAdapter._send`), so `messages.stream` is a
    context manager whose `get_final_message()` yields what `create` would return.
    `stop_reason` is settable so truncation can be exercised.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.stop_reason = "end_turn"
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="answer")],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )

        class _Stream:
            def __enter__(inner):
                return inner

            def __exit__(inner, *exc):
                return False

            def get_final_message(inner):
                return message

        return _Stream()


@pytest.fixture
def adapter():
    fake = _FakeAnthropic()
    return AnthropicAdapter(default_registry().get("anthropic"), "", raw_client=fake), fake


# ── the table ────────────────────────────────────────────────────────────────


def test_known_models_resolve_to_their_own_ceiling():
    assert claude_max_output_tokens("claude-opus-4-8") == 128_000
    assert claude_max_output_tokens("claude-sonnet-4-6") == 128_000
    # Haiku 4.5 is the one current model that caps lower — it must not inherit 128k.
    assert claude_max_output_tokens("claude-haiku-4-5") == 64_000


def test_dated_and_prefixed_ids_match_their_base_model():
    assert claude_max_output_tokens("claude-haiku-4-5-20251001") == 64_000
    assert claude_max_output_tokens("anthropic.claude-opus-4-8") == 128_000


def test_unknown_model_falls_back_to_the_smallest_current_ceiling():
    assert claude_max_output_tokens("claude-something-new") == UNKNOWN_MODEL_MAX_OUTPUT_TOKENS


def test_explicit_max_tokens_wins_and_is_not_clamped():
    # Above the model's max: passed through, so Anthropic's own 400 is what the
    # caller sees — inferenceswitch never silently rewrites an explicit number.
    assert resolve_max_tokens("claude-haiku-4-5", 200_000) == 200_000
    assert resolve_max_tokens("claude-haiku-4-5", 512) == 512


# ── what actually goes on the wire ───────────────────────────────────────────


def test_chat_defaults_to_the_model_max(adapter):
    a, fake = adapter
    a.chat(model="claude-opus-4-8", messages=[user("hi")])
    assert fake.calls[-1]["max_tokens"] == 128_000

    a.chat(model="claude-haiku-4-5", messages=[user("hi")])
    assert fake.calls[-1]["max_tokens"] == 64_000


def test_generate_text_and_cached_chat_use_the_same_default(adapter):
    a, fake = adapter
    a.generate_text(model="claude-opus-4-8", prompt="hi")
    assert fake.calls[-1]["max_tokens"] == 128_000

    handle = a.create_cache(model="claude-opus-4-8", messages=[user("doc")])
    a.chat(model="claude-opus-4-8", messages=[user("q")], cache=handle)
    assert fake.calls[-1]["max_tokens"] == 128_000


def test_caller_max_tokens_overrides_the_default(adapter):
    a, fake = adapter
    a.chat(model="claude-opus-4-8", messages=[user("hi")], max_tokens=1000)
    assert fake.calls[-1]["max_tokens"] == 1000


def test_table_edit_lowers_the_default_process_wide(adapter, monkeypatch):
    """The table is the control surface: an app can impose a house ceiling."""
    a, fake = adapter
    monkeypatch.setitem(CLAUDE_MAX_OUTPUT_TOKENS, "claude-opus-4-8", 8_000)
    a.chat(model="claude-opus-4-8", messages=[user("hi")])
    assert fake.calls[-1]["max_tokens"] == 8_000


# ── truncation reporting ─────────────────────────────────────────────────────


def test_truncated_structured_json_reports_the_model_max(adapter):
    a, fake = adapter
    fake.stop_reason = "max_tokens"
    with pytest.raises(StructuredOutputError) as excinfo:
        a.generate_structured_json(
            model="claude-opus-4-8", prompt="p", schema={"type": "object"}, max_tokens=1000
        )
    err = excinfo.value
    assert err.context["max_tokens"] == 1000
    assert err.context["model_max_tokens"] == 128_000
    assert "raise max_tokens" in str(err)


def test_truncation_at_the_model_max_advises_splitting(adapter):
    a, fake = adapter
    fake.stop_reason = "max_tokens"
    with pytest.raises(StructuredOutputError) as excinfo:
        a.generate_structured_json(
            model="claude-opus-4-8", prompt="p", schema={"type": "object"}
        )
    assert "split the request" in str(excinfo.value)
