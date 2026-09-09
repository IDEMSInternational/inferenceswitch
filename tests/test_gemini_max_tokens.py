"""Gemini output caps: default to the model's own maximum, caller-controllable,
and report a truncation as a truncation.

Offline — the adapter is driven with a fake SDK client that records the request
config, so every assertion is about what llmswitchboard *sends*, or about what it
raises when the fake reports a MAX_TOKENS finish.

The companion file is ``test_max_tokens.py`` (Anthropic). The two adapters are
meant to behave identically here, so the cases are deliberately parallel.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llmswitchboard import (
    GEMINI_MAX_OUTPUT_TOKENS,
    StructuredOutputError,
    gemini_max_output_tokens,
    user,
)
from llmswitchboard.adapters.gemini import (
    UNKNOWN_GEMINI_MODEL_MAX_OUTPUT_TOKENS,
    GeminiAdapter,
    resolve_max_tokens,
)
from llmswitchboard.registry import default_registry

MODEL = "gemini-3.5-flash"


class _FakeFinishReason:
    """Stands in for the SDK's finish-reason enum, whose ``.name`` is the wire
    value. Using an object rather than a bare string keeps the test honest about
    the shape the adapter actually meets in production."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeGenai:
    """A stand-in google.genai Client recording every generate_content config.

    ``text`` and ``finish_reason`` are settable so truncation, refusal and
    genuinely-malformed output can each be exercised without a network call.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.text = "{}"
        self.finish_reason = "STOP"
        self.prompt_token_count = 4096
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        part = SimpleNamespace(text=self.text, function_call=None)
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[part]),
            finish_reason=_FakeFinishReason(self.finish_reason),
        )
        return SimpleNamespace(
            candidates=[candidate],
            text=self.text,
            usage_metadata=SimpleNamespace(
                prompt_token_count=self.prompt_token_count,
                candidates_token_count=65_536,
                cached_content_token_count=0,
                thoughts_token_count=0,
            ),
        )


@pytest.fixture
def adapter():
    fake = _FakeGenai()
    return GeminiAdapter(default_registry().get("gemini"), "", raw_client=fake), fake


def _sent_max_tokens(fake) -> int | None:
    """The max_output_tokens on the last config, whether the config is a real
    ``types.GenerateContentConfig`` or (offline) a plain object."""
    return getattr(fake.calls[-1]["config"], "max_output_tokens", None)


# ── the table ────────────────────────────────────────────────────────────────


def test_known_models_resolve_to_their_own_ceiling():
    assert gemini_max_output_tokens("gemini-3.5-flash") == 65_536
    assert gemini_max_output_tokens("gemini-2.5-pro") == 65_536
    assert gemini_max_output_tokens("gemini-2.5-flash") == 65_536


def test_dated_and_prefixed_ids_match_their_base_model():
    assert gemini_max_output_tokens("gemini-3-flash-preview") == 65_536
    assert gemini_max_output_tokens("gemini-2.5-flash-002") == 65_536
    assert gemini_max_output_tokens("google/gemini-3.5-flash") == 65_536


def test_longest_match_wins_over_a_nested_prefix(monkeypatch):
    """``gemini-3.5-flash`` is a substring of ``gemini-3.5-flash-lite``. The
    ceilings happen to be equal today, so diverge them to prove the resolver
    picks the specific row rather than the first one that matches."""
    monkeypatch.setitem(GEMINI_MAX_OUTPUT_TOKENS, "gemini-3.5-flash-lite", 8_192)
    assert gemini_max_output_tokens("gemini-3.5-flash-lite") == 8_192
    assert gemini_max_output_tokens("gemini-3.5-flash-lite-preview") == 8_192
    assert gemini_max_output_tokens("gemini-3.5-flash") == 65_536


def test_unknown_model_falls_back_to_the_current_line_ceiling():
    assert (
        gemini_max_output_tokens("gemini-9-something-new")
        == UNKNOWN_GEMINI_MODEL_MAX_OUTPUT_TOKENS
    )


def test_explicit_max_tokens_wins_and_is_not_clamped():
    # Above the model's max: passed through, so Gemini's own 400 is what the
    # caller sees — llmswitchboard never silently rewrites an explicit number.
    assert resolve_max_tokens(MODEL, 200_000) == 200_000
    assert resolve_max_tokens(MODEL, 512) == 512


# ── what actually goes on the wire ───────────────────────────────────────────


def test_structured_json_defaults_to_the_model_max(adapter):
    a, fake = adapter
    a.generate_structured_json(model=MODEL, prompt="p", schema={"type": "object"})
    assert _sent_max_tokens(fake) == 65_536


def test_generate_text_and_chat_use_the_same_default(adapter):
    a, fake = adapter
    a.generate_text(model=MODEL, prompt="hi")
    assert _sent_max_tokens(fake) == 65_536

    a.chat(model=MODEL, messages=[user("hi")])
    assert _sent_max_tokens(fake) == 65_536


def test_caller_max_tokens_overrides_the_default_at_every_site(adapter):
    a, fake = adapter
    a.generate_structured_json(
        model=MODEL, prompt="p", schema={"type": "object"}, max_tokens=1000
    )
    assert _sent_max_tokens(fake) == 1000

    a.generate_text(model=MODEL, prompt="hi", max_tokens=1000)
    assert _sent_max_tokens(fake) == 1000

    a.chat(model=MODEL, messages=[user("hi")], max_tokens=1000)
    assert _sent_max_tokens(fake) == 1000


def test_an_explicit_value_above_the_ceiling_is_sent_unclamped(adapter):
    a, fake = adapter
    a.chat(model=MODEL, messages=[user("hi")], max_tokens=200_000)
    assert _sent_max_tokens(fake) == 200_000


def test_table_edit_lowers_the_default_process_wide(adapter, monkeypatch):
    """The table is the control surface: an app can impose a house ceiling."""
    a, fake = adapter
    monkeypatch.setitem(GEMINI_MAX_OUTPUT_TOKENS, MODEL, 8_000)
    a.chat(model=MODEL, messages=[user("hi")])
    assert _sent_max_tokens(fake) == 8_000


# ── truncation reporting ─────────────────────────────────────────────────────


def test_truncated_structured_json_reports_truncation_not_a_parse_error(adapter):
    """The regression this issue is about: a MAX_TOKENS finish used to surface as
    ``Unterminated string ... char 85119``, sending the caller to debug a schema
    that was fine."""
    a, fake = adapter
    fake.finish_reason = "MAX_TOKENS"
    fake.text = '{"items": [{"name": "half a str'  # a valid JSON *prefix*
    with pytest.raises(StructuredOutputError) as excinfo:
        a.generate_structured_json(
            model=MODEL, prompt="p", schema={"type": "object"}, max_tokens=1000
        )
    err = excinfo.value
    message = str(err)
    assert "output cap" in message
    assert "1000" in message                       # the ceiling in effect
    assert "65536" in message                      # the model's own maximum
    assert "raise max_tokens" in message           # room to raise -> say so
    assert "4096 tokens" in message                # points at input size
    assert "Unterminated" not in message
    assert err.context["max_tokens"] == 1000
    assert err.context["model_max_tokens"] == 65_536
    assert err.context["stop_reason"] == "MAX_TOKENS"
    assert err.context["input_tokens"] == 4096


def test_truncation_at_the_model_max_advises_splitting(adapter):
    a, fake = adapter
    fake.finish_reason = "MAX_TOKENS"
    fake.text = "{"
    with pytest.raises(StructuredOutputError) as excinfo:
        a.generate_structured_json(model=MODEL, prompt="p", schema={"type": "object"})
    assert "split the request" in str(excinfo.value)
    assert "raise max_tokens" not in str(excinfo.value)


def test_genuinely_malformed_json_still_reports_a_parse_error(adapter):
    """The truncation check must not swallow the case it was carved out of: a
    STOP finish with unparseable text is still a parse error."""
    a, fake = adapter
    fake.text = "I'm afraid I can't do that."
    with pytest.raises(StructuredOutputError) as excinfo:
        a.generate_structured_json(model=MODEL, prompt="p", schema={"type": "object"})
    err = excinfo.value
    assert "non-JSON content" in str(err)
    assert err.context["stop_reason"] == "STOP"
    assert err.context["max_tokens"] == 65_536


def test_untruncated_structured_json_is_returned_normally(adapter):
    a, fake = adapter
    fake.text = json.dumps({"ok": True})
    assert a.generate_structured_json(
        model=MODEL, prompt="p", schema={"type": "object"}
    ) == {"ok": True}


def test_generate_text_returns_the_partial_answer_rather_than_raising(adapter):
    """Deliberate asymmetry with structured JSON: truncated prose is still a
    usable string, and the only error class available would misdescribe a call
    that requested no structure. `chat` is where the stop reason is reported."""
    a, fake = adapter
    fake.finish_reason = "MAX_TOKENS"
    fake.text = "the answer begins and then"
    assert a.generate_text(model=MODEL, prompt="p") == "the answer begins and then"


def test_chat_still_surfaces_max_tokens_as_a_stop_reason(adapter):
    from llmswitchboard import StopReason

    a, fake = adapter
    fake.finish_reason = "MAX_TOKENS"
    fake.text = "cut off"
    assert a.chat(model=MODEL, messages=[user("hi")]).stop_reason is StopReason.MAX_TOKENS
