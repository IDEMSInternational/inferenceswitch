"""The Gemma 4 e2b card (#24) and reasoning_effort on LM Studio (#26) — offline.

The SDK client is a fake that records each request. The facts the cards and the
registry encode were seen against a live LM Studio on 2026-11-09: it takes
``reasoning_effort`` (400 on a value outside none/minimal/low/medium/high/xhigh)
and ``none`` switches Gemma's thinking off.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from inferenceswitch import (
    Capability,
    Client,
    Effort,
    NoModelAvailableError,
    Reliability,
    Rung,
    UnsupportedCapabilityError,
    default_cards,
    static_availability,
    user,
)
from inferenceswitch.adapters.openai_compat import OpenAICompatibleAdapter
from inferenceswitch.registry import default_registry

GEMMA = "lmstudio/google/gemma-4-e2b"


class _RecordingOpenAI:
    def __init__(self):
        self.requests: list[dict] = []
        response = NS(
            choices=[NS(message=NS(content="ok", tool_calls=None), finish_reason="stop")],
            usage=NS(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None),
        )

        def create(**kwargs):
            self.requests.append(kwargs)
            return response

        self.chat = NS(completions=NS(create=create))


def _client(provider: str) -> tuple[Client, _RecordingOpenAI]:
    client = Client(key_provider=lambda spec: "k")
    fake = _RecordingOpenAI()
    spec = default_registry().get(provider)
    client._adapters[(provider, "k")] = OpenAICompatibleAdapter(spec, "k", raw_client=fake)
    return client, fake


# ── the card ────────────────────────────────────────────────────────────────


def test_the_gemma_card_is_a_nano_lmstudio_model_with_what_was_seen():
    card = default_cards().get("gemma-4-e2b")
    assert card.rung is Rung.NANO
    assert [r.provider for r in card.routes] == ["lmstudio"]
    assert card.route("lmstudio").models == ("google/gemma-4-e2b",)
    assert card.route("lmstudio").cost is None
    # LM Studio's /api/v0/models reports max_context_length 131072.
    assert card.context_window == 131072
    # json_object, which is what inferenceswitch sends LM Studio today, is a 400 (#25).
    assert card.structured_output is Reliability.NONE
    assert card.open_weight.quantization == "Q4_K_M"


def test_the_client_finds_the_card_and_its_capabilities():
    client = Client()
    assert client.card(GEMMA).id == "gemma-4-e2b"
    for capability in (
        Capability.TOOL_CALLING,
        Capability.STREAMING,
        Capability.REASONING_EFFORT,
        Capability.STREAMED_THINKING,
        Capability.VISION,
    ):
        assert client.supports(capability, model=GEMMA), capability
    # Nothing is claimed beyond that.
    assert not client.supports(Capability.STRICT_SCHEMA_OUTPUT, model=GEMMA)
    assert not client.supports(Capability.PARALLEL_TOOL_USE, model=GEMMA)


def test_select_model_can_choose_it_only_at_nano():
    client = Client(
        availability=static_availability({"lmstudio": ["google/gemma-4-e2b"]})
    )
    sel = client.select_model("nano", needs=["tool_calling", "reasoning_effort"])
    assert sel.model == "lmstudio/google/gemma-4-e2b"
    # Small or above, which the STACK Planner asks for, never picks a nano model.
    with pytest.raises(NoModelAvailableError):
        client.select_model("small", needs=["tool_calling", "reasoning_effort"])


# ── reasoning_effort on LM Studio is per card, not per provider ──────────────


def test_the_lmstudio_provider_does_not_declare_reasoning_effort():
    client = Client()
    assert not client.supports(Capability.REASONING_EFFORT, "lmstudio")
    assert not client.supports(Capability.REASONING_EFFORT, "ollama")


@pytest.mark.parametrize(
    "effort,sent",
    [
        (Effort.NONE, "none"),  # Gemma stops thinking only on "none", not on "minimal"
        (Effort.LOW, "low"),
        (Effort.MEDIUM, "medium"),
        (Effort.HIGH, "high"),
        (Effort.MAX, "high"),
    ],
)
def test_effort_reaches_lmstudio_for_the_carded_model(effort, sent):
    client, fake = _client("lmstudio")
    client.chat(model=GEMMA, messages=[user("?")], effort=effort)
    assert fake.requests[0]["reasoning_effort"] == sent


def test_no_effort_sends_no_reasoning_field():
    client, fake = _client("lmstudio")
    client.chat(model=GEMMA, messages=[user("?")])
    assert "reasoning_effort" not in fake.requests[0]


def test_an_lmstudio_model_without_a_card_still_fails_closed():
    client, fake = _client("lmstudio")
    with pytest.raises(UnsupportedCapabilityError, match="reasoning_effort"):
        client.chat(model="lmstudio/some-other-model", messages=[user("?")], effort=Effort.LOW)
    assert fake.requests == []


def test_ollama_is_unchanged():
    client, fake = _client("ollama")
    with pytest.raises(UnsupportedCapabilityError, match="reasoning_effort"):
        client.chat(model="ollama/qwen3:8b", messages=[user("?")], effort=Effort.LOW)
    client.chat(model="ollama/qwen3:8b", messages=[user("?")])
    assert "reasoning_effort" not in fake.requests[0]
