"""Reasoning-effort normalization — one `effort` level, three native knobs.

Mappers are pure functions, tested offline; the capability gate is tested
against the default registry.
"""
from __future__ import annotations

import pytest

from inferenceswitch import Capability, Effort, Client, UnsupportedCapabilityError, user
from inferenceswitch.adapters.anthropic import anthropic_reasoning_kwargs
from inferenceswitch.adapters.gemini import gemini_thinking_budget
from inferenceswitch.adapters.openai_compat import openai_reasoning_kwargs


# ── OpenAI: reasoning_effort, only when the provider declares the param ────────


def test_openai_reasoning_maps_to_param():
    cfg = {"reasoning_param": "reasoning_effort"}
    assert openai_reasoning_kwargs(Effort.HIGH, cfg) == {"reasoning_effort": "high"}
    assert openai_reasoning_kwargs(Effort.MAX, cfg) == {"reasoning_effort": "high"}  # clamp
    assert openai_reasoning_kwargs(Effort.NONE, cfg) == {"reasoning_effort": "minimal"}


def test_openai_reasoning_noop_without_param_or_effort():
    # Providers without a reasoning_param (mistral/groq/locals) send nothing...
    assert openai_reasoning_kwargs(Effort.HIGH, {}) == {}
    # ...and no effort means no kwargs regardless.
    assert openai_reasoning_kwargs(None, {"reasoning_param": "reasoning_effort"}) == {}


# ── Anthropic: adaptive thinking + output_config.effort ───────────────────────


def test_anthropic_reasoning_maps_to_thinking_and_effort():
    assert anthropic_reasoning_kwargs(Effort.HIGH) == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    assert anthropic_reasoning_kwargs(Effort.MAX)["output_config"] == {"effort": "max"}


def test_anthropic_none_disables_thinking():
    assert anthropic_reasoning_kwargs(Effort.NONE) == {"thinking": {"type": "disabled"}}
    assert anthropic_reasoning_kwargs(None) == {}


# ── Gemini: thinking-token budget ─────────────────────────────────────────────


def test_gemini_reasoning_maps_to_budget():
    assert gemini_thinking_budget(Effort.LOW) == 1024
    assert gemini_thinking_budget(Effort.MAX) == -1   # dynamic / max
    assert gemini_thinking_budget(Effort.NONE) == 0   # off
    assert gemini_thinking_budget(None) is None       # leave unset


# ── gate: effort on a provider without control -> loud failure ────────────────


def test_effort_requires_capability():
    client = Client()
    # Groq has no REASONING_EFFORT — the gate fires before any adapter is built.
    with pytest.raises(UnsupportedCapabilityError):
        client.chat(model="llama-3.3-70b", provider="groq",
                    messages=[user("hi")], effort=Effort.HIGH)


def test_discovery_lists_effort_capable_models():
    client = Client()
    provs = set(client.providers_for(Capability.REASONING_EFFORT))
    assert {"anthropic", "openai", "gemini"} == provs   # exactly these
    assert "deepseek" not in provs                       # reasoning via model choice, not effort


@pytest.mark.parametrize(
    "model, sent",
    [
        ("gpt-5", "minimal"),
        ("gpt-5-mini", "minimal"),
        ("gpt-5-nano-2025-08-07", "minimal"),
        ("gpt-5.1", "none"),
        ("gpt-5.4-mini", "none"),
        ("gpt-5.6-terra", "none"),
        ("gpt-6-sol", "none"),
        ("gpt-6-luna", "none"),
        ("gpt-6-astra", "low"),
        ("gpt-6.1-sol", "low"),
        ("o3", "low"),
        ("o4-mini", "low"),
        ("some-future-model", "low"),
    ],
)
def test_openai_effort_none_is_the_least_each_model_accepts(model, sent):
    from inferenceswitch import default_registry

    config = default_registry().get("openai").capabilities.config
    assert openai_reasoning_kwargs(Effort.NONE, config, model) == {"reasoning_effort": sent}
    # Other levels are not per model.
    assert openai_reasoning_kwargs(Effort.LOW, config, model) == {"reasoning_effort": "low"}


@pytest.mark.parametrize(
    "model, sent",
    [
        ("gpt-5", "high"),
        ("gpt-5-mini", "high"),
        ("gpt-5.1", "high"),
        ("gpt-5.4", "xhigh"),
        ("gpt-5.4-nano", "xhigh"),
        ("gpt-5.6-luna", "max"),
        ("gpt-6-astra", "max"),
        ("gpt-6.1-sol", "max"),
        ("gpt-6-luna", "max"),
        ("o3", "high"),
        ("some-future-model", "high"),
    ],
)
def test_openai_effort_max_is_the_most_each_model_accepts(model, sent):
    from inferenceswitch import default_registry

    config = default_registry().get("openai").capabilities.config
    assert openai_reasoning_kwargs(Effort.MAX, config, model) == {"reasoning_effort": sent}
    assert openai_reasoning_kwargs(Effort.HIGH, config, model) == {"reasoning_effort": "high"}
