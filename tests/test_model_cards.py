"""Model card data and per-model capability tests — offline."""
from __future__ import annotations

import json

import pytest

from inferenceswitch import (
    Capability,
    Client,
    ModelCard,
    ModelCardError,
    ModelCards,
    Reliability,
    Route,
    Rung,
    Tool,
    UnsupportedCapabilityError,
    default_cards,
    default_registry,
    user,
)


def test_shipped_cards_load():
    cards = default_cards()
    assert len(cards) >= 6
    assert cards.updated
    assert cards.get("claude-haiku-4-5").rung is Rung.MID


def test_every_route_names_a_registered_provider():
    registry = default_registry()
    for card in default_cards():
        for route in card.routes:
            assert route.provider in registry, (card.id, route.provider)


def test_every_curated_model_has_a_card():
    # Keeps the cards in step with the registry's catalog: a model added to
    # known_models without a card would be invisible to select_model.
    cards = default_cards()
    for spec in default_registry():
        for model in spec.known_models:
            assert cards.find(spec.name, model) is not None, f"{spec.name}/{model}"


def test_local_routes_have_no_cost_and_hosted_ones_do():
    registry = default_registry()
    for card in default_cards():
        for route in card.routes:
            if registry[route.provider].is_local:
                assert route.cost is None, card.id
            else:
                assert route.cost is not None, card.id


def test_open_weight_cards_carry_family_size_and_quantization():
    card = default_cards().get("qwen3-8b")
    assert card.open_weight.family == "qwen3"
    assert card.open_weight.params_b == 8
    assert card.open_weight.quantization
    # One card, several servers: the same weights under Ollama and LM Studio.
    assert {r.provider for r in card.routes} == {"ollama", "lmstudio"}


def test_round_trip():
    cards = default_cards()
    again = ModelCards.from_json(json.dumps(cards.to_dict()))
    assert [c for c in again] == [c for c in cards]
    assert again.updated == cards.updated


def _card_dict(**overrides):
    data = {
        "id": "x",
        "rung": "small",
        "routes": {"ollama": {"models": ["x:1b"], "cost": None}},
        "structured_output": "best_effort",
        "context_window": 8192,
    }
    data.update(overrides)
    return data


def _file(*cards, fmt=1):
    return {"format": fmt, "updated": "2026-01-01", "cards": list(cards)}


@pytest.mark.parametrize(
    "card",
    [
        _card_dict(context_windw=8192),          # typo'd field
        _card_dict(rung="medium"),               # not a rung
        _card_dict(structured_output="mostly"),  # not a grade
        _card_dict(routes={}),                   # no routes
        _card_dict(routes={"ollama": {"models": "x:1b"}}),  # string, not list
        _card_dict(context_window="8k"),
        _card_dict(features={"add": ["telepathy"]}),
        {"id": "x", "rung": "small"},            # missing fields
    ],
)
def test_malformed_cards_are_rejected(card):
    with pytest.raises(ModelCardError):
        ModelCards.from_dict(_file(card))


def test_unknown_format_is_rejected():
    with pytest.raises(ModelCardError, match="format"):
        ModelCards.from_dict(_file(_card_dict(), fmt=2))


def test_duplicate_ids_are_rejected():
    with pytest.raises(ModelCardError, match="Duplicate"):
        ModelCards.from_dict(_file(_card_dict(), _card_dict()))


def test_merged_adds_and_overrides_without_touching_defaults():
    defaults = default_cards()
    mine = ModelCards.from_dict(
        _file(
            _card_dict(id="claude-haiku-4-5", rung="small",
                       routes={"anthropic": {"models": ["claude-haiku-4-5"]}},
                       structured_output="reliable"),
            _card_dict(id="my-model"),
        )
    )
    merged = defaults.merged(mine)
    assert merged.get("claude-haiku-4-5").rung is Rung.SMALL
    assert "my-model" in merged
    assert defaults.get("claude-haiku-4-5").rung is Rung.MID
    assert "my-model" not in defaults
    assert len(merged) == len(defaults) + 1
    assert "my-model" not in merged.without("my-model")


def test_load_from_a_file(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(_file(_card_dict())), encoding="utf-8")
    cards = ModelCards.load(path)
    assert cards.get("x").structured_output is Reliability.BEST_EFFORT


# ── cards override provider capabilities per model ───────────────────────────


def test_card_features_override_the_provider_default():
    client = Client()
    # Provider default stays fail-closed...
    assert not client.supports(Capability.SAMPLING_PARAMS, "anthropic")
    # ...while the cards add it back where the model accepts it.
    assert client.supports(Capability.SAMPLING_PARAMS, model="claude-haiku-4-5")
    assert not client.supports(Capability.SAMPLING_PARAMS, model="claude-opus-4-8")
    # A model with no card gets the provider default.
    assert not client.supports(Capability.SAMPLING_PARAMS, model="claude-unknown-9")


def test_card_lookup_routes_like_resolve():
    client = Client()
    assert client.card("claude-haiku-4-5").id == "claude-haiku-4-5"
    assert client.card("ollama/qwen3:8b").id == "qwen3-8b"
    assert client.card("qwen3:8b", provider="lmstudio") is None  # not its LM Studio id
    assert client.card("claude-unknown-9") is None


def _no_tools_card() -> ModelCard:
    return ModelCard(
        id="tiny",
        rung=Rung.NANO,
        routes=(Route("ollama", ("tiny:1b",)),),
        structured_output=Reliability.NONE,
        features_remove=frozenset({Capability.TOOL_CALLING}),
    )


def test_removed_feature_fails_closed_in_chat():
    client = Client(cards=default_cards().merged([_no_tools_card()]))
    assert client.supports(Capability.TOOL_CALLING, "ollama")
    assert not client.supports(Capability.TOOL_CALLING, model="ollama/tiny:1b")
    tool = Tool("lookup", "Look something up.", {"type": "object", "properties": {}})
    # Raised before any adapter or network is touched.
    with pytest.raises(UnsupportedCapabilityError, match="tiny:1b"):
        client.chat(model="ollama/tiny:1b", messages=[user("hi")], tools=[tool])


# ── grades and per-model overrides on the wider card set ─────────────────────

_ENFORCED = {"output_config_json_schema", "strict_tool_use", "response_schema"}


def test_reliable_grade_only_where_inferenceswitch_enforces_the_schema():
    # The grade describes the path inferenceswitch uses, so a card can only be
    # "reliable" on a provider whose mechanism (as sent) is enforced.
    registry = default_registry()
    for card in default_cards():
        if card.structured_output is not Reliability.RELIABLE:
            continue
        for route in card.routes:
            caps = registry[route.provider].capabilities
            mode = caps.structured_output_for(route.models[0]).value
            strict_json_schema = (
                mode == "response_format_json_schema" and caps.config.get("openai_strict")
            )
            assert mode in _ENFORCED or strict_json_schema, (card.id, route.provider)


def test_openai_cards_encode_chat_completions_limits():
    client = Client()
    # Tool calling: Responses-only on 6.1 Sol; only at effort none on 6 Sol/Luna.
    for model in ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna"):
        assert not client.supports(Capability.TOOL_CALLING, model=f"openai/{model}")
    assert client.supports(Capability.TOOL_CALLING, model="openai/gpt-5.6-terra")
    # Non-reasoning models: temperature yes, reasoning_effort no.
    assert client.supports(Capability.SAMPLING_PARAMS, model="openai/gpt-4.1")
    assert not client.supports(Capability.REASONING_EFFORT, model="openai/gpt-4.1")
    assert client.supports(Capability.REASONING_EFFORT, model="openai/gpt-6-luna")


def test_effort_on_a_non_reasoning_openai_model_fails_closed():
    from inferenceswitch import Effort

    with pytest.raises(UnsupportedCapabilityError, match="reasoning_effort"):
        Client().chat(model="openai/gpt-4o-mini", messages=[user("hi")], effort=Effort.LOW)


def test_one_open_weight_card_spans_hosted_and_local_routes():
    card = default_cards().get("gpt-oss-120b")
    assert {r.provider for r in card.routes} == {"groq", "ollama", "lmstudio"}
    assert card.route("groq").cost is not None
    assert card.route("ollama").cost is None
    assert Client().card("groq/openai/gpt-oss-120b") is card


def test_registry_ids_that_differ_from_the_vendors_resolve_to_the_same_card():
    cards = default_cards()
    # Gemini 3 Flash only exists as a preview id; the registry and card use it.
    assert cards.find("gemini", "gemini-3-flash-preview").id == "gemini-3-flash"
    # Google serves gemini-3.5-flash with 3.6 Flash; the card says so.
    assert "3.6" in cards.get("gemini-3.5-flash").notes
