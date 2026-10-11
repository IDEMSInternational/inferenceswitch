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
