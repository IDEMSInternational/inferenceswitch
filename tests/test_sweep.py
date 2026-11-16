"""The sweep script's pure parts and the card data it keeps up to date — offline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from inferenceswitch import Client, ModelCards, default_cards, default_registry

SWEEP_PATH = Path(__file__).resolve().parent.parent / "tools" / "sweep.py"
_spec = importlib.util.spec_from_file_location("sweep", SWEEP_PATH)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

CARDS_PATH = Path(sweep.CARDS_PATH)


def test_the_writer_reproduces_the_shipped_file_exactly():
    # The sweep rewrites model_cards.json; if its layout drifted from the
    # hand-kept one, every sweep would be a whole-file diff.
    text = CARDS_PATH.read_text(encoding="utf-8")
    assert sweep.dump_cards(json.loads(text)) == text


def _page(*records: dict, version: str = "5.0.1") -> str:
    """An Artificial Analysis page: the records inside a streamed Next.js chunk."""
    body = ",".join(json.dumps(r, separators=(",", ":")) for r in records)
    chunk = '2a:[{"models":[' + body + "]}]\n"
    push = "self.__next_f.push([1," + json.dumps(chunk) + "])"
    return f"<html><body>Intelligence Index v{version}<script>{push}</script></body></html>"


def _record(**over) -> dict:
    rec = {
        "id": "1", "slug": "m-1", "name": "M 1 (Max)", "isReasoning": True,
        "effort": {"slug": "max", "label": "max", "level": 60},
        "intelligenceIndex": 31.64, "gpqa": 0.9123, "hle": 0.4, "lcr": None,
        "scicode": 0.5162,
    }
    rec.update(over)
    return rec


def test_a_page_is_read_for_its_scores_and_index_version():
    aa = sweep.read_aa_page(_page(_record()), "m-1")
    assert aa["version"] == "5.0.1"
    bench = sweep.benchmarks_from(aa, "2026-11-16")
    assert bench["aa_intelligence_index"] == {
        "score": 31.6, "source": "artificialanalysis.ai/models/m-1",
        "date": "2026-11-16", "version": "5.0.1", "setting": "max",
    }
    # Fractions become percentages; a null score is left out, not zero.
    assert bench["gpqa_diamond"]["score"] == 91.2
    assert "aa_lcr" not in bench


def test_a_zero_on_the_long_context_test_is_not_recorded_as_a_score():
    # Models with a 32K window score 0 on it because the prompts do not fit.
    aa = sweep.read_aa_page(_page(_record(lcr=0.0)), "m-1")
    assert "aa_lcr" not in sweep.benchmarks_from(aa, "2026-11-16")


def test_the_highest_effort_record_for_a_slug_is_the_one_read():
    low = _record(intelligenceIndex=10.0, effort={"slug": "low", "label": "low", "level": 10})
    page = _page(low, _record())
    assert sweep.read_aa_page(page, "m-1")["record"]["intelligenceIndex"] == 31.64


def test_a_page_of_another_shape_is_unreadable_not_an_error():
    assert sweep.read_aa_page("<html>nothing here</html>", "m-1") is None
    assert sweep.read_aa_page(_page(_record(slug="other")), "m-1") is None


@pytest.mark.parametrize("score,rung", [
    (3.8, None),      # gpt-oss-120b: below the index's resolution
    (11.9, None), (12.0, "small"),
    (20.1, "small"),  # Qwen3.8 27B
    (26.7, "mid"),    # MiMo-V2.6-Flash
    (31.7, "mid"),    # GLM-5.3
    (46.5, "large"),  # Kimi K3.1 Pro
    (59.1, "frontier"),
])
def test_rung_suggestions_follow_the_published_bands(score, rung):
    assert sweep.suggest_rung(score) == rung


def test_normalised_names_match_across_catalogs():
    assert sweep.norm("z-ai/glm-5.3-flash:free") == sweep.norm("glm-5-3-flash")
    assert sweep.norm("qwen/qwen3.8-27b") == sweep.norm("qwen3-8-27b")


def test_a_card_names_the_page_its_scores_came_from():
    card = default_cards().get("glm-5.3-flash")
    assert sweep.aa_slug(card.to_dict()) == "glm-5-3-flash"
    assert sweep.aa_slug({"id": "x", "benchmarks": {}}) is None


# ── the data the sweep keeps ────────────────────────────────────────────────


OPEN_MODELS_FROM_ISSUE_29 = [
    "qwen3-30b-a3b", "glm-5.3", "glm-5.3-flash", "minimax-m3.1-flash",
    "mimo-v2.6-flash", "kimi-k3.1-pro", "deepseek-v4.1-flash",
]


@pytest.mark.parametrize("card_id", OPEN_MODELS_FROM_ISSUE_29)
def test_the_open_models_from_issue_29_have_cards_a_caller_can_pin(card_id):
    cards = default_cards()
    card = cards.get(card_id)
    assert card.context_window and card.open_weight
    route = card.route("openrouter") or card.route("ollama")
    model = route.models[0]
    provider = "openrouter" if card.route("openrouter") else "ollama"
    assert Client().card(f"{provider}/{model}") is card


def test_the_self_hosted_qwen_card_offers_no_graded_effort():
    card = default_cards().get("qwen3-30b-a3b")
    assert {"ollama", "lmstudio", "openrouter"} == {r.provider for r in card.routes}
    assert "reasoning_effort" not in {c.value for c in card.features_add}


def test_openrouter_is_a_keyed_provider_with_no_bare_name_routing():
    spec = default_registry()["openrouter"]
    assert spec.takes_key and not spec.is_local
    assert spec.base_url == "https://openrouter.ai/api/v1"
    assert not spec.model_prefixes and not spec.known_models


def test_every_benchmark_says_where_and_when_it_was_read():
    for card in default_cards():
        for bench in card.benchmarks:
            assert bench.source and bench.date, (card.id, bench.name)
            assert bench.date <= default_cards().updated, (card.id, bench.name)
            # An ECI is an index around 100 to 180; every other score is a percentage.
            assert bench.score >= 0 and (bench.name == "epoch_eci" or bench.score <= 100), (card.id, bench.name)
            if bench.interval:
                assert bench.interval[0] <= bench.score <= bench.interval[1], (card.id, bench.name)


def test_scores_for_one_card_come_from_one_page_per_source():
    for card in default_cards():
        aa = {b.source for b in card.benchmarks if b.source.startswith("artificialanalysis.ai/")}
        epoch = {b.source for b in card.benchmarks if b.source.startswith("epoch.ai/")}
        assert len(aa) <= 1 and len(epoch) <= 1, card.id
        assert len(aa) + len(epoch) == len({b.source for b in card.benchmarks}), card.id


@pytest.mark.parametrize("score,rung", [(125.0, "small"), (151.9, "small"), (152.0, "mid"),
                                        (161.4, "mid"), (163.0, "large"), (168.0, "frontier")])
def test_eci_rung_edges(score, rung):
    assert sweep.suggest_rung_eci(score) == rung


def test_a_card_is_placed_on_the_lower_of_its_two_views():
    g = sweep.grade({
        "epoch_eci": {"score": 152.6, "interval": [150.6, 154.8]},
        "aa_intelligence_index": {"score": 20.1, "version": "5.0.1"},
    })
    assert g["views"] == {"eci": "mid", "aa": "small"}
    assert (g["suggested"], g["agree"], g["borderline"]) == ("small", False, True)
    # The AA index at its floor has no view, so the ECI alone speaks.
    only = sweep.grade({"epoch_eci": {"score": 140.0}, "aa_intelligence_index": {"score": 3.8, "version": "5.0.1"}})
    assert only["views"] == {"eci": "small"} and only["agree"]
    assert sweep.grade({})["suggested"] is None


def test_every_card_sits_on_the_rung_its_scores_put_it_on():
    # Rungs are capability levels, not a vendor's tier names, so a card whose
    # scores say another rung is wrong until it is moved (or re-scored). A card
    # with no score is placed by size or judgement and is not tested.
    off = {}
    for card in default_cards():
        g = sweep.grade(card.to_dict().get("benchmarks"))
        if not sweep.settles(card.rung.value, g["suggested"]):
            off[card.id] = (card.rung.value, g["suggested"])
    assert not off, off


def test_the_rung_edges_are_frozen():
    # They define what a rung means to every tool pinned to one. Moving them
    # needs a decision, not an edit: this fails so that it cannot be quiet.
    assert sweep.ECI_EDGES == (("mid", 152.0), ("large", 163.0), ("frontier", 168.0))
    assert sweep.AA_EDGES == {"5.0.1": (("small", 12.0), ("mid", 26.0), ("large", 40.0), ("frontier", 55.0))}


def test_an_index_from_a_version_with_no_edges_has_no_view():
    assert sweep.suggest_rung(40.0, "6.0.0") is None
    assert sweep.suggest_rung(40.0, None) is None
    g = sweep.grade({"aa_intelligence_index": {"score": 50.0, "version": "6.0.0"}})
    assert g["suggested"] is None


def test_a_score_that_says_small_settles_a_nano_card():
    assert sweep.settles("nano", "small") and not sweep.settles("nano", "mid")
    assert sweep.settles("mid", None)
