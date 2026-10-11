"""Rung-based selection tests — offline: availability is always faked."""
from __future__ import annotations

import io
import json
import urllib.request

import pytest

from inferenceswitch import (
    Capability,
    Client,
    MissingAPIKeyError,
    ModelCard,
    ModelCards,
    NoModelAvailableError,
    ProviderAvailability,
    Reliability,
    Route,
    Rung,
    default_registry,
    probe_local_server,
)
from inferenceswitch.cards import Cost
from inferenceswitch.selection import key_availability

def _hosted(card_id, rung, provider, price, context):
    return {
        "id": card_id, "rung": rung, "structured_output": "reliable",
        "context_window": context,
        "routes": {provider: {"models": [card_id], "cost": {"input": price[0], "output": price[1]}}},
    }


def _local(card_id, rung, ollama_ids, params, context):
    return {
        "id": card_id, "rung": rung, "structured_output": "best_effort",
        "context_window": context,
        "routes": {"ollama": {"models": ollama_ids}, "lmstudio": {"models": [card_id]}},
        "open_weight": {"family": card_id.split("-")[0], "params_b": params},
    }


# A fixed card set, so ranking tests don't move when the shipped data does.
CARDS = ModelCards.from_dict({
    "format": 1,
    "updated": "2026-01-01",
    "cards": [
        _hosted("claude-haiku-4-5", "mid", "anthropic", (1.0, 5.0), 200_000),
        _hosted("claude-sonnet-4-6", "large", "anthropic", (3.0, 15.0), 1_000_000),
        _hosted("claude-opus-4-8", "frontier", "anthropic", (5.0, 25.0), 1_000_000),
        _hosted("gemini-2.5-flash", "mid", "gemini", (0.30, 2.50), 1_048_576),
        _hosted("gemini-3-flash", "mid", "gemini", (0.50, 3.00), 1_048_576),
        _local("qwen3-4b", "nano", ["qwen3:4b"], 4, 32_768),
        _local("qwen3-8b", "small", ["qwen3:8b", "qwen3:latest"], 8, 32_768),
        _local("qwen3-32b", "mid", ["qwen3:32b"], 32, 32_768),
        _local("llama3.3-70b", "mid", ["llama3.3:70b"], 70, 131_072),
    ],
})

EVERYTHING = {
    "anthropic": True,
    "gemini": True,
    "ollama": ["qwen3:4b", "qwen3:8b", "qwen3:32b", "llama3.3:70b"],
    "lmstudio": False,
}


def client(availability=EVERYTHING, **kwargs) -> Client:
    kwargs.setdefault("cards", CARDS)
    return Client(availability=availability, **kwargs)


def test_selected_model_routes_back_to_its_provider():
    sel = client().select_model("mid")
    c = client()
    assert c.resolve(sel.model).spec.name == sel.provider
    assert c.card(sel.model) is sel.card


def test_prefer_local_picks_a_local_model_on_the_rung():
    sel = client().select_model("mid", prefer="local")
    assert sel.provider == "ollama"
    assert sel.card.rung is Rung.MID
    assert "preference is local" in sel.reason


def test_prefer_local_falls_back_to_hosted_and_says_so():
    sel = client({"anthropic": True, "ollama": ["qwen3:8b"]}).select_model(
        "mid", prefer="local"
    )
    assert sel.model == "anthropic/claude-haiku-4-5"
    assert "no local model is eligible" in sel.reason


def test_prefer_cheapest_counts_local_as_free():
    sel = client().select_model("mid", prefer="cheapest")
    assert sel.provider == "ollama"
    hosted_only = client({"anthropic": True, "gemini": True}).select_model("mid")
    # Gemini 2.5 Flash has the lowest blended list price on the mid rung.
    assert hosted_only.model == "gemini/gemini-2.5-flash"


def test_prefer_fastest_takes_the_closest_rung_and_hosted_before_local():
    sel = client().select_model("small", prefer="fastest")
    # Nothing hosted sits on small, so the closest rung is the local 8B.
    assert sel.model == "ollama/qwen3:8b"
    sel = client().select_model("mid", prefer="fastest")
    assert not sel.card.open_weight  # hosted mid beats the local mid models
    assert sel.card.rung is Rung.MID


def test_rung_or_above_by_default_never_below():
    sel = client({"anthropic": True}).select_model("small")
    assert sel.card.rung is Rung.MID  # nearest at or above
    assert "nearest available at or above small" in sel.reason
    ranks = [c.rung for c in sel.candidates if c.eligible]
    assert Rung.NANO not in ranks and Rung.SMALL not in ranks


def test_exact_rung_when_higher_is_not_allowed():
    with pytest.raises(NoModelAvailableError, match="exact rung requested"):
        client({"anthropic": True}).select_model("small", allow_higher=False)


def test_needs_filter_on_card_capabilities():
    sel = client().select_model("small", needs=[Capability.VISION])
    assert sel.provider in {"anthropic", "gemini"}
    qwen = next(c for c in sel.candidates if c.qualified == "ollama/qwen3:8b")
    assert qwen.excluded == ("lacks vision",)


def test_structured_output_grade_is_a_floor():
    sel = client().select_model("small", structured_output="reliable", prefer="local")
    assert not sel.card.open_weight
    assert any(
        "structured output is best_effort, reliable required" in r
        for c in sel.excluded for r in c.excluded
    )
    best_effort = client().select_model("small", structured_output="best_effort", prefer="local")
    assert best_effort.provider == "ollama"


def test_min_context():
    sel = client().select_model("mid", min_context=64_000, prefer="local")
    # qwen3-32b is 32k; llama3.3-70b has 128k.
    assert sel.model == "ollama/llama3.3:70b"


def test_local_only():
    sel = client().select_model("mid", local_only=True, prefer="cheapest")
    assert sel.provider == "ollama"
    with pytest.raises(NoModelAvailableError, match="local only"):
        client({"anthropic": True}).select_model("mid", local_only=True)


def test_served_models_decide_which_local_id_is_used():
    # Ollama names the default qwen3 tag "qwen3:latest"; the card lists it.
    sel = client({"ollama": ["qwen3:latest"]}).select_model("small")
    assert sel.model == "ollama/qwen3:latest"
    # A model the server does not have is excluded, with the ids looked for.
    with pytest.raises(NoModelAvailableError) as exc:
        client({"ollama": ["qwen3:8b"]}).select_model("mid")
    assert "ollama does not serve it" in str(exc.value)


def test_nothing_fits_names_what_was_missing():
    with pytest.raises(NoModelAvailableError) as exc:
        client({}).select_model("large", needs=["vision"])
    message = str(exc.value)
    assert "rung large or above; needs vision" in message
    assert "anthropic (declared unavailable)" in message
    assert "anthropic/claude-sonnet-4-6" in message  # would fit if available
    assert exc.value.candidates
    assert exc.value.selection_request.rung is Rung.LARGE


def test_hosted_availability_follows_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    check = key_availability(Client().key_provider)
    registry = default_registry()
    yes = check(registry["anthropic"])
    no = check(registry["gemini"])
    assert yes.available and "ANTHROPIC_API_KEY is set" in yes.detail
    assert not no.available and "GEMINI_API_KEY is not set" in no.detail
    assert "sk-test" not in repr(yes)


def test_custom_key_provider_decides_availability():
    def keys(spec):
        if spec.name == "gemini":
            return "per-user-key"
        raise MissingAPIKeyError("no key")

    check = key_availability(keys)
    registry = default_registry()
    assert check(registry["gemini"]).available
    assert not check(registry["anthropic"]).available


class _FakeOpener:
    def __init__(self, body=None, error=None):
        self.body, self.error, self.urls = body, error, []

    def open(self, url, timeout):
        self.urls.append(url)
        if self.error:
            raise self.error
        return io.BytesIO(json.dumps(self.body).encode())


def test_probe_lists_a_local_servers_models(monkeypatch):
    opener = _FakeOpener({"data": [{"id": "qwen3:8b"}, {"id": "llama3.3:70b"}]})
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: opener)
    result = probe_local_server(default_registry()["ollama"])
    assert result.available
    assert result.models == {"qwen3:8b", "llama3.3:70b"}
    assert opener.urls == ["http://localhost:11434/v1/models"]


def test_probe_failure_is_unavailable(monkeypatch):
    opener = _FakeOpener(error=OSError("connection refused"))
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: opener)
    result = probe_local_server(default_registry()["lmstudio"])
    assert not result.available
    assert "localhost:1234" in result.detail


def test_availability_is_cached_until_refresh():
    calls = []

    def check(spec):
        calls.append(spec.name)
        return ProviderAvailability(spec.name, spec.name == "anthropic", "fake")

    c = Client(availability=check, cards=CARDS)
    c.select_model("mid")
    first = len(calls)
    c.select_model("large")
    assert len(calls) == first
    c.select_model("large", refresh=True)
    assert len(calls) == 2 * first
    assert [a.provider for a in c.availability(["anthropic"])] == ["anthropic"]


def test_record_is_json_serialisable():
    sel = client().select_model("mid", needs=["tool_calling"], prefer="local")
    record = json.loads(json.dumps(sel.to_dict()))
    assert record["model"] == sel.model
    assert record["request"]["prefer"] == "local"
    assert record["cards_updated"] == sel.cards_updated
    assert record["candidates"][0]["rank"] == 1
    assert sel.explain().startswith(f"Selected {sel.model}")


def test_consumer_cards_take_part_in_selection():
    mine = ModelCard(
        id="house-model",
        rung=Rung.LARGE,
        routes=(Route("groq", ("house-70b",), Cost(0.10, 0.10)),),
        structured_output=Reliability.RELIABLE,
        context_window=131072,
    )
    c = Client(
        cards=ModelCards([mine]).merged([]),
        availability={"groq": True},
    )
    sel = c.select_model("large")
    assert sel.model == "groq/house-70b"
    assert c.resolve(sel.model).spec.name == "groq"


def test_route_to_an_unregistered_provider_is_excluded():
    card = ModelCard(
        id="elsewhere",
        rung=Rung.MID,
        routes=(Route("vllm", ("m",)),),
        structured_output=Reliability.RELIABLE,
    )
    with pytest.raises(NoModelAvailableError, match="not registered"):
        Client(cards=ModelCards([card]), availability={}).select_model("mid")


def test_strings_and_enums_are_interchangeable():
    a = client().select_model("mid", needs=["tool_calling"], structured_output="reliable")
    b = client().select_model(
        Rung.MID, needs=[Capability.TOOL_CALLING], structured_output=Reliability.RELIABLE
    )
    assert a.model == b.model
