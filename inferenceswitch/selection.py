"""Resolve (rung, requirements, preference) to an available model.

Three stages, each recorded so the caller can show why:

1. **Availability.** A hosted provider is available when its key resolves
   (the client's ``key_provider``; by default, its env var is set). A local
   server is available when it answers ``GET {base_url}/models``; the models it
   lists are then the only ones of its routes that count. Tests and
   environments that must not probe pass their own availability.
2. **Requirements.** Rung (this rung or above, unless ``allow_higher=False``),
   capability tokens, structured-output grade, context window, locality.
3. **Preference.** The eligible candidates are ranked by ``prefer``; see
   :class:`Preference` for each ordering.

The result is a :class:`Selection`: the routable ``provider/model`` plus every
candidate considered, why each excluded one was excluded, and why the winner
won. :meth:`Selection.to_dict` gives a JSON-able record for run logs.
"""
from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .capabilities import Capability
from .cards import ModelCard, ModelCards, Reliability, Rung
from .errors import NoModelAvailableError
from .registry import ProviderSpec, Registry

#: Seconds to wait for a local server's model list before calling it unreachable.
LOCAL_PROBE_TIMEOUT = 0.5


class Preference(str, Enum):
    """How eligible candidates are ranked. Each ordering ends with the
    candidate's position in the card set, so results are deterministic."""

    #: Local first, then the closest rung, then the lowest price.
    LOCAL = "local"
    #: Lowest blended price (local counts as free), then the closest rung.
    CHEAPEST = "cheapest"
    #: Closest rung, then hosted before local (local throughput depends on the
    #: machine and is unknown), then fewer parameters, then the lowest price as
    #: a proxy for the faster tier. There is no measured speed on the cards.
    FASTEST = "fastest"


# ── availability ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderAvailability:
    """Whether a provider can be used here, and why (never containing a key)."""

    provider: str
    available: bool
    detail: str
    #: Model ids the provider reported serving, or ``None`` when it was not
    #: asked (hosted providers): every id on the cards is then trusted.
    models: frozenset[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "available": self.available,
            "detail": self.detail,
            "models": None if self.models is None else sorted(self.models),
        }


#: Decides availability for one provider.
AvailabilityCheck = Callable[[ProviderSpec], ProviderAvailability]


def static_availability(
    available: Mapping[str, bool | Iterable[str]],
) -> AvailabilityCheck:
    """Availability declared up front instead of checked, for tests and for
    environments that must not probe.

    Map a provider name to ``True`` (available, trust the cards' ids), to an
    iterable of model ids (available, serving exactly those), or to ``False``.
    A provider not in the mapping is unavailable.
    """

    def check(spec: ProviderSpec) -> ProviderAvailability:
        value = available.get(spec.name, False)
        if value is False:
            return ProviderAvailability(spec.name, False, "declared unavailable")
        if value is True:
            return ProviderAvailability(spec.name, True, "declared available")
        models = frozenset(value)
        return ProviderAvailability(
            spec.name, True, f"declared available with {len(models)} model(s)", models
        )

    return check


def key_availability(key_provider: Callable[[ProviderSpec], str]) -> AvailabilityCheck:
    """The default check: hosted providers by key, local servers by probe."""

    def check(spec: ProviderSpec) -> ProviderAvailability:
        if spec.is_local:
            return probe_local_server(spec)
        try:
            key = key_provider(spec)
        except Exception:  # noqa: BLE001 - any failure means "no key here"
            key = None
        if key:
            where = f"${spec.api_key_env} is set" if spec.api_key_env else "key provided"
            return ProviderAvailability(spec.name, True, where)
        where = f"${spec.api_key_env} is not set" if spec.api_key_env else "no key"
        return ProviderAvailability(spec.name, False, where)

    return check


def probe_local_server(
    spec: ProviderSpec, timeout: float = LOCAL_PROBE_TIMEOUT
) -> ProviderAvailability:
    """Ask an OpenAI-compatible local server which models it serves.

    Uses the registry's fixed ``base_url`` only, bypasses any proxy, and sends
    no key. Unreachable, or an answer that is not a model list, is unavailable.
    """
    url = f"{(spec.base_url or '').rstrip('/')}/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        models = frozenset(
            str(m["id"]) for m in body.get("data", ()) if isinstance(m, dict) and "id" in m
        )
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return ProviderAvailability(spec.name, False, f"no server answering at {spec.base_url}")
    return ProviderAvailability(
        spec.name, True, f"server at {spec.base_url} lists {len(models)} model(s)", models
    )


# ── selection ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SelectionRequest:
    """What was asked for, normalised. Recorded on every result."""

    rung: Rung
    needs: frozenset[Capability] = frozenset()
    structured_output: Reliability | None = None
    min_context: int | None = None
    local_only: bool = False
    prefer: Preference = Preference.CHEAPEST
    allow_higher: bool = True

    def describe(self) -> str:
        parts = [f"rung {self.rung.value}{' or above' if self.allow_higher else ' exactly'}"]
        if self.needs:
            parts.append("needs " + ", ".join(sorted(c.value for c in self.needs)))
        if self.structured_output is not None:
            parts.append(f"structured output {self.structured_output.value} or better")
        if self.min_context:
            parts.append(f"context >= {self.min_context:,}")
        if self.local_only:
            parts.append("local only")
        return "; ".join(parts) + f"; prefer {self.prefer.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung.value,
            "needs": sorted(c.value for c in self.needs),
            "structured_output": None if self.structured_output is None
            else self.structured_output.value,
            "min_context": self.min_context,
            "local_only": self.local_only,
            "prefer": self.prefer.value,
            "allow_higher": self.allow_higher,
        }


@dataclass(frozen=True)
class Candidate:
    """One (card, provider) route that was considered."""

    card_id: str
    provider: str
    #: The bare model id that would be sent: the first of the route's ids the
    #: provider lists, else the route's first id.
    model: str
    rung: Rung
    is_local: bool
    #: Blended USD per million tokens; 0 for local, ``None`` when unknown.
    blended_cost: float | None
    #: Why it was excluded; empty when eligible.
    excluded: tuple[str, ...] = ()
    #: 1-based position among eligible candidates; ``None`` when excluded.
    rank: int | None = None

    @property
    def eligible(self) -> bool:
        return not self.excluded

    @property
    def qualified(self) -> str:
        return f"{self.provider}/{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "card": self.card_id,
            "provider": self.provider,
            "model": self.model,
            "rung": self.rung.value,
            "is_local": self.is_local,
            "blended_cost": self.blended_cost,
            "excluded": list(self.excluded),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class Selection:
    """The chosen model and the explanation for it."""

    #: Routable ``provider/model`` — pass straight to ``chat(model=...)``.
    model: str
    provider: str
    card: ModelCard
    request: SelectionRequest
    #: Why the winner won, in one sentence.
    reason: str
    #: Every candidate: eligible ones in rank order, then the excluded.
    candidates: tuple[Candidate, ...]
    availability: tuple[ProviderAvailability, ...]
    #: ``updated`` date of the card data used.
    cards_updated: str | None = None

    @property
    def excluded(self) -> tuple[Candidate, ...]:
        return tuple(c for c in self.candidates if not c.eligible)

    def explain(self) -> str:
        """A multi-line, human-readable account of the decision."""
        lines = [f"Selected {self.model} for {self.request.describe()}.", self.reason]
        for c in self.candidates:
            if c.eligible:
                lines.append(f"  #{c.rank} {c.qualified} ({c.rung.value})")
            else:
                lines.append(f"  -- {c.qualified} [{c.card_id}]: {'; '.join(c.excluded)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able record of the decision, for run logs."""
        return {
            "model": self.model,
            "provider": self.provider,
            "card": self.card.id,
            "rung": self.card.rung.value,
            "reason": self.reason,
            "request": self.request.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "availability": [a.to_dict() for a in self.availability],
            "cards_updated": self.cards_updated,
        }


def make_request(
    rung: Rung | str,
    *,
    needs: Iterable[Capability | str] = (),
    structured_output: Reliability | str | None = None,
    min_context: int | None = None,
    local_only: bool = False,
    prefer: Preference | str = Preference.CHEAPEST,
    allow_higher: bool = True,
) -> SelectionRequest:
    """Normalise caller input (strings from a profile file are fine)."""
    if isinstance(needs, (str, Capability)):
        needs = (needs,)
    return SelectionRequest(
        rung=Rung(rung),
        needs=frozenset(Capability(n) for n in needs),
        structured_output=None if structured_output is None else Reliability(structured_output),
        min_context=min_context,
        local_only=local_only,
        prefer=Preference(prefer),
        allow_higher=allow_higher,
    )


def select(
    request: SelectionRequest,
    cards: ModelCards,
    registry: Registry,
    availability: Callable[[str], ProviderAvailability],
) -> Selection:
    """Run the three stages. ``availability`` maps a provider name to its
    (possibly cached) :class:`ProviderAvailability`."""
    checked: dict[str, ProviderAvailability] = {}
    eligible: list[tuple[tuple, Candidate, ModelCard]] = []
    excluded: list[Candidate] = []

    for position, card in enumerate(cards):
        for route in card.routes:
            if route.provider not in registry:
                excluded.append(Candidate(
                    card.id, route.provider, route.models[0], card.rung, False,
                    None, (f"provider {route.provider!r} is not registered",),
                ))
                continue
            spec = registry[route.provider]
            if route.provider not in checked:
                checked[route.provider] = availability(route.provider)
            avail = checked[route.provider]

            reasons: list[str] = []
            model = route.models[0]
            if not avail.available:
                reasons.append(f"{route.provider} unavailable: {avail.detail}")
            elif avail.models is not None:
                served = next((m for m in route.models if m in avail.models), None)
                if served is not None:
                    model = served
                else:
                    reasons.append(
                        f"{route.provider} does not serve it (looked for "
                        f"{', '.join(route.models)})"
                    )
            reasons.extend(_requirement_failures(request, card, spec))

            if spec.is_local:
                cost: float | None = 0.0
            else:
                cost = None if route.cost is None else round(route.cost.blended, 6)
            candidate = Candidate(
                card.id, route.provider, model, card.rung, spec.is_local, cost, tuple(reasons)
            )
            if reasons:
                excluded.append(candidate)
            else:
                eligible.append((_rank_key(request, card, candidate, position), candidate, card))

    eligible.sort(key=lambda item: item[0])
    ranked = [replace(c, rank=i) for i, (_, c, _) in enumerate(eligible, start=1)]
    candidates = tuple(ranked) + tuple(excluded)
    avail_list = tuple(checked.values())

    if not ranked:
        raise NoModelAvailableError(
            _nothing_fits_message(request, excluded, avail_list),
            request=request,
            candidates=candidates,
            availability=avail_list,
        )

    winner, winner_card = ranked[0], eligible[0][2]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return Selection(
        model=winner.qualified,
        provider=winner.provider,
        card=winner_card,
        request=request,
        reason=_reason(request, winner, runner_up),
        candidates=candidates,
        availability=avail_list,
        cards_updated=cards.updated,
    )


def _requirement_failures(
    request: SelectionRequest, card: ModelCard, spec: ProviderSpec
) -> list[str]:
    reasons: list[str] = []
    distance = card.rung.index - request.rung.index
    if distance < 0:
        reasons.append(f"rung {card.rung.value} is below {request.rung.value}")
    elif distance > 0 and not request.allow_higher:
        reasons.append(f"rung {card.rung.value} is not {request.rung.value} (exact rung requested)")
    missing = request.needs - card.features(spec.capabilities)
    if missing:
        reasons.append("lacks " + ", ".join(sorted(c.value for c in missing)))
    if (
        request.structured_output is not None
        and card.structured_output.index < request.structured_output.index
    ):
        reasons.append(
            f"structured output is {card.structured_output.value}, "
            f"{request.structured_output.value} required"
        )
    if request.min_context:
        if card.context_window is None:
            reasons.append("context window unknown")
        elif card.context_window < request.min_context:
            reasons.append(
                f"context window {card.context_window:,} < {request.min_context:,}"
            )
    if request.local_only and not spec.is_local:
        reasons.append("hosted, and local only was requested")
    return reasons


def _rank_key(
    request: SelectionRequest, card: ModelCard, candidate: Candidate, position: int
) -> tuple:
    distance = card.rung.index - request.rung.index
    cost = math.inf if candidate.blended_cost is None else candidate.blended_cost
    params = card.open_weight.params_b if card.open_weight else 0.0
    if request.prefer is Preference.LOCAL:
        return (not candidate.is_local, distance, cost, position)
    if request.prefer is Preference.FASTEST:
        return (distance, candidate.is_local, params, cost, position)
    return (cost, distance, position)


def _describe(c: Candidate) -> str:
    cost = (
        "local, no per-token cost" if c.is_local
        else "price unknown" if c.blended_cost is None
        else f"${c.blended_cost:.2f}/M tokens blended"
    )
    return f"{c.qualified} (rung {c.rung.value}, {cost})"


def _reason(request: SelectionRequest, winner: Candidate, runner_up: Candidate | None) -> str:
    rung_note = (
        "on the requested rung" if winner.rung is request.rung
        else f"on rung {winner.rung.value}, the nearest available at or above {request.rung.value}"
    )
    basis = {
        Preference.LOCAL: "the preference is local" if winner.is_local
        else "the preference is local, but no local model is eligible",
        Preference.CHEAPEST: "the preference is cheapest",
        Preference.FASTEST: "the preference is fastest: closest rung, hosted before "
        "local, then smaller and cheaper as a proxy for speed",
    }[request.prefer]
    text = f"Chose {_describe(winner)}, {rung_note}, because {basis}."
    if runner_up is not None:
        text += f" Next best: {_describe(runner_up)}."
    else:
        text += " It was the only eligible candidate."
    return text


def _nothing_fits_message(
    request: SelectionRequest,
    excluded: list[Candidate],
    availability: tuple[ProviderAvailability, ...],
) -> str:
    unavailable = [a for a in availability if not a.available]
    reasons: dict[str, int] = {}
    for c in excluded:
        for r in c.excluded:
            if " unavailable: " in r:
                continue  # summarised per provider below
            key = r.split(" (looked for ")[0]
            reasons[key] = reasons.get(key, 0) + 1
    lines = [f"No available model fits {request.describe()}."]
    fit_if_available = [
        c.qualified for c in excluded
        if all(" unavailable: " in r or " does not serve it " in r for r in c.excluded)
    ]
    if fit_if_available:
        lines.append(
            "Would fit if their provider were available and serving them: "
            + ", ".join(fit_if_available) + "."
        )
    if unavailable:
        lines.append(
            "Unavailable providers: "
            + "; ".join(f"{a.provider} ({a.detail})" for a in unavailable)
            + "."
        )
    if reasons:
        lines.append(
            "Other exclusions: "
            + "; ".join(f"{r} (x{n})" for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]))
            + "."
        )
    if not excluded:
        lines.append("There are no model cards to choose from.")
    return " ".join(lines)
