"""Model cards — what each *model* is, as opposed to what each provider can do.

A :class:`ModelCard` describes one model (for open-weight models: one family +
size + quantization) with a vendor-neutral :class:`Rung`, a coarse
structured-output :class:`Reliability` grade, its context window, per-model
capability overrides and, per provider that serves it, the model ids and the
list price. Cards ship as a versioned JSON data file inside the package
(``model_cards.json``) and are parsed with :mod:`json` — no dynamic code, in
keeping with the static provider registry.

Cards override the provider's :class:`~inferenceswitch.Capabilities` per model
rather than replacing them: a card adds or removes feature tokens relative to
its provider's defaults, and the provider's structured-output *mechanism*
(``Capabilities.structured_output_for``) still decides how a request is made.

See ``docs/model-cards.md`` for the placement rubric and field reference.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from .capabilities import Capabilities, Capability
from .errors import ModelCardError
from .messages import Usage

#: The card data format this version of inferenceswitch reads. A file with any
#: other ``format`` is rejected rather than half-understood.
CARD_FORMAT = 1


class Rung(str, Enum):
    """A vendor-neutral capability rung, lowest first. Provisional names."""

    NANO = "nano"
    SMALL = "small"
    MID = "mid"
    LARGE = "large"
    FRONTIER = "frontier"

    @property
    def index(self) -> int:
        return _RUNG_ORDER.index(self)


_RUNG_ORDER = tuple(Rung)


class Reliability(str, Enum):
    """How dependably a model returns schema-valid JSON through inferenceswitch,
    lowest first. A grade rather than a number: the evidence is judgement."""

    #: No usable structured output.
    NONE = "none"
    #: Usually valid, but not enforced: JSON mode plus the schema in the prompt,
    #: validated client-side. Expect occasional retries on long or nested output.
    BEST_EFFORT = "best_effort"
    #: Enforced by the provider (constrained decoding or a strict schema).
    RELIABLE = "reliable"

    @property
    def index(self) -> int:
        return _RELIABILITY_ORDER.index(self)


_RELIABILITY_ORDER = tuple(Reliability)


@dataclass(frozen=True)
class Cost:
    """List price in USD per million tokens."""

    input: float
    output: float
    #: Prompt tokens read from the provider's cache, and written to it (the
    #: default cache lifetime: Anthropic's 5-minute write). ``None`` where the
    #: card gives no cache price; :meth:`usd` then leaves those tokens out and
    #: says so.
    cache_read: float | None = None
    cache_write: float | None = None

    #: Input:output weighting for :attr:`blended`, the common 3:1 convention
    #: for prompt-heavy work.
    BLEND_INPUT_WEIGHT = 3

    @property
    def blended(self) -> float:
        """One number to rank by: (3 x input + output) / 4."""
        w = self.BLEND_INPUT_WEIGHT
        return (w * self.input + self.output) / (w + 1)

    def usd(self, usage: Usage) -> tuple[float, tuple[str, ...]]:
        """What ``usage`` cost at these prices, and the cache counts it could not
        price (``"cache_read"``, ``"cache_write"``): tokens the call reported
        that this card has no price for, left out of the figure rather than
        priced as zero without saying so. A count not reported adds nothing."""
        total = (usage.input_tokens or 0) * self.input + (usage.output_tokens or 0) * self.output
        unpriced: list[str] = []
        for name, count, price in (
            ("cache_read", usage.cache_read_tokens, self.cache_read),
            ("cache_write", usage.cache_write_tokens, self.cache_write),
        ):
            if not count:
                continue
            if price is None:
                unpriced.append(name)
            else:
                total += count * price
        return total / 1e6, tuple(unpriced)


@dataclass(frozen=True)
class Route:
    """How one provider serves a card's model."""

    provider: str
    #: Model ids this provider accepts for the model, preferred first. For a
    #: local server the first id it actually lists is used.
    models: tuple[str, ...]
    #: ``None`` for local servers (free per token) or when the price is unknown.
    cost: Cost | None = None


@dataclass(frozen=True)
class OpenWeight:
    """Identity of an open-weight model, independent of who serves it."""

    family: str
    #: Parameter count in billions (total, for mixture-of-experts models).
    params_b: float
    #: The quantization the route ids resolve to by default (e.g. ``Q4_K_M``).
    #: A different quantization of the same weights is a separate card.
    quantization: str | None = None
    #: Active parameters per token in billions, for mixture-of-experts models.
    active_params_b: float | None = None


@dataclass(frozen=True)
class ModelCard:
    id: str
    rung: Rung
    routes: tuple[Route, ...]
    structured_output: Reliability
    #: Maximum context in tokens. For local servers this is the model's native
    #: window; the server may be configured lower.
    context_window: int | None = None
    #: Capability tokens added to / removed from the provider's defaults.
    features_add: frozenset[Capability] = frozenset()
    features_remove: frozenset[Capability] = frozenset()
    open_weight: OpenWeight | None = None
    notes: str = ""

    def route(self, provider: str) -> Route | None:
        return next((r for r in self.routes if r.provider == provider), None)

    def features(self, provider_caps: Capabilities) -> frozenset[Capability]:
        """This model's features on a provider whose defaults are ``provider_caps``."""
        return (provider_caps.features | self.features_add) - self.features_remove

    def capabilities(self, provider_caps: Capabilities) -> Capabilities:
        """``provider_caps`` with this card's feature overrides applied."""
        return replace(provider_caps, features=self.features(provider_caps))

    # ── (de)serialisation ────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelCard:
        where = f"card {data.get('id', '?')!r}"
        _check_keys(data, _CARD_KEYS, where)
        try:
            routes_raw = data["routes"]
            if not isinstance(routes_raw, Mapping) or not routes_raw:
                raise ModelCardError(f"{where}: 'routes' must be a non-empty object")
            routes = tuple(_route_from(p, r, where) for p, r in routes_raw.items())
            features = data.get("features") or {}
            _check_keys(features, {"add", "remove"}, f"{where} features")
            ow = data.get("open_weight")
            if ow is not None:
                _check_keys(ow, _OPEN_WEIGHT_KEYS, f"{where} open_weight")
                ow = OpenWeight(**ow)
            return cls(
                id=_str(data["id"], f"{where} id"),
                rung=Rung(data["rung"]),
                routes=routes,
                structured_output=Reliability(data["structured_output"]),
                context_window=_context_window(data.get("context_window"), where),
                features_add=frozenset(Capability(c) for c in features.get("add", ())),
                features_remove=frozenset(Capability(c) for c in features.get("remove", ())),
                open_weight=ow,
                notes=data.get("notes", ""),
            )
        except KeyError as exc:
            raise ModelCardError(f"{where}: missing field {exc.args[0]!r}") from None
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ModelCardError):
                raise
            raise ModelCardError(f"{where}: {exc}") from None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "rung": self.rung.value,
            "routes": {
                r.provider: {
                    "models": list(r.models),
                    "cost": None if r.cost is None else _cost_dict(r.cost),
                }
                for r in self.routes
            },
            "structured_output": self.structured_output.value,
            "context_window": self.context_window,
        }
        if self.features_add or self.features_remove:
            out["features"] = {
                "add": sorted(c.value for c in self.features_add),
                "remove": sorted(c.value for c in self.features_remove),
            }
        if self.open_weight is not None:
            out["open_weight"] = {
                k: v for k, v in vars(self.open_weight).items() if v is not None
            }
        if self.notes:
            out["notes"] = self.notes
        return out


_CARD_KEYS = {
    "id", "rung", "routes", "structured_output", "context_window",
    "features", "open_weight", "notes",
}
_OPEN_WEIGHT_KEYS = {"family", "params_b", "quantization", "active_params_b"}


def _check_keys(data: Any, allowed: set[str], where: str) -> None:
    if not isinstance(data, Mapping):
        raise ModelCardError(f"{where}: expected an object")
    unknown = set(data) - allowed
    if unknown:
        # Strict, so a typo ("context_windw") fails loudly instead of silently
        # leaving a field at its default.
        raise ModelCardError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelCardError(f"{where}: expected a non-empty string")
    return value


def _context_window(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelCardError(f"{where}: 'context_window' must be a positive integer or null")
    return value


def _price(value: Any) -> float | None:
    return None if value is None else float(value)


def _cost_dict(cost: Cost) -> dict[str, float]:
    out = {"input": cost.input, "output": cost.output}
    if cost.cache_read is not None:
        out["cache_read"] = cost.cache_read
    if cost.cache_write is not None:
        out["cache_write"] = cost.cache_write
    return out


def _route_from(provider: str, data: Any, where: str) -> Route:
    _check_keys(data, {"models", "cost"}, f"{where} route {provider!r}")
    models = data.get("models")
    if not models or isinstance(models, str):
        raise ModelCardError(f"{where} route {provider!r}: 'models' must be a non-empty list")
    cost = data.get("cost")
    if cost is not None:
        _check_keys(
            cost, {"input", "output", "cache_read", "cache_write"},
            f"{where} route {provider!r} cost",
        )
        cost = Cost(
            float(cost["input"]),
            float(cost["output"]),
            _price(cost.get("cache_read")),
            _price(cost.get("cache_write")),
        )
    return Route(provider, tuple(_str(m, f"{where} model id") for m in models), cost)


class ModelCards:
    """An ordered, id-keyed set of :class:`ModelCard`.

    Immutable: :meth:`merged` returns a new set, so the shipped defaults are
    never changed in place by one consumer.
    """

    def __init__(self, cards: Iterable[ModelCard] = (), *, updated: str | None = None) -> None:
        self._by_id: dict[str, ModelCard] = {}
        for card in cards:
            if card.id in self._by_id:
                raise ModelCardError(f"Duplicate model card id: {card.id!r}")
            self._by_id[card.id] = card
        #: ISO date the data was last reviewed, from the file's ``updated``.
        self.updated = updated

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelCards:
        _check_keys(data, {"format", "updated", "cards"}, "model card file")
        if data.get("format") != CARD_FORMAT:
            raise ModelCardError(
                f"Unsupported model card format {data.get('format')!r}; "
                f"this inferenceswitch reads format {CARD_FORMAT}."
            )
        return cls(
            (ModelCard.from_dict(c) for c in data.get("cards", ())),
            updated=data.get("updated"),
        )

    @classmethod
    def from_json(cls, text: str) -> ModelCards:
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> ModelCards:
        """Read a card file in the same format as the shipped one."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ── combining ────────────────────────────────────────────────────────────

    def merged(self, other: ModelCards | Iterable[ModelCard]) -> ModelCards:
        """A new set: these cards, with ``other``'s added and any card sharing
        an id replaced by ``other``'s. ``updated`` is kept from ``self``."""
        by_id = dict(self._by_id)
        for card in other:
            by_id[card.id] = card
        return ModelCards(by_id.values(), updated=self.updated)

    def without(self, *ids: str) -> ModelCards:
        """A new set with the named cards removed."""
        return ModelCards(
            (c for c in self if c.id not in ids), updated=self.updated
        )

    # ── lookup ───────────────────────────────────────────────────────────────

    def get(self, card_id: str) -> ModelCard:
        try:
            return self._by_id[card_id]
        except KeyError:
            raise KeyError(f"No model card {card_id!r}") from None

    def find(self, provider: str, model: str) -> ModelCard | None:
        """The card whose ``provider`` route lists ``model``, if any."""
        for card in self:
            route = card.route(provider)
            if route is not None and model in route.models:
                return card
        return None

    def __iter__(self) -> Iterator[ModelCard]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, card_id: object) -> bool:
        return card_id in self._by_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CARD_FORMAT,
            "updated": self.updated,
            "cards": [c.to_dict() for c in self],
        }


@lru_cache(maxsize=1)
def default_cards() -> ModelCards:
    """The cards shipped with this version of inferenceswitch."""
    text = resources.files(__package__).joinpath("model_cards.json").read_text("utf-8")
    return ModelCards.from_json(text)
