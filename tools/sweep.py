#!/usr/bin/env python
"""Sweep the model cards against public catalogs: prices, new models, benchmarks.

Run it by hand or from ``.github/workflows/sweep.yml``. It uses the standard
library only, makes read-only requests to two public sources and never needs a
key:

* **OpenRouter's catalog** (``https://openrouter.ai/api/v1/models``) and each
  model's endpoint list: which open models exist, their context sizes, tool and
  structured-output support, and the price each host charges. It is the source
  of truth for ``openrouter`` routes, priced at the **median host that supports
  structured output**: the catalog's own headline price is the cheapest host,
  which for an open model is often an outlier (Kimi K3.1 Pro lists at $0.24
  input against its maker's $2). For another provider's route it is only a
  cross-check: where a model has one first-party price (Anthropic, OpenAI)
  OpenRouter repeats it, but it can show a blended or introductory price where
  the vendor's own list says otherwise. A difference is reported, never applied.
* **Epoch AI's Capabilities Index** (``https://epoch.ai/data/eci_scores.csv``,
  CC-BY 4.0, credit Epoch AI): one score stitched from 50+ benchmarks of very
  different difficulty, with a 95% interval. The main view of a model's rung.
* **Artificial Analysis** (``https://artificialanalysis.ai/models/<slug>``): an
  independent Intelligence Index and benchmark scores, read from the data the
  model's page embeds. It covers the newest models before Epoch does. A page
  that changes shape is reported as unreadable; it never fails the run.

Commands::

    python tools/sweep.py report              # markdown report on stdout
    python tools/sweep.py update              # rewrite openrouter prices and benchmarks in place
    python tools/sweep.py report --cards PATH # a card file other than the shipped one
    python tools/sweep.py report --out report.md --findings count.txt
    python tools/sweep.py chart               # redraw docs/model-cards-chart.svg and the README table

``update`` also redraws the chart and the provider-by-rung table the README
carries, so a sweep's pull request shows the cards that moved.

``update`` changes only what a machine can know: ``openrouter`` route prices,
the Artificial Analysis ``benchmarks`` of cards that already name a page in their
``source``, and the ``epoch_eci`` of cards whose id or Artificial Analysis name
matches a row in Epoch's data. Rungs, notes, feature flags and other providers' prices
stay a person's call; the report suggests, a reviewer decides.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS_PATH = ROOT / "inferenceswitch" / "model_cards.json"
CHART_PATH = ROOT / "docs" / "model-cards-chart.svg"
README_PATH = ROOT / "README.md"

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
AA_PAGE = "https://artificialanalysis.ai/models/{slug}"
AA_SOURCE = "artificialanalysis.ai/models/{slug}"
USER_AGENT = "inferenceswitch-sweep (+https://github.com/IDEMSInternational/inferenceswitch)"

#: Artificial Analysis field -> (benchmark name on the card, factor to a 0-100
#: scale). Fields not listed are not recorded: a card keeps few, comparable
#: scores rather than every column a page has.
AA_FIELDS = {
    "intelligenceIndex": ("aa_intelligence_index", 1.0),
    "gpqa": ("gpqa_diamond", 100.0),
    "hle": ("hle", 100.0),
    "scicode": ("scicode", 100.0),
    "lcr": ("aa_lcr", 100.0),
    "ifbench": ("ifbench", 100.0),
    "tau2": ("tau2", 100.0),
    "terminalBench21": ("terminal_bench_2_1", 100.0),
    "livecodebench": ("livecodebench", 100.0),
}
INDEX = "aa_intelligence_index"

#: A model is a candidate for a new card when OpenRouter lists it with open
#: weights, tools and structured output, and it is newer than this.
NEW_MODEL_DAYS = 120

RUNGS = ("nano", "small", "mid", "large", "frontier")


# ── fetching ────────────────────────────────────────────────────────────────


def _get(url: str, *, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_openrouter() -> dict[str, dict]:
    """OpenRouter's catalog by model id."""
    return {m["id"]: m for m in json.loads(_get(OPENROUTER_URL))["data"]}


def _flight(html: str) -> str:
    """The text a Next.js page streams to its client, joined back together."""
    parts = []
    for m in re.finditer(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html):
        parts.append(json.loads(m.group(1)))
    return "".join(parts)


def read_aa_page(html: str, slug: str) -> dict | None:
    """The model record, and the index version, on an Artificial Analysis model
    page; ``None`` when the page does not have the shape this expects."""
    flight = _flight(html)
    decoder = json.JSONDecoder()
    records = []
    for m in re.finditer(r'"intelligenceIndex":', flight):
        start = flight.rfind('{"id":"', 0, m.start())
        if start < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(flight[start:])
        except ValueError:
            continue
        if obj.get("slug") == slug:
            records.append(obj)
    if not records:
        return None
    # Several effort settings can share a slug; the highest-effort one is the
    # model at its best, which is what a rung is graded on.
    record = max(records, key=lambda r: ((r.get("effort") or {}).get("level") or 0))
    version = re.search(r"Intelligence Index v(\d+(?:\.\d+)*)", html)
    return {"record": record, "version": version.group(1) if version else None}


def fetch_aa(slug: str) -> dict | None:
    try:
        return read_aa_page(_get(AA_PAGE.format(slug=slug)), slug)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def benchmarks_from(aa: dict, today: str) -> dict[str, dict]:
    """The card's ``benchmarks`` entries for an Artificial Analysis record."""
    rec, version = aa["record"], aa["version"]
    setting = (rec.get("effort") or {}).get("label")
    if setting is None and rec.get("isReasoning") is False:
        setting = "non-reasoning"
    out = {}
    for field, (name, factor) in AA_FIELDS.items():
        value = rec.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if field == "lcr" and value == 0:
            # A 0 on the long-context test is a model whose window is shorter
            # than the test's prompts (Qwen3 8B, 32B, 30B-A3B), not a score.
            continue
        entry = {
            "score": round(value * factor, 1),
            "source": AA_SOURCE.format(slug=rec["slug"]),
            "date": today,
        }
        if name == INDEX and version:
            entry["version"] = version
        if setting:
            entry["setting"] = setting
        out[name] = entry
    return out


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.split("/")[-1].split(":")[0].lower())


# ── reading the cards ───────────────────────────────────────────────────────


def load_cards(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aa_slug(card: dict) -> str | None:
    """The Artificial Analysis page a card's scores were read from."""
    for bench in (card.get("benchmarks") or {}).values():
        m = re.match(r"artificialanalysis\.ai/models/([\w.-]+)$", bench.get("source", ""))
        if m:
            return m.group(1)
    return None


def price_of(model: dict) -> tuple[float, float]:
    p = model["pricing"]
    return round(float(p["prompt"]) * 1e6, 4), round(float(p["completion"]) * 1e6, 4)


def endpoint_price(model_id: str) -> dict | None:
    """What an ``openrouter`` route to ``model_id`` is priced at: the median host
    (by 3:1 blended price) among those that take ``response_format``
    structured outputs, or among all hosts when none says it does. ``None`` when
    the endpoint list cannot be read."""
    try:
        data = json.loads(_get(f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"))["data"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        return None
    hosts = data.get("endpoints") or []
    capable = [e for e in hosts if "structured_outputs" in (e.get("supported_parameters") or [])]
    pool = capable or hosts
    if not pool:
        return None

    def per_m(e: dict, field: str) -> float | None:
        v = e["pricing"].get(field)
        return None if v in (None, "") else round(float(v) * 1e6, 6)

    def blended(e: dict) -> float:
        return (3 * (per_m(e, "prompt") or 0) + (per_m(e, "completion") or 0)) / 4

    ranked = sorted(pool, key=blended)
    pick = ranked[(len(ranked) - 1) // 2]
    cost = {"input": per_m(pick, "prompt"), "output": per_m(pick, "completion")}
    for key, field in (("cache_read", "input_cache_read"), ("cache_write", "input_cache_write")):
        value = per_m(pick, field)
        if value:
            cost[key] = value
    return cost


#: A rung is a capability level with a fixed threshold, not a vendor's name for a
#: tier: a card goes on the rung its measured scores put it on, so that changing
#: provider at one rung gives roughly the same results.
#:
#: **These thresholds are frozen.** They were set once, on 2026-11-16, from the
#: reference models of that day, and are never moved when a new generation of
#: references arrives: raising a rung's bar would silently break every tool
#: already pinned to it. Models are re-scored and may change rung; if models
#: outgrow the top rung, add a rung above it. Do not edit these numbers.
#:
#: Placement reads two independent views, because each benchmark is a sigmoid in
#: capability: it separates models only where it is neither at its floor nor at
#: its ceiling, and a model that tops one still gets a fair score on a harder one
#: (Epoch's point in building the ECI).
#:
#: * Epoch Capabilities Index (ECI): one scale stitched from 50+ benchmarks of
#:   very different difficulty, with a 95% interval. Broad and general, so the
#:   main view, but it covers a model only once Epoch has results for it.
#: * Artificial Analysis Intelligence Index: nine agentic, coding and
#:   professional-task evaluations. Covers the newest models first, leans to
#:   agentic work, and sits at its floor for models that are not agentic
#:   (gpt-oss-120b 3.8), where it gives no view. Artificial Analysis re-bases it
#:   between versions, so its edges are fixed *per version*: a score from a
#:   version with no edges here has no view until a person adds them.
#:
#: The 2026-11-16 reference scores the edges sit between (ECI, then AA v5.0.1):
#: small: Gemini 3.5 Flash-Lite 145.6 / 11.7, GPT-5.4 nano 146.4, GPT-5.6 Luna
#: 156.7 / 23.8. mid: Haiku 5.5 160.4 / 36.5, GPT-5.6 Terra 160.7 / 32.3, Gemini
#: 3.8 Flash 158.8 / 30.8, GPT-6 Luna 156.4 / 27.9. large: Sonnet 5.5 166.6 /
#: 54.5, GPT-6 Sol 163.8 / 44.2. frontier: GPT-6 Astra 169.0 / 56.9, Opus 5.5
#: 168.5 / 59.1. ``small`` is wide (about 111 to 152 on the ECI), so its lower
#: part is ``nano``: below 130 on the ECI (GPT-4o mini 123.9, Llama 3.3 70B 126.1,
#: Llama 3.1 8B 111.1; Qwen3 8B 132.1 and GPT-4.1 mini 134.2 are ``small``). The
#: AA index has no nano edge, so a card with no ECI is placed ``nano`` by size
#: (on-device, about 4B and under) and the index only says "small or below".
ECI = "epoch_eci"
ECI_EDGES = (("small", 130.0), ("mid", 152.0), ("large", 163.0), ("frontier", 168.0))
AA_EDGES = {"5.0.1": (("small", 12.0), ("mid", 26.0), ("large", 40.0), ("frontier", 55.0))}
ECI_CSV = "https://epoch.ai/data/eci_scores.csv"
ECI_SOURCE = "epoch.ai/data/eci_scores.csv#{model}"


def _rung_of(score: float, edges, floor: str | None) -> str | None:
    rung = floor
    for name, edge in edges:
        if score >= edge:
            rung = name
    return rung


def suggest_rung(score: float, version: str | None = "5.0.1") -> str | None:
    """The rung the AA index puts a score on; ``None`` below its first edge or for
    a version with no edges."""
    edges = AA_EDGES.get(version or "")
    return _rung_of(score, edges, None) if edges else None


def suggest_rung_eci(score: float) -> str:
    """The rung the ECI puts a score on (``nano`` below the ``small`` edge)."""
    return _rung_of(score, ECI_EDGES, "nano")


#: The individual benchmarks that can stand in for the ECI when Epoch has no row
#: for a card. Each is a sigmoid in capability, so it is used only where it is
#: neither at its floor nor at its ceiling (3% to 97%): a model that tops it still
#: gets its say from a harder one.
EST_BENCH = ("gpqa_diamond", "hle", "ifbench", "tau2", "aa_lcr", "scicode",
             "terminal_bench_2_1", "livecodebench")
EST_RANGE = (3.0, 97.0)
EST_MIN_CARDS = 6


def _logit(score: float) -> float:
    p = score / 100
    return math.log(p / (1 - p))


def _fit(cards: list[dict], skip: dict | None = None) -> dict[str, tuple[float, float, float]]:
    """Per benchmark, ECI = a + b * logit(score), fit by least squares on the cards
    that have both, with the residual spread. A benchmark with fewer than
    ``EST_MIN_CARDS`` such cards is left out."""
    models = {}
    for name in EST_BENCH:
        xs, ys = [], []
        for card in cards:
            b = card.get("benchmarks") or {}
            if card is skip or ECI not in b or name not in b:
                continue
            if EST_RANGE[0] < b[name]["score"] < EST_RANGE[1]:
                xs.append(_logit(b[name]["score"]))
                ys.append(b[ECI]["score"])
        n = len(xs)
        if n < EST_MIN_CARDS:
            continue
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx == 0:
            continue
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        icpt = my - slope * mx
        sd = math.sqrt(sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
        models[name] = (icpt, slope, max(sd, 0.5))
    return models


def _combine(card: dict, models: dict) -> tuple[float, list[str]] | None:
    """The benchmarks' ECI estimates for one card, weighted by inverse residual
    variance; ``None`` when none of its benchmarks is in range."""
    b = card.get("benchmarks") or {}
    parts, used = [], []
    for name, (icpt, slope, sd) in models.items():
        if name in b and EST_RANGE[0] < b[name]["score"] < EST_RANGE[1]:
            parts.append((icpt + slope * _logit(b[name]["score"]), 1 / sd ** 2))
            used.append(name)
    if not parts:
        return None
    return sum(e * w for e, w in parts) / sum(w for _, w in parts), used


class Estimator:
    """An ECI estimated from a card's other benchmarks, for cards Epoch has not
    scored. Fit on the cards that have both, so it is re-fit with each sweep;
    the rung edges it is read against are not. ``rmse`` is its leave-one-out
    error on the cards that have an ECI (held out one at a time), and the 95%
    interval it reports is 1.96 times that: a rough guide, not a guarantee."""

    def __init__(self, cards: list[dict]) -> None:
        self.models = _fit(cards)
        errors = []
        for card in cards:
            eci = (card.get("benchmarks") or {}).get(ECI)
            if not eci:
                continue
            got = _combine(card, _fit(cards, skip=card))
            if got:
                errors.append(got[0] - eci["score"])
        self.rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None
        self.n_checked = len(errors)

    def __call__(self, card: dict) -> dict | None:
        """``{"score", "lo", "hi", "used"}`` for a card, or ``None`` when it has no
        usable benchmark or the estimator could not be checked."""
        got = _combine(card, self.models)
        if got is None or self.rmse is None:
            return None
        score, used = got
        half = 1.96 * max(self.rmse, 1.0)  # a floor: the ECI itself is only good to about 2.5
        return {"score": round(score, 1), "lo": round(score - half, 1), "hi": round(score + half, 1), "used": used}


def grade(benchmarks: dict | None, est: dict | None = None) -> dict:
    """What a card's recorded scores say about its rung.

    ``views`` maps each view that has a say (``eci``, ``est``, ``aa``) to the rung
    it suggests. A card Epoch has scored is read from its ECI; one it has not is
    read from ``est`` (an estimate from its other benchmarks, see
    :class:`Estimator`), if given. ``suggested`` is the lower of the views (a
    consumer pays more for a model placed too low, and gets a wrong answer from
    one placed too high). ``ok`` is the set of rungs the scores do not rule out:
    the suggestion, plus every rung an estimate's 95% interval spans, plus
    ``nano`` when only the AA index spoke and it said ``small`` (the index cannot
    tell them apart). ``borderline`` is true when an ECI or estimate interval
    crosses an edge; ``agree`` is false when the views differ. Everything is
    ``None``/empty when the card has no benchmark that speaks.
    """
    b = benchmarks or {}
    views: dict[str, str] = {}
    borderline = False
    ok: set[str] = set()
    eci = b.get(ECI)
    if eci:
        views["eci"] = suggest_rung_eci(eci["score"])
        low, high = eci.get("interval") or (eci["score"], eci["score"])
        borderline = suggest_rung_eci(low) != suggest_rung_eci(high)
    elif est:
        views["est"] = suggest_rung_eci(est["score"])
        lo_r, hi_r = suggest_rung_eci(est["lo"]), suggest_rung_eci(est["hi"])
        borderline = lo_r != hi_r
        ok |= {r for r in RUNGS if RUNGS.index(lo_r) <= RUNGS.index(r) <= RUNGS.index(hi_r)}
    idx = b.get(INDEX)
    if idx and suggest_rung(idx["score"], idx.get("version")):
        views["aa"] = suggest_rung(idx["score"], idx.get("version"))
    suggested = min(views.values(), key=RUNGS.index) if views else None
    if suggested is not None:
        ok.add(suggested)
        if suggested == "small" and "eci" not in views and "est" not in views:
            ok.add("nano")
    return {"views": views, "suggested": suggested, "borderline": borderline,
            "agree": len(set(views.values())) <= 1, "ok": ok or None}


def settles(rung: str, g: dict) -> bool:
    """Whether a card's ``rung`` is one its scores do not rule out (``grade``'s ``ok``)."""
    return g["ok"] is None or rung in g["ok"]


def fetch_eci() -> dict[str, dict]:
    """Epoch's ECI rows (CC-BY 4.0, https://epoch.ai/benchmarks/eci) by model name."""
    import csv
    import io
    try:
        text = _get(ECI_CSV)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return {}
    return {r["Model"]: r for r in csv.DictReader(io.StringIO(text)) if r.get("eci")}


def eci_entry(row: dict, today: str) -> dict:
    return {
        "score": round(float(row["eci"]), 1),
        "source": ECI_SOURCE.format(model=row["Model"]),
        "date": today,
        "interval": [round(float(row["eci_ci_low"]), 1), round(float(row["eci_ci_high"]), 1)],
    }


def eci_row_for(card: dict, rows: dict[str, dict], aa_name: str | None = None) -> dict | None:
    """The ECI row for a card: the one its recorded source names, or else the one
    whose normalised name equals the card's id or its Artificial Analysis name."""
    for bench in (card.get("benchmarks") or {}).values():
        m = re.match(r"epoch\.ai/data/eci_scores\.csv#(.+)$", bench.get("source", ""))
        if m:
            return rows.get(m.group(1))
    wanted = {norm(card["id"]), norm(card["id"]) + "it"}  # Epoch lists "Gemma 4 31B IT"
    if aa_name:
        wanted.add(norm(re.sub(r"\(.*?\)", "", aa_name)))
    return next((r for name, r in rows.items() if norm(name) in wanted), None)


# ── the report ──────────────────────────────────────────────────────────────

#: Providers whose list price OpenRouter repeats for their own models, so a
#: difference is worth a look. Others (Groq, a vendor's peak price) are routes
#: OpenRouter does not price.
FIRST_PARTY_NAMES = {"anthropic": "anthropic", "openai": "openai", "gemini": "google",
                     "mistral": "mistralai", "deepseek": "deepseek", "xai": "x-ai"}


def route_drift(cards: list[dict], catalog: dict[str, dict]) -> tuple[list[str], list[str]]:
    """(openrouter routes whose price changed, other routes OpenRouter lists differently)."""
    by_norm: dict[str, list[dict]] = {}
    for m in catalog.values():
        if m["id"].split(":")[-1] in ("free", "batch") or m["id"].startswith("~"):
            continue
        by_norm.setdefault(norm(m["id"]), []).append(m)
    changed, differs = [], []
    for card in cards:
        for provider, route in card["routes"].items():
            cost = route.get("cost")
            if provider == "openrouter":
                mid = route["models"][0]
                if mid not in catalog:
                    changed.append(f"- `{card['id']}`: `{mid}` is no longer on OpenRouter")
                    continue
                fresh = endpoint_price(mid)
                time.sleep(0.5)
                if fresh is None:
                    changed.append(f"- `{card['id']}`: endpoint list for `{mid}` unreadable, price kept")
                elif fresh != cost:
                    changed.append(f"- `{card['id']}` via openrouter: {cost} -> {fresh} USD per M tokens")
            elif cost and provider in FIRST_PARTY_NAMES:
                vendor = FIRST_PARTY_NAMES[provider]
                for mid in route["models"]:
                    found = [m for m in by_norm.get(norm(mid), []) if m["id"].startswith(vendor + "/")]
                    if found:
                        new = price_of(found[0])
                        if (cost["input"], cost["output"]) != new:
                            differs.append(
                                f"- `{card['id']}` on {provider}: card {cost['input']} / {cost['output']}, "
                                f"OpenRouter lists {new[0]} / {new[1]} (`{found[0]['id']}`)"
                            )
                        break
    return changed, differs


def uncarded_open_models(cards: list[dict], catalog: dict[str, dict], today: dt.date) -> list[dict]:
    carded = {norm(mid) for c in cards for r in c["routes"].values() for mid in r["models"]}
    carded |= {norm(c["id"]) for c in cards}
    out = []
    for m in catalog.values():
        if not m.get("hugging_face_id") or m["id"].split(":")[-1] in ("free", "batch"):
            continue
        if norm(m["id"]) in carded:
            continue
        age = (today - dt.date.fromtimestamp(m["created"])).days
        params = m.get("supported_parameters", [])
        if age <= NEW_MODEL_DAYS and "tools" in params and "structured_outputs" in params:
            out.append(m)
    return sorted(out, key=lambda m: -m["created"])


def report(cards_path: Path, *, use_aa: bool = True) -> str:
    today = dt.date.today()
    data = load_cards(cards_path)
    cards = data["cards"]
    catalog = fetch_openrouter()
    lines = [f"# Model card sweep, {today.isoformat()}", "",
             f"Cards file last reviewed {data['updated']}; {len(cards)} cards; "
             f"OpenRouter lists {len(catalog)} models.", ""]

    changed, differs = route_drift(cards, catalog)
    lines += ["## OpenRouter route prices", "",
              *(changed or ["Every `openrouter` route matches the catalog."]), ""]
    lines += ["## Other routes OpenRouter prices differently", "",
              "A cross-check only: OpenRouter shows the cheapest host or a blended price "
              "for some models. Compare with the vendor's own price page before changing a card.", "",
              *(differs or ["None."]), ""]

    fresh = uncarded_open_models(cards, catalog, today)
    lines += [f"## New open models without a card (last {NEW_MODEL_DAYS} days)", ""]
    if not fresh:
        lines += ["None.", ""]
    else:
        lines += ["| OpenRouter id | released | context | USD in / out | image | ECI | AA page |",
                  "|---|---|---|---|---|---|---|"]
        slugs = _aa_slug_index() if use_aa else {}
        eci_rows = {norm(k): r for k, r in fetch_eci().items()} if use_aa else {}
        for m in fresh:
            price = price_of(m)
            page = slugs.get(norm(m["id"]))
            eci = eci_rows.get(norm(m["id"]))
            lines.append(
                f"| `{m['id']}` | {dt.date.fromtimestamp(m['created'])} | {m['context_length']} | "
                f"{price[0]} / {price[1]} | {'yes' if 'image' in m['architecture']['input_modalities'] else 'no'} | "
                f"{round(float(eci['eci']), 1) if eci else '-'} | "
                f"{('`' + page + '`') if page else '-'} |"
            )
        lines.append("")

    lines += rung_section(cards)
    return "\n".join(lines) + "\n"


def _aa_slug_index() -> dict[str, str]:
    """Artificial Analysis model slugs by normalised name, from its models page."""
    try:
        flight = _flight(_get("https://artificialanalysis.ai/models"))
    except (urllib.error.URLError, TimeoutError, ValueError):
        return {}
    slugs = re.findall(r'\{"slug":"([^"]+)","name":"[^"]+","deprecated":false,"releaseDate"', flight)
    return {norm(s): s for s in slugs}


def rung_section(cards: list[dict]) -> list[str]:
    """Cards whose recorded scores put them on another rung."""
    est = Estimator(cards)
    lines = ["## Rungs against the scores", "",
             "Every card is graded on up to three views and placed on the lower: the Epoch "
             "Capabilities Index (ECI), or for a card Epoch has not scored an ECI estimated from "
             "its other benchmarks"
             + (f" (leave-one-out error {est.rmse:.1f} ECI points over {est.n_checked} cards)" if est.rmse else "")
             + ", and the Artificial Analysis index. The rung thresholds are frozen (see "
             "`tools/sweep.py`); a card changes rung when its scores do. A suggestion prompts a look "
             "and a decision: it is never applied by the sweep. Edges: ECI "
             + ", ".join(f"`{r}` from {e:g}" for r, e in ECI_EDGES)
             + "; index v5.0.1 " + ", ".join(f"`{r}` from {e:g}" for r, e in AA_EDGES["5.0.1"]) + ".", ""]
    rows, soft, bare = [], [], []
    for card in cards:
        b = card.get("benchmarks") or {}
        e = None if ECI in b else est(card)
        g = grade(b, e)
        if g["suggested"] is None:
            bare.append(card["id"])
            continue
        if not settles(card["rung"], g):
            eci, idx = b.get(ECI), b.get(INDEX)
            eci_txt = (f"{eci['score']} [{eci['interval'][0]}, {eci['interval'][1]}]" if eci
                       else f"~{e['score']} [{e['lo']}, {e['hi']}]" if e else "-")
            views = ", ".join(f"{k} {v}" for k, v in g["views"].items())
            rows.append(f"| `{card['id']}` | `{card['rung']}` | {eci_txt} | {idx['score'] if idx else '-'} | "
                        f"**suggest `{g['suggested']}`** ({views}) |")
        elif not g["agree"] or g["borderline"]:
            soft.append(card["id"])
    lines += ["| card | rung | ECI [95%] (~ estimated) | AA index | |", "|---|---|---|---|---|",
              *(rows or ["| none | | | | |"]), ""]
    if soft:
        lines += ["Close calls, not counted as findings (the views differ, or an interval crosses an "
                  "edge): " + ", ".join(f"`{c}`" for c in soft) + ".", ""]
    if bare:
        lines += ["No benchmark recorded that speaks, so placed by size or judgement: "
                  + ", ".join(f"`{c}`" for c in bare) + ".", ""]
    return lines


# ── the chart ───────────────────────────────────────────────────────────────

_CHART_STYLE = """<style>
.bg{fill:#fff}.t{fill:#1f2328}.m{fill:#59636e}.g{stroke:#59636e;stroke-opacity:.25}.e{stroke:#59636e}
.z1{fill:#59636e;fill-opacity:.04}.z2{fill:#59636e;fill-opacity:.10}
.small{--c:#8EBCEB}.mid{--c:#3E8FDD}.large{--c:#1F6BC0}.frontier{--c:#0B4EA2}
.dot{fill:var(--c)}.ring{fill:none;stroke:var(--c);stroke-width:2}.bar{stroke:var(--c);stroke-opacity:.55}
.band{fill:var(--c);fill-opacity:.13}.rule{stroke:#59636e;stroke-opacity:.35}
.nano{--c:#B7C4D6}.floor{fill:#59636e;fill-opacity:.55}.estbar{stroke:var(--c);stroke-opacity:.55;stroke-dasharray:3 2}
@media (prefers-color-scheme:dark){.bg{fill:#0d1117}.t{fill:#e6edf3}.m{fill:#9198a1}.g{stroke:#9198a1;stroke-opacity:.25}.e{stroke:#9198a1}
.z1{fill:#9198a1;fill-opacity:.05}.z2{fill:#9198a1;fill-opacity:.12}
.nano{--c:#4A5A70}.small{--c:#3F78B8}.mid{--c:#5C9BE0}.large{--c:#8FBDF2}.frontier{--c:#C7E0FF}}
</style>"""


def render_chart(data: dict) -> str:
    """Every card as a row, grouped by rung, against both views its rung is read
    from: the Epoch Capabilities Index on the left (a dot with its 95% interval)
    and the Artificial Analysis index on the right (a dot), each inside the rung's
    band on that scale. A card missing from one view says so there, so
    nothing is left hanging: the cards Epoch has not scored yet sit on their
    Artificial Analysis dot. An orange ring marks a view that puts the card on a
    different rung from the one it carries (the lower view wins). Written by
    ``update`` and ``chart``, embedded in the README. Deterministic: the same
    cards give the same bytes, so a sweep's diff is the cards that moved."""
    est = Estimator(data["cards"])
    groups: dict[str, list] = {r: [] for r in RUNGS}
    for card in data["cards"]:
        b = card.get("benchmarks") or {}
        eci, idx = b.get(ECI), b.get(INDEX)
        guess = None if eci else est(card)
        low, high = (eci.get("interval") or (eci["score"], eci["score"])) if eci else (None, None)
        groups[card["rung"]].append({
            "id": card["id"], "open": bool(card.get("open_weight")),
            "eci": eci["score"] if eci else None, "lo": low, "hi": high,
            "aa": idx["score"] if idx else None, "aa_v": idx.get("version") if idx else None,
            "est": guess,
        })
    for rows in groups.values():
        rows.sort(key=lambda r: (r["eci"] is None, -(r["eci"] or (r["est"] or {}).get("score") or 0), r["id"]))
    n_all = sum(len(rows) for rows in groups.values())
    n_eci = sum(1 for rows in groups.values() for r in rows if r["eci"] is not None)
    n_aa = sum(1 for rows in groups.values() for r in rows if r["aa"] is not None)
    n_est = sum(1 for rows in groups.values() for r in rows if r["eci"] is None and r["est"])
    n_neither = sum(1 for rows in groups.values() for r in rows if r["eci"] is None and not r["est"] and r["aa"] is None)

    width, label_right = 900, 178
    e_l, e_r, a_l, a_r = 196, 566, 616, 860
    top, rh, hh = 54, 15, 24
    e_lo, e_hi, a_lo, a_hi = 100.0, 180.0, 0.0, 62.0
    order = [r for r in reversed(RUNGS) if groups[r]]
    plot_h = sum(hh + len(groups[r]) * rh + 6 for r in order)
    height = top + plot_h + 138

    def ex(v: float) -> float:
        return round(e_l + (v - e_lo) / (e_hi - e_lo) * (e_r - e_l), 1)

    def ax(v: float) -> float:
        return round(a_l + (v - a_lo) / (a_hi - a_lo) * (a_r - a_l), 1)

    e_edges = list(ECI_EDGES)
    a_edges = list(AA_EDGES["5.0.1"])
    e_bounds = {"nano": (e_lo, e_edges[0][1])}
    for i, (name, e) in enumerate(e_edges):
        e_bounds[name] = (e, e_edges[i + 1][1] if i + 1 < len(e_edges) else e_hi)
    # The AA index cannot tell nano from small, so the nano band spans both.
    a_bounds = {"nano": (a_lo, a_edges[1][1])}
    for i, (name, e) in enumerate(a_edges):
        a_bounds[name] = (e, a_edges[i + 1][1] if i + 1 < len(a_edges) else a_hi)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'role="img" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
        "<title>All model cards by rung, on the Epoch Capabilities Index and the Artificial Analysis index</title>",
        f"<desc>{n_all} model cards grouped by rung. {n_eci} have an Epoch Capabilities Index score (left, with its "
        f"95% interval) and {n_aa} have an Artificial Analysis index score (right), each shown inside the rung's band; "
        "cards without an Epoch score show an ECI estimated from their other benchmarks.</desc>",
        _CHART_STYLE,
        f'<rect class="bg" width="{width}" height="{height}"/>',
        f'<text class="t" x="{round((e_l + e_r) / 2, 1)}" y="{top - 32}" text-anchor="middle" font-size="12" font-weight="600">Epoch Capabilities Index</text>',
        f'<text class="t" x="{round((a_l + a_r) / 2, 1)}" y="{top - 32}" text-anchor="middle" font-size="12" font-weight="600">Artificial Analysis index (v5.0.1)</text>',
    ]
    for v in range(int(e_lo) + 10, int(e_hi), 10):
        out.append(f'<text class="m" x="{ex(v)}" y="{top - 8}" text-anchor="middle" font-size="11">{v}</text>')
    for v in range(10, int(a_hi), 10):
        out.append(f'<text class="m" x="{ax(v)}" y="{top - 8}" text-anchor="middle" font-size="11">{v}</text>')

    def disagrees(view: str | None, rung: str, from_aa: bool) -> bool:
        if view is None or view == rung:
            return False
        return not (from_aa and rung == "nano" and view == "small")  # the index cannot tell these apart

    y = float(top)
    for rung in order:
        rows = groups[rung]
        band_h = hh + len(rows) * rh + 6
        (el, er), (al, ar) = e_bounds[rung], a_bounds[rung]
        out.append(f'<g class="{rung}"><rect class="band" x="{ex(el)}" y="{y}" width="{round(ex(er) - ex(el), 1)}" height="{band_h}"/>')
        out.append(f'<rect class="band" x="{ax(al)}" y="{y}" width="{round(ax(ar) - ax(al), 1)}" height="{band_h}"/>')
        out.append(f'<line class="rule" x1="28" x2="{a_r}" y1="{y}" y2="{y}" stroke-width="0.5"/>')
        out.append(f'<text class="t" x="28" y="{y + 16}" font-size="13" font-weight="600">{rung}</text>')
        out.append(f'<text class="m" x="86" y="{y + 16}" font-size="11">{len(rows)} cards</text></g>')
        y += hh
        for r in rows:
            cy = round(y + rh / 2, 1)
            cid = r["id"]
            kind = "open weights" if r["open"] else "hosted only"
            if r["eci"] is not None:
                e_view = suggest_rung_eci(r["eci"])
                title = f"{cid}: ECI {r['eci']} [{r['lo']}, {r['hi']}], {e_view} on this view, rung {rung}, {kind}"
            elif r["est"]:
                e_view = suggest_rung_eci(r["est"]["score"])
                title = (f"{cid}: no Epoch score; ECI estimated at {r['est']['score']} [{r['est']['lo']}, {r['est']['hi']}] "
                         f"from {', '.join(r['est']['used'])}; rung {rung}, {kind}")
            else:
                e_view = None
                title = f"{cid}: no Epoch score yet, rung {rung}, {kind}"
            a_view = suggest_rung(r["aa"], r["aa_v"]) if r["aa"] is not None else None
            out.append(f'<g class="{rung}"><title>{title}' + (f"; AA index {r['aa']:g}" if r["aa"] is not None else "") + "</title>")
            out.append(f'<text class="t" x="{label_right}" y="{round(cy + 4, 1)}" text-anchor="end" font-size="11">{cid}</text>')
            if r["eci"] is not None:
                out.append(f'<line class="bar" x1="{ex(r["lo"])}" x2="{ex(r["hi"])}" y1="{cy}" y2="{cy}" stroke-width="2" stroke-linecap="round"/>')
                out.append(f'<circle class="{"dot" if r["open"] else "ring"}" cx="{ex(r["eci"])}" cy="{cy}" r="{4.5 if r["open"] else 3.8}"/>')
                if disagrees(e_view, rung, False):
                    out.append(f'<circle cx="{ex(r["eci"])}" cy="{cy}" r="7.5" fill="none" stroke="#eb6834" stroke-width="1.5"/>')
            elif r["est"]:
                g_est = r["est"]
                lo_r, hi_r = suggest_rung_eci(g_est["lo"]), suggest_rung_eci(g_est["hi"])
                inside = RUNGS.index(lo_r) <= RUNGS.index(rung) <= RUNGS.index(hi_r)
                x0, x1 = ex(max(g_est["lo"], e_lo)), ex(min(g_est["hi"], e_hi))
                out.append(f'<line class="estbar" x1="{x0}" x2="{x1}" y1="{cy}" y2="{cy}" stroke-width="2" stroke-linecap="round"/>')
                cx = ex(g_est["score"])
                out.append(f'<rect class="{"dot" if r["open"] else "ring"}" x="{round(cx - 3.8, 1)}" y="{round(cy - 3.8, 1)}" width="7.6" height="7.6" transform="rotate(45 {cx} {cy})"/>')
                if not inside:
                    out.append(f'<circle cx="{cx}" cy="{cy}" r="7.5" fill="none" stroke="#eb6834" stroke-width="1.5"/>')
            elif r["aa"] is None:
                out.append(f'<text class="m" x="{e_l + 8}" y="{round(cy + 4, 1)}" font-size="11">no benchmarks recorded</text>')
            else:
                out.append(f'<text class="m" x="{e_l + 8}" y="{round(cy + 4, 1)}" font-size="11">no Epoch score yet</text>')
            if r["aa"] is not None and r["aa_v"] in AA_EDGES:
                floor = a_view is None  # under the first edge: the index gives no view
                cls = "floor" if floor else ("dot" if r["open"] else "ring")
                out.append(f'<circle class="{cls}" cx="{ax(r["aa"])}" cy="{cy}" r="{3.6 if floor else (4.5 if r["open"] else 3.8)}"/>')
                if disagrees(a_view, rung, True):
                    out.append(f'<circle cx="{ax(r["aa"])}" cy="{cy}" r="7.5" fill="none" stroke="#eb6834" stroke-width="1.5"/>')
            elif r["aa"] is None:
                out.append(f'<text class="m" x="{a_l + 8}" y="{round(cy + 4, 1)}" font-size="11">no index score</text>')
            out.append("</g>")
            y += rh
        y += 6
    for _, e in e_edges:
        out.append(f'<line class="e" x1="{ex(e)}" x2="{ex(e)}" y1="{top}" y2="{round(y, 1)}" stroke-width="1" stroke-dasharray="4 3"/>')
        out.append(f'<text class="t" x="{ex(e)}" y="{round(y + 14, 1)}" text-anchor="middle" font-size="11">{e:g}</text>')
    for _, e in a_edges:
        out.append(f'<line class="e" x1="{ax(e)}" x2="{ax(e)}" y1="{top}" y2="{round(y, 1)}" stroke-width="1" stroke-dasharray="4 3"/>')
        out.append(f'<text class="t" x="{ax(e)}" y="{round(y + 14, 1)}" text-anchor="middle" font-size="11">{e:g}</text>')
    notes = [
        "Each card sits in its rung's band on both scales; dashed lines are the rung edges. A card is placed on the lower of its views.",
        "Dot: open weights. Ring: hosted only. Orange ring: this view puts the card on a different rung from the one it carries.",
        "Diamond with a dashed bar: no Epoch score, so the ECI is estimated from the card's other benchmarks (HLE, GPQA, IFBench and so on),",
        f"with a 95% interval of about plus or minus {1.96 * max(est.rmse or 0, 1.0):.0f} from leave-one-out error. Hover a row for the benchmarks used.",
        "Small grey dot: the index is at its floor (under 12), so it gives no view. The index cannot tell nano from small.",
        "ECI with 95% interval: Epoch AI (CC-BY 4.0). Index: Artificial Analysis. Grey words: that source has nothing for the card yet.",
        f"{n_eci} of {n_all} cards have an Epoch score, {n_est} an estimate from other benchmarks, {n_aa} an index score, "
        f"{n_neither} nothing recorded (placed by size or judgement). Cards updated {data['updated']}.",
    ]
    for i, line in enumerate(notes):
        out.append(f'<text class="m" x="28" y="{round(y + 36 + i * 14, 1)}" font-size="11">{line}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


_PROVIDER_LABELS = {
    "anthropic": "Anthropic", "gemini": "Google Gemini", "openai": "OpenAI", "mistral": "Mistral",
    "deepseek": "DeepSeek", "xai": "xAI", "groq": "Groq", "openrouter": "OpenRouter",
    "ollama": "Ollama (local)", "lmstudio": "LM Studio (local)",
}
TABLE_START, TABLE_END = "<!-- model-table:start -->", "<!-- model-table:end -->"


def render_table(data: dict) -> str:
    """Provider by rung: the model cards each provider serves, on the rung they
    are placed on, best first. A model served by several providers is in each
    row, so a column reads as the equivalent choices across providers. ``*`` marks
    a card Epoch has not scored (its rung comes from an estimate or the index)."""
    cards = data["cards"]
    est = Estimator(cards)
    score: dict[str, float] = {}
    star: set[str] = set()
    for card in cards:
        eci = (card.get("benchmarks") or {}).get(ECI)
        if eci:
            score[card["id"]] = eci["score"]
        else:
            star.add(card["id"])
            guess = est(card)
            if guess:
                score[card["id"]] = guess["score"]
    providers: list[str] = []
    for card in cards:
        for p in card["routes"]:
            if p not in providers:
                providers.append(p)
    known = [p for p in _PROVIDER_LABELS if p in providers]
    providers = known + sorted(p for p in providers if p not in _PROVIDER_LABELS)
    lines = ["| Provider | " + " | ".join(RUNGS) + " |", "|---|" + "---|" * len(RUNGS)]
    for p in providers:
        cells = []
        for rung in RUNGS:
            mine = [c for c in cards if p in c["routes"] and c["rung"] == rung]
            mine.sort(key=lambda c: (c["id"] not in score, -score.get(c["id"], 0), c["id"]))
            cells.append("<br>".join(f"`{c['id']}`" + ("\\*" if c["id"] in star else "") for c in mine) or "–")
        lines.append(f"| {_PROVIDER_LABELS.get(p, p)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_table(cards_path: Path, readme_path: Path) -> bool:
    """Regenerate the README's provider-by-rung table between its markers;
    ``True`` when the README changed."""
    text = readme_path.read_text(encoding="utf-8")
    a, b = text.index(TABLE_START) + len(TABLE_START), text.index(TABLE_END)
    new = text[:a] + "\n" + render_table(load_cards(cards_path)) + text[b:]
    if new == text:
        return False
    readme_path.write_text(new, encoding="utf-8", newline="\n")
    return True


def write_chart(cards_path: Path, chart_path: Path) -> bool:
    """Regenerate the chart from a card file; ``True`` when the file changed."""
    svg = render_chart(load_cards(cards_path))
    if chart_path.exists() and chart_path.read_text(encoding="utf-8") == svg:
        return False
    chart_path.write_text(svg, encoding="utf-8", newline="\n")
    return True


def write_outputs(cards_path: Path) -> list[str]:
    """Redraw the chart and the README table; the names of what changed."""
    changed = []
    if write_chart(cards_path, CHART_PATH):
        changed.append(str(CHART_PATH.relative_to(ROOT)))
    if write_table(cards_path, README_PATH):
        changed.append(str(README_PATH.relative_to(ROOT)))
    return changed


# ── updating the card file ──────────────────────────────────────────────────


def _cost_json(cost: dict | None) -> str:
    if cost is None:
        return "null"
    return "{" + ", ".join(f'"{k}": {json.dumps(v)}' for k, v in cost.items()) + "}"


def _route_json(route: dict) -> str:
    return '{"models": ' + json.dumps(route["models"], ensure_ascii=False) + ', "cost": ' + _cost_json(route.get("cost")) + "}"


def _inline(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def dump_cards(data: dict) -> str:
    """The card file in the layout it is kept in: one card per block, one route
    per line, one benchmark per line. Writing it back this way keeps a sweep's
    diff down to the lines it changed."""
    out = ["{", f'  "format": {data["format"]},', f'  "updated": {_inline(data["updated"])},', '  "cards": [']
    blocks = []
    for card in data["cards"]:
        lines = ["    {"]
        fields = []
        fields.append(f'      "id": {_inline(card["id"])}')
        fields.append(f'      "rung": {_inline(card["rung"])}')
        routes = ",\n".join(f'        {_inline(p)}: {_route_json(r)}' for p, r in card["routes"].items())
        fields.append('      "routes": {\n' + routes + "\n      }")
        fields.append(f'      "structured_output": {_inline(card["structured_output"])}')
        fields.append(f'      "context_window": {_inline(card["context_window"])}')
        if "features" in card:
            fields.append(f'      "features": {_inline(card["features"])}')
        if "open_weight" in card:
            fields.append(f'      "open_weight": {_inline(card["open_weight"])}')
        if card.get("benchmarks"):
            rows = ",\n".join(f'        {_inline(n)}: {_inline(b)}' for n, b in card["benchmarks"].items())
            fields.append('      "benchmarks": {\n' + rows + "\n      }")
        if "notes" in card:
            fields.append(f'      "notes": {_inline(card["notes"])}')
        lines.append(",\n".join(fields))
        lines.append("    }")
        blocks.append("\n".join(lines))
    out.append(",\n".join(blocks))
    out += ["  ]", "}"]
    return "\n".join(out) + "\n"


def update(cards_path: Path) -> list[str]:
    """Refresh ``openrouter`` prices and ``benchmarks``: the Artificial Analysis
    scores of cards that name a page there, and the Epoch ECI of cards Epoch
    has a row for. Returns what changed, one line each."""
    data = load_cards(cards_path)
    catalog = fetch_openrouter()
    eci_rows = fetch_eci()
    today = dt.date.today().isoformat()
    changes = []
    for card in data["cards"]:
        route = card["routes"].get("openrouter")
        if route:
            if route["models"][0] not in catalog:
                changes.append(f"{card['id']}: {route['models'][0]} no longer on OpenRouter, price kept")
            else:
                cost = endpoint_price(route["models"][0])
                time.sleep(0.5)
                if cost is not None and route.get("cost") != cost:
                    changes.append(f"{card['id']}: openrouter price {route.get('cost')} -> {cost}")
                    route["cost"] = cost
        old = card.get("benchmarks") or {}
        epoch_old = {k: v for k, v in old.items() if v["source"].startswith("epoch.ai/")}
        aa_old = {k: v for k, v in old.items() if v["source"].startswith("artificialanalysis.ai/")}
        other = {k: v for k, v in old.items() if k not in epoch_old and k not in aa_old}
        aa_new, aa_name = aa_old, None
        slug = aa_slug(card)
        if slug:
            aa = fetch_aa(slug)
            time.sleep(1)
            if aa is None:
                changes.append(f"{card['id']}: Artificial Analysis page {slug} unreadable, scores kept")
            else:
                aa_name = aa["record"].get("name")
                aa_new = benchmarks_from(aa, today)
        epoch_new = epoch_old
        row = eci_row_for(card, eci_rows, aa_name) if eci_rows else None
        if row is not None:
            epoch_new = {ECI: eci_entry(row, today)}
        elif epoch_old and eci_rows:
            changes.append(f"{card['id']}: no longer in Epoch's ECI data, score kept")

        def key(entries: dict):
            return {k: (v["score"], v.get("version"), tuple(v.get("interval") or ())) for k, v in entries.items()}

        if key(aa_new) != key(aa_old) or key(epoch_new) != key(epoch_old):
            changes.append(f"{card['id']}: benchmarks refreshed")
            card["benchmarks"] = {**epoch_new, **aa_new, **other}
            if not card["benchmarks"]:
                del card["benchmarks"]
    if changes:
        data["updated"] = today
        cards_path.write_text(dump_cards(data), encoding="utf-8", newline="\n")
    if cards_path == CARDS_PATH:
        for name in write_outputs(cards_path):
            changes.append(f"redrawn: {name}")
    return changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["report", "update", "chart"])
    ap.add_argument("--cards", type=Path, default=CARDS_PATH)
    ap.add_argument("--no-aa", action="store_true", help="skip Artificial Analysis (OpenRouter only)")
    ap.add_argument("--out", type=Path, help="report: write the report here instead of stdout")
    ap.add_argument("--findings", type=Path,
                    help="report: write how many items need a person's look here (0 when none)")
    args = ap.parse_args(argv)
    if args.command == "report":
        text = report(args.cards, use_aa=not args.no_aa)
        if args.out:
            args.out.write_text(text, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(text)
        if args.findings:
            # Every item the report raises is a bullet or a table row that opens
            # with a code-quoted name; headings, prose and table headers do not.
            count = sum(1 for line in text.splitlines() if line.startswith(("- `", "| `")))
            args.findings.write_text(f"{count}\n", encoding="utf-8")
    elif args.command == "chart":
        changed = write_outputs(args.cards)
        print("Redrawn: " + ", ".join(changed) if changed else "Chart and README table unchanged.")
    else:
        changes = update(args.cards)
        print("\n".join(changes) if changes else "Nothing changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
