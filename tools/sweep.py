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
* **Artificial Analysis** (``https://artificialanalysis.ai/models/<slug>``): an
  independent Intelligence Index and benchmark scores, read from the data the
  model's page embeds. A page that changes shape is reported as unreadable; it
  never fails the run.

Commands::

    python tools/sweep.py report              # markdown report on stdout
    python tools/sweep.py update              # rewrite openrouter prices and benchmarks in place
    python tools/sweep.py report --cards PATH # a card file other than the shipped one
    python tools/sweep.py report --out report.md --findings count.txt

``update`` changes only what a machine can know: ``openrouter`` route prices
and the ``benchmarks`` of cards that already name an Artificial Analysis page
in their ``source``. Rungs, notes, feature flags and other providers' prices
stay a person's call; the report suggests, a reviewer decides.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS_PATH = ROOT / "inferenceswitch" / "model_cards.json"

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


#: Where each rung starts on the Artificial Analysis Intelligence Index (v5), read
#: off the reference models in ``docs/model-cards.md``: Gemini 3.5 Flash-Lite
#: 11.7 and GPT-5.6 Luna 23.8 (small); GPT-6 Luna 27.9, Gemini 3.8 Flash 30.8,
#: GPT-5.6 Terra 32.3 and Haiku 5.5 36.5 (mid); GPT-6 Sol 44.2 and Sonnet 5.5
#: 54.5 (large); GPT-6 Astra 56.9 and Opus 5.5 59.1 (frontier). A vendor's tiers
#: overlap on it (Opus 5 scores 44.2), so it places open-weight models only.
#: Below the first edge the index cannot tell models apart (gpt-oss-120b 3.8,
#: Gemma 4 31B 4.4), so there is no suggestion and parameters decide.
RUNG_EDGES = (("small", 12.0), ("mid", 26.0), ("large", 40.0), ("frontier", 55.0))


#: Open-weight cards whose rung is not the one the bands suggest, and why. The
#: report leaves them out of its suggestions, and a test fails for any other
#: card that sits off its band, so a new exception has to be written here.
OFF_BAND = {
    "mistral-large-4": "placed by its vendor's tier (Mistral Large is a large anchor), rule 1",
    "qwen3.8-27b": "the index suggests small (20.1 against GPT-5.6 Luna's 23.8); "
                   "left on mid for the team to move, see #29",
}


def suggest_rung(score: float) -> str | None:
    """The rung ``RUNG_EDGES`` puts an index score on; ``None`` below the first."""
    rung = None
    for name, edge in RUNG_EDGES:
        if score >= edge:
            rung = name
    return rung


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
        lines += ["| OpenRouter id | released | context | USD in / out | image | AA page |", "|---|---|---|---|---|---|"]
        slugs = _aa_slug_index() if use_aa else {}
        for m in fresh:
            price = price_of(m)
            page = slugs.get(norm(m["id"]))
            lines.append(
                f"| `{m['id']}` | {dt.date.fromtimestamp(m['created'])} | {m['context_length']} | "
                f"{price[0]} / {price[1]} | {'yes' if 'image' in m['architecture']['input_modalities'] else 'no'} | "
                f"{('`' + page + '`') if page else '-'} |"
            )
        lines.append("")

    if use_aa:
        lines += aa_section(cards, today)
    return "\n".join(lines) + "\n"


def _aa_slug_index() -> dict[str, str]:
    """Artificial Analysis model slugs by normalised name, from its models page."""
    try:
        flight = _flight(_get("https://artificialanalysis.ai/models"))
    except (urllib.error.URLError, TimeoutError, ValueError):
        return {}
    slugs = re.findall(r'\{"slug":"([^"]+)","name":"[^"]+","deprecated":false,"releaseDate"', flight)
    return {norm(s): s for s in slugs}


def aa_section(cards: list[dict], today: dt.date) -> list[str]:
    lines = ["## Benchmarks and rungs (Artificial Analysis)", ""]
    lines += ["Rung suggestions for open-weight cards use these starting points on the index: "
              + ", ".join(f"`{r}` from {e}" for r, e in RUNG_EDGES)
              + ". A suggestion is a prompt to look again, not a decision.", ""]
    rows, unreadable = [], []
    for card in cards:
        slug = aa_slug(card)
        if not slug:
            continue
        aa = fetch_aa(slug)
        time.sleep(1)  # be polite: one page a second
        if aa is None:
            unreadable.append(card["id"])
            continue
        fresh = benchmarks_from(aa, today.isoformat()).get(INDEX)
        old = (card.get("benchmarks") or {}).get(INDEX)
        if fresh is None:
            continue
        moved = old is None or abs(fresh["score"] - old["score"]) >= 0.5 or fresh.get("version") != old.get("version")
        suggestion = suggest_rung(fresh["score"]) if card.get("open_weight") else None
        if card["id"] in OFF_BAND:
            suggestion = None
        flag = "" if suggestion in (None, card["rung"]) else f" **suggest `{suggestion}`**"
        if moved or flag:
            was = f"{old['score']} (v{old.get('version')})" if old else "-"
            rows.append(f"| `{card['id']}` | `{card['rung']}`{flag} | {was} | {fresh['score']} (v{fresh.get('version')}) |")
    lines += ["| card | rung | recorded index | now |", "|---|---|---|---|", *(rows or ["| none changed | | | |"]), ""]
    if unreadable:
        lines += ["Pages that could not be read (the site may have changed): "
                  + ", ".join(f"`{c}`" for c in unreadable) + ".", ""]
    return lines


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
    """Refresh ``openrouter`` prices and the ``benchmarks`` of cards that name an
    Artificial Analysis page. Returns what changed, one line each."""
    data = load_cards(cards_path)
    catalog = fetch_openrouter()
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
        slug = aa_slug(card)
        if slug:
            aa = fetch_aa(slug)
            time.sleep(1)
            if aa is None:
                changes.append(f"{card['id']}: Artificial Analysis page {slug} unreadable, scores kept")
                continue
            fresh = benchmarks_from(aa, today)
            old = card.get("benchmarks") or {}
            if {k: (v["score"], v.get("version")) for k, v in fresh.items()} != \
                    {k: (v["score"], v.get("version")) for k, v in old.items()}:
                changes.append(f"{card['id']}: benchmarks refreshed")
                card["benchmarks"] = fresh
    if changes:
        data["updated"] = today
        cards_path.write_text(dump_cards(data), encoding="utf-8", newline="\n")
    return changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["report", "update"])
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
    else:
        changes = update(args.cards)
        print("\n".join(changes) if changes else "Nothing changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
