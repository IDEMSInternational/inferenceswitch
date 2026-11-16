# Model cards

A model card says what one model is, so that `Client.select_model` can turn
"a `mid` model that does reliable structured output" into whichever model the
machine actually has. The decision behind this is
[#7](https://github.com/IDEMSInternational/inferenceswitch/issues/7).

Cards are a **starting guess** about equivalence, not evidence. A consumer's
own task evals decide whether a substitute works, and a pinned model id always
wins over selection. Placements below are provisional and open to review.

## Placement rubric

### Rungs

A rung is a **capability level**, defined by a fixed score threshold, not a
vendor's name for a tier. A card goes on the rung its measured scores put it on,
so that switching provider at one rung gives roughly the same results: a
provider whose "mid" model measures as `small` does not get to be `mid`, because
a tool built for `mid` would break on it.

| Rung | Epoch ECI from | What is on it (at calibration, 2026-11-16) | Unmeasured, open-weight starting guess |
|---|---|---|---|
| `nano` | below 130 | GPT-4o mini 123.9, Llama 3.3 70B 126.1, Llama 3.1 8B 111.1; unscored on-device models (about 4B and under): Gemma 4 E2B and E4B, Qwen3 4B | up to ~4B parameters |
| `small` | 130 | Gemini 3.5 Flash-Lite 145.6, GPT-5.4 nano 146.4, Qwen3.6 27B 146.7, gpt-oss-120b 140.2 | ~7–14B |
| `mid` | 152 | Haiku 5.5 160.4, GPT-5.6 Terra 160.7, Gemini 3.8 Flash 158.8, GPT-6 Luna 156.4, GLM-5.3 157.1 | ~30B and up, dense |
| `large` | 163 | GPT-6 Sol 163.8, Opus 5 164.7, Sonnet 5.5 166.6 | — |
| `frontier` | 168 | GPT-6 Astra 169.0, Opus 5.5 168.5 | — |

**The thresholds are frozen.** They were set once, on 2026-11-16, from the
reference models of that day, and are never moved when a new generation arrives:
raising a rung's bar would silently break every tool already pinned to it.
Models are re-scored and may change rung (the sweep reports it; a person moves
the card); if models outgrow the top rung, a rung is added above it rather than
the existing ones moving. A test fails if an edge in `tools/sweep.py` changes.

How to place a model:

1. **Measured models go where their scores put them** (below). This applies to
   hosted models and open-weight ones alike, and to every vendor: a vendor's own
   tier name (Haiku, Flash, Sonnet, Opus, "Luna") is not evidence. Haiku 4.5 is
   `small`, Opus 4.8 is `mid` and Mistral Large 4 is `mid` on their scores, and
   their cards say they were moved.
2. **Models Epoch has not scored** are placed on an ECI **estimated from their other
   benchmarks** (below), and on the Artificial Analysis index if it speaks. A card
   inside the estimate's wide band is not moved. Mark such a card in `notes` if
   its rung rests on little (a few benchmarks, or an estimate near an edge).
3. **Models with no benchmark that speaks** (none recorded, or all at their floor or
   ceiling) are placed by judgement until they are scored: open-weight ones from the
   parameter-count column, moved **down** a rung if they look clearly weaker (an
   older generation, say), hosted ones by what their vendor says the model is. Say
   in `notes` that the card is unmeasured; the sweep lists those cards. When
   unsure, place lower: under-claiming costs a consumer a more expensive model,
   over-claiming costs them a wrong answer. There will always be open models nobody
   scores; the rung is a best effort, not an exact figure.
4. **Mixture-of-experts models** are placed by their scores, or when unmeasured
   by quality using total parameters as the starting point; record
   `active_params_b` too.
5. Use `notes` for anything a reviewer should know: "candidate for large if
   evals support it", memory needed locally.

### Benchmarks as evidence

Parameter count is a starting point, not a measure: GLM-5.3 has 753B parameters
and scores like a `mid` model, and a dense 31B Gemma 4 scores below Haiku-class
work. The card therefore records published scores (`benchmarks`) and the rung is
checked against them.

**Never one benchmark.** Each benchmark is a sigmoid in capability: it separates
models only in its middle, sits at its floor for models below it and at its
ceiling for models above. GSM8K is saturated by every current model, GPQA
Diamond is above 90% for many of them, and Artificial Analysis's index is at
its floor (3 to 6) for anything that is not agentic. A model that tops one
benchmark still gets a fair score on a harder one, so a single benchmark
mis-places both ends. The card is graded on up to three views, each combining
many benchmarks:

- **Epoch Capabilities Index (ECI)**, from [Epoch AI](https://epoch.ai/benchmarks/eci)
  (CC-BY 4.0). One scale stitched from 50+ benchmarks of very different
  difficulty (GPQA Diamond, FrontierMath, SimpleQA, ARC-AGI, Terminal-Bench,
  OSWorld, ProofBench and others, fit jointly so that each benchmark's
  difficulty is estimated rather than assumed), with a 95% interval. It is the
  broad, general view, but covers a model only once Epoch has results for it,
  weeks after release, and from a different subset of benchmarks for each model.
- **Artificial Analysis Intelligence Index (v5.0.1)**, which averages nine
  evaluations: AA-Briefcase, AutomationBench-AA, Terminal-Bench 4.0,
  FrontierCode-AA 2.0, AA-Omniscience, GDP.pdf, HLE-Diamond, CritPt (hard
  subset) and Terminal-Bench-Science. It covers the newest models first, but is
  weighted to agentic, coding and professional-task work, where a rung is about
  general reasoning and instruction following. Where it is at its floor it gives
  no view. Artificial Analysis re-bases it between versions, so its edges are
  fixed per version (`AA_EDGES` in `tools/sweep.py`): a score from a version with
  no edges has no view until a person adds them.

- **An ECI estimated from the card's other benchmarks**, for a model Epoch has not
  scored. For each of GPQA Diamond, HLE, IFBench, tau2, AA-LCR, SciCode,
  Terminal-Bench 2.1 and LiveCodeBench, the sweep fits `ECI = a + b * logit(score)`
  by least squares on the cards that have both (a sigmoid in capability is a line in
  logit space), and uses a benchmark only where the card's score is between 3% and
  97%, so a floored or saturated one says nothing. A card's estimate is the average
  of the benchmarks it has, weighted by how well each tracks the ECI. Leaving each
  scored card out in turn, the estimate misses its real ECI by about 4 points (RMSE),
  so the band it carries is 1.96 times that, about plus or minus 7; a card inside the
  band is not moved. It is re-fit with each sweep, but the edges it is read against
  never move. The estimate is computed, not recorded: the benchmarks it comes from
  are on the card.

The ECI and Artificial Analysis index are recorded as `epoch_eci` and
`aa_intelligence_index`, next to the individual benchmarks Artificial Analysis
publishes (`hle`, `gpqa_diamond`, `ifbench` and others), which are used only through
the estimate above and are otherwise kept for a reviewer to read. **The lower of the
views places a card** (a view that is not published, or at its floor, has no say).
That is the under-claim rule. A card whose views differ, or whose interval crosses a
rung's edge, is a close call: the sweep lists it, and it may say so in its `notes`.
A card with only one view is placed on that one and picks up the others when they
appear. The README's chart shows, for every card, the evidence in its rung's band on
both scales (an estimate as a diamond with a dashed bar), and its table lists the
cards by provider and rung.

The edges (ECI `small` 130, `mid` 152, `large` 163, `frontier` 168; index
v5.0.1 `small` 12, `mid` 26, `large` 40, `frontier` 55) sit between the medians of
the reference models of calibration day, listed in `tools/sweep.py`. The `small`
edge at 130 splits a wide rung: the current `small` references sit at 145 to 157,
and Qwen3 8B (132) and GPT-4.1 mini (134) are `small`, while GPT-4o mini (124) and
Llama 3.3 70B (126) are `nano` whatever their size: a rung is capability, not
parameters. Only the ECI has a `nano` edge. The index cannot tell `nano` from
`small`, so a card with no Epoch score is `nano` by size (on-device, about 4B and
under) and a score that says `small` leaves it where it is; a card with an ECI
goes where the ECI puts it.

Selection never picks a model **below** the requested rung. By default it may
pick one above (`allow_higher=False` asks for the exact rung), preferring the
closest rung unless the preference says otherwise.

### Structured-output reliability

Graded for the path **inferenceswitch** uses with that provider, not for the
model in the abstract:

| Grade | Meaning | Typical case |
|---|---|---|
| `reliable` | The provider enforces the schema (constrained decoding or a strict schema) | Anthropic structured outputs, Gemini `response_schema`, OpenAI strict `json_schema` |
| `best_effort` | Not enforced: JSON mode plus the schema in the prompt, or a non-strict `json_schema`; expect occasional invalid output on long or nested schemas | Local servers, Mistral, DeepSeek, Groq |
| `none` | No usable structured output | Base models, very small models |

OpenAI's `reliable` holds for schemas strict mode can express. inferenceswitch
translates the schema to the strict subset (every field required, optional ones
nullable, closed objects), and a schema it cannot express (a free-form object,
a multi-entry `allOf`) is sent non-strict instead. If inferenceswitch starts
enforcing schemas on another provider, that provider's cards move up a grade in
the same change. A test fails if a card is graded `reliable` on a provider whose
mechanism inferenceswitch does not enforce.

### Other fields

- **`context_window`**: the model's maximum context. For local servers this is
  the model's native window, which the server may be configured below (Ollama's
  default context is much smaller than most models' maximum).
- **`cost`**: list price in USD per million input and output tokens, standard
  tier, no batch discount. `null` for local servers (free per token)
  and when unknown; unknown hosted prices rank last under `cheapest`.
  Optionally `cache_read` and `cache_write`: the price of prompt tokens read
  from and written to the provider's cache (Anthropic: the 5-minute write,
  1.25x input). Leave them out when the vendor's price is not known;
  `Cost.usd(usage)` then leaves those tokens out of a call's cost and names
  them, rather than pricing them as zero. `Usage.input_tokens` is the uncached
  part of the prompt on every provider, so each count has one price.
- **`features`**: `add` and `remove` lists of `Capability` values, applied to
  the provider's defaults. Use them only where the model really differs from
  its provider (Haiku 4.5 accepts `temperature`; the Anthropic default does
  not claim it).
- **`benchmarks`**: published scores for the model, an object keyed by
  benchmark name, each `{"score", "source", "date"}` and optionally `version`,
  `setting` and `interval`. A score is on the benchmark's own scale (a percentage
  for most; the ECI is an index that is not bounded by 100). `interval` is the
  source's 95% interval `[low, high]` where it publishes one (the ECI does).
  `source` is
  who measured it (`artificialanalysis.ai/models/<slug>` for a page there, or a
  vendor's name for a score from its own model card, which is not comparable
  with an independent one); `date` is when it was read; `version` is the index
  or benchmark version, because an index is re-based when its parts change and
  scores from different versions do not compare; `setting` is how the model
  was run (`max` effort). They are evidence for `rung`; selection does not read
  them. The sweep records `epoch_eci` (source
  `epoch.ai/data/eci_scores.csv#<model>`) where Epoch has a row for the model,
  and `aa_intelligence_index`, `gpqa_diamond`, `hle`, `scicode`, `aa_lcr`,
  `ifbench`, `tau2`, `terminal_bench_2_1` and `livecodebench` where Artificial
  Analysis publishes them, from one page there. Epoch AI's data is used under
  CC-BY 4.0.
- **`open_weight`**: `family`, `params_b`, `quantization` and optionally
  `active_params_b`. A different quantization of the same weights is a separate
  card (e.g. `qwen3-8b-q8`), since it can change both quality and memory.

## File format

```json
{
  "format": 1,
  "updated": "2026-10-11",
  "cards": [
    {
      "id": "qwen3-8b",
      "rung": "small",
      "routes": {
        "ollama":   {"models": ["qwen3:8b", "qwen3:latest"], "cost": null},
        "lmstudio": {"models": ["qwen/qwen3-8b"], "cost": null}
      },
      "structured_output": "best_effort",
      "context_window": 32768,
      "features": {"add": [], "remove": []},
      "open_weight": {"family": "qwen3", "params_b": 8, "quantization": "Q4_K_M"},
      "benchmarks": {
        "epoch_eci": {"score": 157.1, "source": "epoch.ai/data/eci_scores.csv#GLM-5.3", "date": "2026-11-16", "interval": [154.9, 159.5]},
        "aa_intelligence_index": {"score": 31.7, "source": "artificialanalysis.ai/models/glm-5-3", "date": "2026-11-16", "version": "5.0.1", "setting": "max"}
      },
      "notes": "Roughly a rung below Haiku on general reasoning."
    }
  ]
}
```

- An `openrouter` route is priced at the **median host that supports
  structured output**, by 3:1 blended price. OpenRouter's catalog headline is
  the cheapest host, which for an open model is often an outlier (Kimi K3.1 Pro
  lists at $0.24 input against its maker's $2). `Cost` is a list price and an
  OpenRouter call may be routed to a cheaper or dearer host. Its `vision` flag
  comes from OpenRouter's declared input modalities, not from a call against a
  server.
- One card per **model**, with one **route** per provider that serves it. The
  same weights under Ollama and LM Studio are one card with two routes, so rung
  and capabilities are written once. Locality comes from the provider's
  registry entry (`is_local`), price from the route.
- `models` lists the ids a provider accepts for the model, preferred first.
  For a local server the first id it lists is used, so put alternative tags
  there (`qwen3:latest`).
- Unknown fields are rejected, so a typo fails loudly. A file with a different
  `format` is rejected rather than half-read.
- A route naming a provider that is not in the client's registry is ignored
  (shown as excluded). To serve a card from your own server (vLLM, a gateway),
  add a `ProviderSpec` for it and a route on the card.

Consumers add or override cards with a file in this format
(`default_cards().merged(ModelCards.load(path))`) or with `ModelCard` objects
in code; a card with the same `id` replaces the shipped one.

## Keeping cards current

- A model added to a provider's `known_models` gets a card in the same change.
  A test enforces this.
- Review prices and placements when a vendor releases or reprices a model, and
  at least before each release. Bump `updated` when you do.
- **A weekly sweep does the looking up.** `.github/workflows/sweep.yml` runs
  `tools/sweep.py` (standard library only, no keys, no model calls) every
  Monday. It reads OpenRouter's public catalog, Epoch AI's ECI data and
  Artificial Analysis's model pages, and
  - **rewrites** the `openrouter` route prices and the `benchmarks` (ECI, and
    Artificial Analysis for cards that name a page there), **redraws the chart**
    the README embeds (`docs/model-cards-chart.svg`) **and the README's
    provider-by-rung table**, from the same data (a test fails if either is
    stale), then opens a pull request if the file
    changed (after the test suite passes on it);
  - **reports**, in one standing issue refreshed each week, what only a person
    can decide: open models from the last 120 days with no card yet, other
    providers' prices that OpenRouter lists differently (a cross-check, since
    OpenRouter can show a blended or cheapest-host price where a vendor's own
    list says otherwise: compare with the vendor's page before changing a
    card), and open-weight cards that sit off their index band.

  It never changes a rung, a note, a feature flag or another provider's price.
  Run it by hand with `python tools/sweep.py report` or `python tools/sweep.py
  update`. A new card starts from the report's "new open models" table: add the
  card with its first `benchmarks` entry (the `source` is what ties it to its
  Artificial Analysis page for later sweeps), place it by the bands above, and
  the sweep keeps it fresh.
- When a provider shuts a model down or closes it to new users, delete its
  card and add it to `RETIRED_MODELS` in `tests/test_model_cards.py`. A card
  is an offer `select_model` makes to every caller, so a model only some
  accounts can still call does not get one. Callers who still have access can
  merge the card back in themselves.
- Cards do not warn when they are old. Every `Selection` records
  `cards_updated`, so a consumer's run log shows which card data chose the
  model, and the consumer can decide what is too old for them.
