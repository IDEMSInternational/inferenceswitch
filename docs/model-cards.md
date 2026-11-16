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

Each rung is anchored on reference models. A model goes on the rung of the
reference model it most nearly substitutes for on **general reasoning and
instruction following**: not coding, not one benchmark, and not price.

| Rung | Anchors | Open-weight starting point |
|---|---|---|
| `nano` | On-device models | up to ~4B parameters |
| `small` | Gemini Flash-Lite, GPT nano / 5.6 Luna | ~7–14B |
| `mid` | Claude Haiku, Gemini Flash, GPT mini / 5.6 Terra / 6 Luna | ~30B and up, dense |
| `large` | Claude Sonnet, Gemini Pro, GPT Sol, Mistral Large | — |
| `frontier` | Claude Opus and Fable, GPT-6 Astra | — |

How to place a model:

1. **Hosted models from a vendor with a tiered line** go on the rung of their
   tier (Haiku and Flash on `mid`, Sonnet and Pro on `large`, Opus on
   `frontier`). Where a vendor's naming shifts between generations, follow
   its own description of the tier: GPT-6 Luna is priced and described like
   Haiku, so `mid`, while GPT-5.6 Luna is the old "nano" tier, so `small`. Move a model off its tier's rung only with evidence, and say
   why in `notes`.
2. **Open-weight models** start from the parameter-count column, then move
   **down** a rung if they are clearly weaker than that rung's anchors (for
   example, an older generation). Move one up only with eval evidence. When
   unsure, place lower: under-claiming costs a consumer a more expensive model;
   over-claiming costs them a wrong answer.
3. **Mixture-of-experts models** are placed by quality, using total
   parameters as the starting point; record `active_params_b` too.
4. Use `notes` for anything a reviewer should know: "lower end of mid",
   "candidate for large if evals support it", memory needed locally.

### Benchmarks as evidence

Parameter count is a starting point, not a measure: GLM-5.3 has 753B parameters
and scores like a `mid` model, and a dense 31B Gemma 4 scores below Haiku-class
work. For open-weight models the card therefore records published scores
(`benchmarks`), and the rung is checked against the **Artificial Analysis
Intelligence Index** (v5), an independent score that is re-run by the evaluator
rather than taken from the vendor. Read off the reference models:

| Index | Rung | Reference models on it |
|---|---|---|
| under 12 | no suggestion | gpt-oss-120b 3.8, Mistral Small 4 4.3: the index cannot tell them apart; parameters and HLE decide |
| 12 to 26 | `small` | Gemini 3.5 Flash-Lite 11.7, GPT-5.6 Luna 23.8, Qwen3.8 27B 20.1 |
| 26 to 40 | `mid` | GPT-6 Luna 27.9, Gemini 3.8 Flash 30.8, GPT-5.6 Terra 32.3, Haiku 5.5 36.5 |
| 40 to 55 | `large` | GPT-6 Sol 44.2, Kimi K3.1 Pro 46.5, Sonnet 5.5 54.5 |
| 55 and up | `frontier` | GPT-6 Astra 56.9, Opus 5.5 59.1 |

The bands are for **open-weight** cards only. A vendor's own tiers overlap on
them (Opus 5 scores 44.2 and is `frontier`; Mistral Large 4 scores 26.1 and is
`large`), and rule 1 above places those by tier. Where an open-weight model has
no index (Artificial Analysis stops re-running older models) place it by HLE and
size against the same references, and say so in `notes`. The bands only prompt a
second look: `tools/sweep.py` lists a card that sits off its band, and a card
that stays off it is written down in `OFF_BAND` there with the reason.

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
  benchmark name, each `{"score", "source", "date"}` and optionally `version`
  and `setting`. Scores are on the benchmark's own 0 to 100 scale. `source` is
  who measured it (`artificialanalysis.ai/models/<slug>` for a page there, or a
  vendor's name for a score from its own model card, which is not comparable
  with an independent one); `date` is when it was read; `version` is the index
  or benchmark version, because an index is re-based when its parts change and
  scores from different versions do not compare; `setting` is how the model
  was run (`max` effort). They are evidence for `rung`; selection does not read
  them. The sweep records `aa_intelligence_index`, `gpqa_diamond`, `hle`,
  `scicode`, `aa_lcr`, `ifbench`, `tau2`, `terminal_bench_2_1` and
  `livecodebench` where Artificial Analysis publishes them, from one page per
  card.
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
  Monday. It reads OpenRouter's public catalog and Artificial Analysis's model
  pages, and
  - **rewrites** the `openrouter` route prices and the `benchmarks` of cards
    that name an Artificial Analysis page, then opens a pull request if the file
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
