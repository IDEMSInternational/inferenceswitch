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
  tier, no caching or batch discount. `null` for local servers (free per token)
  and when unknown; unknown hosted prices rank last under `cheapest`.
- **`features`**: `add` and `remove` lists of `Capability` values, applied to
  the provider's defaults. Use them only where the model really differs from
  its provider (Haiku 4.5 accepts `temperature`; the Anthropic default does
  not claim it).
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
      "notes": "Roughly a rung below Haiku on general reasoning."
    }
  ]
}
```

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
- Cards do not warn when they are old. Every `Selection` records
  `cards_updated`, so a consumer's run log shows which card data chose the
  model, and the consumer can decide what is too old for them.
