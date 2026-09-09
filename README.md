# inferenceswitch

A lightweight, **import-only** multi-provider LLM dispatch library with
first-class structured output. Think liteLLM/OpenRouter, minus the proxy server,
the dynamic provider loading, and the commission — just a static registry and a
thick client so you stop re-writing `match provider` blocks across projects.

## Why it exists

Several of our projects were each re-implementing the same provider dispatch: map a
model name to a provider, build the right SDK client, and — the hard part —
force schema-valid JSON in each provider's own way. `inferenceswitch` owns that once.

It is deliberately small. No network proxy, no plugin system, no
caller-injected base URLs, and no logging at all — so keys cannot be logged.
Because this is a library you import rather than a service you run, there is no
server, no dynamic provider loading, and no deployment of ours to attack: the
runtime surface is the provider SDKs you already trust.

## Install

Provider SDKs are optional extras — install only what you use:

```
pip install "inferenceswitch[anthropic]"   # Anthropic (native)
pip install "inferenceswitch[gemini]"      # Google Gemini (native)
pip install "inferenceswitch[openai]"      # OpenAI + Mistral/DeepSeek/Groq/Ollama/LM Studio
pip install "inferenceswitch[all]"
```

## Usage

```python
from inferenceswitch import LLMClient

client = LLMClient()

# Structured output — same call regardless of provider mechanics.
data = client.generate_structured_json(
    model="claude-opus-4-8",             # routes to anthropic by prefix
    schema=MyPydanticModel.model_json_schema(),
    prompt="...",
    system="...",
)

# Plain text.
text = client.generate_text(model="gemini-3.5-flash", prompt="Summarize ...")

# Key smoke-test (cheapest authenticated call, never raises).
ok, detail = client.verify_key("anthropic")
```

### Selecting a provider

| Form | Example | When |
|---|---|---|
| Prefix routing | `model="claude-opus-4-8"` | Native providers with unambiguous prefixes |
| `provider/model` | `model="groq/llama-3.3-70b-versatile"` | OpenAI-compatible / ambiguous model names |
| Explicit `provider=` | `provider="ollama", model="qwen2.5-coder"` | Local servers, or overriding routing |

Unroutable or ambiguous model strings raise `ProviderResolutionError` — there is
no silent default provider.

## Providers

| Provider | Kind | Key env | Notes |
|---|---|---|---|
| `anthropic` | native | `ANTHROPIC_API_KEY` | structured output via forced tool use |
| `gemini` | native | `GEMINI_API_KEY` | structured output via `response_schema` + dict-dialect translation |
| `openai` | openai-compat | `OPENAI_API_KEY` | |
| `mistral` | openai-compat | `MISTRAL_API_KEY` | |
| `deepseek` | openai-compat | `DEEPSEEK_API_KEY` | |
| `groq` | openai-compat | `GROQ_API_KEY` | select explicitly (`groq/...`) |
| `ollama` | openai-compat | — | local `http://localhost:11434/v1`, best-effort JSON |
| `lmstudio` | openai-compat | — | local `http://localhost:1234/v1`, best-effort JSON |

## Tool calling & the "pick a model for this call" flow

`chat` is the normalized, optionally tool-using turn. You write one shape;
each adapter maps it into that provider's native mechanism (OpenAI `tool_calls`,
Anthropic `tool_use`/`tool_result`, Gemini `functionCall`/`functionResponse`) and
normalizes the reply (`text`, `tool_calls`, `stop_reason`, `usage`).

The full flow you'd build a UI/config around — *declare what the call needs →
list models that can serve it → user picks one → run it*:

```python
from inferenceswitch import LLMClient, Capability, Tool, user

client = LLMClient()

# 1. Which models can serve a tool-using call?
choices = client.models_for(Capability.TOOL_CALLING)
#    -> [ModelChoice("anthropic/claude-opus-4-8"), ..., ModelChoice("groq"), ...]
#    Curated models come back as routable `provider/model` ids. Providers with no
#    curated catalog (Groq, Ollama, LM Studio, your own entries) appear as a bare
#    provider name with `.needs_model` — they qualify, you supply the model.

# 2. User selects one — from a dropdown, .env, hard-coded, wherever.
chosen = pick_one(choices)                              # your UI / config
if chosen.needs_model:                                  # bare provider picked
    chosen = f"{chosen.provider}/{ask_for_model_name()}"

# 3. Run it. No provider switch/case — the adapter for `chosen` handles it.
weather = Tool("get_weather", "Current weather for a city",
               {"type": "object", "properties": {"city": {"type": "string"}},
                "required": ["city"]})

resp = client.chat(model=chosen, messages=[user("Weather in Paris?")], tools=[weather])
for call in resp.tool_calls:      # normalized across providers
    ...                           # execute call.name(call.input), then send results back
```

Or let the library drive the whole loop — you supply `{tool_name: fn}`:

```python
def get_weather(inp): return f"{inp['city']}: 20C, clear"

final = client.run_tools(
    model=chosen,
    messages=[user("Weather in Paris?")],
    tools=[weather],
    handlers={"get_weather": get_weather},
)
print(final.text)
```

Passing `tools` to a provider without `Capability.TOOL_CALLING` raises
`UnsupportedCapabilityError` — it fails loudly instead of silently dropping tools.

## Reasoning effort (one level, three native knobs)

Pass a normalized `effort` to `chat` and each provider maps it to its own
mechanism — Anthropic adaptive thinking + `output_config.effort`, OpenAI
`reasoning_effort`, Gemini thinking-token budget. No provider switch/case:

```python
from inferenceswitch import LLMClient, Effort, Capability, user

client = LLMClient()

# Discover which models let you control effort, then run with it:
for model in client.models_for(Capability.REASONING_EFFORT):
    if model.needs_model:      # a provider with no curated catalog — name a model
        continue               # yourself, or skip it as here
    resp = client.chat(model=model, messages=[user("Prove ...")], effort=Effort.HIGH)
```

`Effort` is `NONE | LOW | MEDIUM | HIGH | MAX`. `NONE` disables reasoning where
the provider allows it; `MAX` maps to each provider's ceiling (Anthropic `max`,
Gemini dynamic budget, OpenAI `high`). Requesting `effort` from a provider
without `Capability.REASONING_EFFORT` (Groq, Mistral, DeepSeek, locals) raises
`UnsupportedCapabilityError`. DeepSeek reasoning is reached by selecting the
`deepseek-reasoner` *model*, not an effort level, so it isn't an effort knob.

> Effort support is per-**model** for some providers (e.g. Anthropic Haiku 4.5
> doesn't take it). The capability is modeled at the provider level; a request
> with `effort` on a model that lacks it surfaces that provider's own 400.

## Output caps (Anthropic and Gemini)

Anthropic requires `max_tokens` on every request, so inferenceswitch has to choose one
when you don't. It sends **that model's own maximum** — 128000 on Sonnet/Opus,
64000 on Haiku 4.5 — rather than a single conservative number: output is billed
per token *generated*, not per token requested, so a lower default only buys you
silent truncation (`StopReason.MAX_TOKENS`) on long answers.

Three levels of control, narrowest first:

```python
from inferenceswitch import CLAUDE_MAX_OUTPUT_TOKENS, claude_max_output_tokens

# 1. Per call — wins over everything, and is never clamped to the model max.
client.chat(model="claude-opus-4-8", messages=[...], max_tokens=4000)

# 2. Process-wide — the table is the control surface. Lower a row for a house
#    ceiling, or add one for a model inferenceswitch doesn't know yet.
CLAUDE_MAX_OUTPUT_TOKENS["claude-opus-4-8"] = 32_000
CLAUDE_MAX_OUTPUT_TOKENS["claude-newthing-9"] = 128_000

# 3. Read it — what would this model default to?
claude_max_output_tokens("claude-haiku-4-5-20251001")   # 64000
```

Dated snapshots and provider-prefixed IDs (`anthropic.claude-opus-4-8`) resolve
to their base model's ceiling. A model matching no row falls back to 64000 — the
smallest ceiling in the current Claude line, and the largest value that is safe
to send blind; older models cap lower, so add a row before using one.

> Because the default is this large, the Anthropic adapter **streams** every
> request: the SDK refuses a non-streaming call whose `max_tokens` could outrun
> its 10-minute timeout. `get_final_message()` returns the same object, so this
> is invisible to callers.

**Gemini works the same way**, through `GEMINI_MAX_OUTPUT_TOKENS`,
`gemini_max_output_tokens()` and the same three levels of control. Gemini treats
`max_output_tokens` as *optional*, but omitting it is not the safe-looking thing
it appears to be: it hands the ceiling to an undocumented provider default that
can move under you, and leaves the same call returning far less from Gemini than
from Claude. The whole current line — Gemini 3.x and 2.5, pro/flash/flash-lite —
caps at 65536, so that is both every row in the table and the unknown-model
fallback. Retired pre-2.5 models capped at 8192; add an explicit row if you are
pinned to one.

```python
from inferenceswitch import GEMINI_MAX_OUTPUT_TOKENS, gemini_max_output_tokens

gemini_max_output_tokens("gemini-3-flash-preview")   # 65536
GEMINI_MAX_OUTPUT_TOKENS["gemini-3.5-flash"] = 16_000
```

When a Gemini response *is* cut at the cap, `generate_structured_json` raises
`StructuredOutputError` saying so — naming the ceiling in effect, the model's
maximum, and the prompt size — instead of letting `json.loads` blame the schema
with a character offset into a document that was simply unfinished.
`generate_text` returns the partial string (truncated prose is still usable);
`chat` reports `StopReason.MAX_TOKENS` on the response.

The remaining providers treat `max_tokens` as optional — inferenceswitch omits it when
you don't pass one, leaving each provider on its own default.

## Capability introspection (for model-specific workflows)

Each provider declares a feature set (`Capability` tokens) describing what it can
do — tool calling, vision, explicit prompt caching, reasoning effort, batch, etc.
Discover and gate on them so model-specific code fails loudly instead of 400-ing
deep in an SDK:

```python
from inferenceswitch import LLMClient, Capability

client = LLMClient()

if client.supports(Capability.EXPLICIT_PROMPT_CACHING, model="claude-opus-4-8"):
    ...  # this model takes inline cache breakpoints (see Prompt caching below)

# Or assert — raises UnsupportedCapabilityError with the list of what IS supported:
client.require(Capability.VISION, provider="gemini")

client.capabilities("anthropic").features   # the full frozenset
```

A feature flag means *the provider* supports it, independent of whether
inferenceswitch wraps it with a first-class method yet. For anything not yet wrapped,
use the **escape hatch** — the native SDK client, with routing and keys still
handled here:

```python
raw = client.raw_client(model="claude-opus-4-8")   # an anthropic.Anthropic()
raw.messages.create(..., extra_headers={...})       # provider-specific call
```

> Feature sets are populated **fail-closed**: a token is present only where
> support is reliable across the provider, so `require()` never green-lights a
> request that would error. (E.g. Anthropic omits `SAMPLING_PARAMS` because newer
> models reject `temperature` — support there is per-model, not per-provider.)

## Prompt caching

Two *different* contracts, kept as separate capabilities because they have
different mechanisms and different providers:

- **`EXPLICIT_PROMPT_CACHING`** — inline, caller-placed cache breakpoints. You
  mark a content block or tool with `cache=True`; the adapter emits Anthropic
  `cache_control`. Only **Anthropic** advertises this (its native mechanism).
- **`IMPLICIT_PROMPT_CACHING`** — automatic prefix caching you don't control.
  **OpenAI, DeepSeek, Gemini** do it server-side; there's nothing to place, and
  the savings show up as `response.usage.cache_read_tokens` (decoded for every
  provider). A workflow that *wants* the guarantee can filter on it.
- **`REUSABLE_PROMPT_CACHE`** — the **portable** create-once / reference-many
  surface: `create_cache(...) -> handle`, then `chat(cache=handle)`. Backed by
  **Gemini** `CachedContent` (a server resource) *and* **Anthropic** (a
  client-side handle that replays the prefix with a `cache_control` breakpoint) —
  so the same caller code caches on either provider. The caller holds the handle,
  so reuse and its cost stay explicit.

There are two layers. The **portable** one — write it once, run it on Anthropic
*or* Gemini, no branching:

```python
from inferenceswitch import LLMClient, Message, Text

client = LLMClient()

def answer(model, questions):                 # model picked from a dropdown / .env
    handle = client.create_cache(              # Gemini: server resource; Anthropic: replay handle
        model=model,
        system=big_system_prompt,
        messages=[Message("user", [Text(reference_doc)])],
        ttl=3600,                              # seconds (exact on Gemini; ~5 min on Anthropic today)
    )
    try:
        # Only the dynamic tail goes here — the cached prefix owns system + tools.
        return [client.chat(model=model, cache=handle, messages=[Message("user", [Text(q)])]).text
                for q in questions]
    finally:
        client.delete_cache(handle)            # frees the Gemini resource; no-op for a replay handle

answer("claude-opus-4-8", qs)     # ✓
answer("gemini-2.5-flash", qs)    # ✓ — identical code
```

`create_cache`/`chat(cache=)` on a provider without `REUSABLE_PROMPT_CACHE`
raises `UnsupportedCapabilityError`; a handle is not portable across providers
(passing one to a different route raises).

And the **low-level, Anthropic-specific** one — `EXPLICIT_PROMPT_CACHING` —
inline breakpoints for arbitrary placement when you want direct control:

```python
resp = client.chat(
    model="claude-opus-4-8",
    messages=[Message("user", [Text(long_context, cache=True), Text(question)])],
)
```

The marker **is** the declaration: its presence derives an
`EXPLICIT_PROMPT_CACHING` requirement, so `models_for(Capability.EXPLICIT_PROMPT_CACHING)`
lists only Anthropic models and routing a `cache=True` request elsewhere raises
rather than silently dropping the breakpoint.

> **Why not one inline flag for everything?** We *don't* fake Gemini's handle
> behind `cache=True`: Gemini's cache pays off only by deliberately reusing a
> named resource, so a stateless per-request marker would create a billable
> resource and never reuse it — forcing it would bury per-process billable state
> in the library. The unification runs the *other* direction (an Anthropic
> handle that replays inline), where the hidden part is free and stateless. Two
> seams stay visible on purpose: cache **minimums** (Anthropic ~1024 tokens,
> silently skipped below; Gemini ~32k, which *raises*) and **TTL granularity**
> (exact seconds on Gemini; 5-min ephemeral on Anthropic today).

See [`examples/prompt_caching.py`](examples/prompt_caching.py) for one runnable
example per mechanism.

## Architecture

- **Capability model** (`capabilities.py`) — adapters declare their *structured-
  output mechanism* and *schema dialect*. The thick call dispatches on these,
  never on provider name. Adding another OpenAI clone is a registry row.
- **Static registry** (`registry.py`) — providers are data, not plugins. Nothing
  is loaded dynamically, but you can hand the client your own registry — see
  [Adding a provider](#adding-a-provider).
- **Adapters** (`adapters/`) — one `OpenAICompatibleAdapter` for all OpenAI-shaped
  providers; native `AnthropicAdapter` and `GeminiAdapter` where the wire format
  genuinely differs. SDKs are imported lazily.
- **Client** (`client.py`) — resolution + key resolution + adapter caching +
  the uniform `generate_structured_json` / `generate_text` calls.

### Adding a provider

The built-in provider set is not a closed list. `LLMClient` takes a `registry`,
so you can add a provider — a self-hosted server, a regional or national API, a
gateway, another OpenAI clone — **without forking inferenceswitch**. A provider is a
`ProviderSpec` value; adding one is writing that value, not writing code.

```python
from inferenceswitch import (
    LLMClient, ProviderSpec, Capabilities, Capability,
    SchemaDialect, StructuredMode, default_registry,
)
from inferenceswitch.registry import KIND_OPENAI

registry = default_registry()          # start from the built-ins ...
registry.add(                          # ... and add your own
    ProviderSpec(
        name="mylab",
        kind=KIND_OPENAI,              # drives the OpenAI-compatible adapter
        base_url="https://llm.mylab.example/v1",
        api_key_env="MYLAB_API_KEY",
        capabilities=Capabilities(
            structured_output=StructuredMode.RESPONSE_FORMAT_JSON_SCHEMA,
            schema_dialect=SchemaDialect.OPENAI,
            features=frozenset({
                Capability.MULTI_TURN, Capability.SYSTEM_PROMPT,
                Capability.TOOL_CALLING, Capability.STRICT_SCHEMA_OUTPUT,
            }),
            config={"openai_strict": False},
        ),
        model_prefixes=("mylab-",),          # bare `mylab-*` routes here
        known_models=("mylab-7b-instruct",), # catalog for models_for()
    )
)

client = LLMClient(registry=registry)
client.generate_text(model="mylab-7b-instruct", prompt="...")
```

Your provider is now a first-class citizen: it routes by prefix, by
`mylab/<model>`, and by bare known-model name; it appears in `models_for()` and
`providers_for()` alongside the built-ins; and `supports()` / `require()` gate on
the features you declared. Pass `Registry([...])` instead of `default_registry()`
to ship a *restricted* set — e.g. locals only, for an offline or
data-residency-constrained deployment.

Filling in the spec:

| Field | What to put |
|---|---|
| `kind` | `KIND_OPENAI` for anything OpenAI-shaped. `KIND_ANTHROPIC` / `KIND_GEMINI` only for those wire formats — a new *format* needs a new adapter, which is the one case that isn't just data. |
| `base_url` | Required for OpenAI-compatible providers; ignored by the native ones. |
| `api_key_env` | Env var to read. `None` + `is_local=True` for keyless local servers. Or skip env vars entirely with `key_provider=` (below). |
| `capabilities` | Declare **fail-closed**: list a feature only where it reliably works, since `require()` trusts this. Local/uncertain servers want `StructuredMode.JSON_OBJECT_BEST_EFFORT`. |
| `model_prefixes` | Leave empty when your model names collide with another host's (as Groq's do) — callers then select you explicitly. |
| `known_models` | Optional. With a catalog you contribute one routable `provider/model` option per model; without one you still appear in `models_for()` as a bare provider whose `needs_model` is True. |

The registry is only consulted at resolution time, so a spec that's wrong is a
loud `ProviderResolutionError` or a provider-side 400 — never a silent
misroute.

### Bring-your-own-key (per-user)

The library does **not** store or encrypt keys. By default it reads each
provider's env var. For per-user BYOK, pass a `key_provider` (or a per-call
`api_key`) that returns the decrypted plaintext key:

```python
def key_provider(spec):
    return decrypt_for_current_user(spec.name)   # your storage/crypto

client = LLMClient(key_provider=key_provider)
```

## Why not just use the OpenAI format for everything?

For plain chat you can — that's why the OpenAI-compatible base adapter carries
six providers. But structured output is where the wire formats diverge
irreconcilably: Gemini rejects `additionalProperties` (so dict fields are
rewritten to key/value arrays and back), Anthropic forces JSON via tool use, and
each accepts a different schema dialect. A shim that "speaks OpenAI" to Gemini or
Anthropic drops schema enforcement, prompt caching, and thinking controls. The
native adapters exist to preserve exactly those.

## Development

```
pip install -e ".[dev]"
pytest            # resolution + schema tests run offline (no SDKs/network)
```
