"""The static, in-code provider registry.

This is deliberately not a plugin system. Providers are declared here as data —
no dynamic module loading, no caller-injected ``base_url`` from untrusted input.
That static surface is most of what keeps this library's security profile small.

A :class:`ProviderSpec` says: which adapter kind drives it, where its key comes
from, (for OpenAI-compatible providers) which ``base_url`` to hit, what it can do
(:class:`Capabilities`), and how bare model names route to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .capabilities import Capabilities, Capability, SchemaDialect, StructuredMode

#: Adapter kinds. Maps to a concrete adapter class in :mod:`inferenceswitch.adapters`.
KIND_OPENAI = "openai"        # OpenAI + every OpenAI-compatible server
KIND_ANTHROPIC = "anthropic"  # native
KIND_GEMINI = "gemini"        # native

#: Placeholder key for local servers that require *some* key string but don't
#: authenticate (Ollama, LM Studio). Never a real secret.
LOCAL_PLACEHOLDER_KEY = "inferenceswitch-local"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: str
    capabilities: Capabilities
    #: Env var holding this provider's API key. ``None`` for keyless local servers.
    api_key_env: str | None = None
    #: Base URL for OpenAI-compatible providers. ``None`` uses the SDK default
    #: (i.e. api.openai.com). Ignored by native adapters.
    base_url: str | None = None
    #: Model-name prefixes that route to this provider when no explicit provider
    #: is given (e.g. ``"claude"`` -> anthropic). Empty means the provider must be
    #: selected explicitly — used where model names are ambiguous across hosts
    #: (Groq/Ollama/LM Studio all serve ``llama-*``).
    model_prefixes: tuple[str, ...] = ()
    #: Optional curated model IDs, both for bare-name routing and as a catalog.
    known_models: tuple[str, ...] = ()
    #: True for local servers with no authentication.
    is_local: bool = False


class Registry:
    """An ordered, name-keyed collection of :class:`ProviderSpec`."""

    def __init__(self, specs: list[ProviderSpec] | None = None) -> None:
        self._by_name: dict[str, ProviderSpec] = {}
        for spec in specs or []:
            self.add(spec)

    def add(self, spec: ProviderSpec) -> None:
        if spec.name in self._by_name:
            raise ValueError(f"Duplicate provider name in registry: {spec.name!r}")
        self._by_name[spec.name] = spec

    def get(self, name: str) -> ProviderSpec:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"Unknown provider {name!r}. Registered: {', '.join(self._by_name)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> ProviderSpec:
        return self.get(name)

    def __iter__(self):
        return iter(self._by_name.values())

    @property
    def names(self) -> list[str]:
        return list(self._by_name)


# ── capability presets ───────────────────────────────────────────────────────
#
# Feature sets describe what each PROVIDER can do (from real advanced usage), not
# only what inferenceswitch currently wraps — so workflows can discover a capability
# and, where the library has no first-class method yet, drop to raw_client().
# Populated conservatively (fail-closed): a token is present only where support
# is reliable across the provider, so `require()` never green-lights a 400.

_C = Capability

# Shared spine for every OpenAI-compatible *hosted* provider.
_OPENAI_COMPAT_BASE = frozenset(
    {
        _C.MULTI_TURN,
        _C.SYSTEM_PROMPT,
        _C.SAMPLING_PARAMS,
        _C.STREAMING,
        _C.STRICT_SCHEMA_OUTPUT,
        _C.TOOL_CALLING,
        _C.PARALLEL_TOOL_USE,
    }
)

_OPENAI_CAPS = Capabilities(
    structured_output=StructuredMode.RESPONSE_FORMAT_JSON_SCHEMA,
    schema_dialect=SchemaDialect.OPENAI,
    features=_OPENAI_COMPAT_BASE
    | {_C.VISION, _C.REASONING_EFFORT, _C.BATCH, _C.IMPLICIT_PROMPT_CACHING},
    # reasoning_param names the request field the OpenAI-compat adapter sets for
    # effort; only OpenAI itself exposes one on chat.completions.
    config={"openai_strict": False, "reasoning_param": "reasoning_effort"},
)
_MISTRAL_CAPS = Capabilities(
    structured_output=StructuredMode.RESPONSE_FORMAT_JSON_SCHEMA,
    schema_dialect=SchemaDialect.OPENAI,
    features=_OPENAI_COMPAT_BASE | {_C.VISION, _C.BATCH},
    config={"openai_strict": False},
)
_DEEPSEEK_CAPS = Capabilities(
    structured_output=StructuredMode.RESPONSE_FORMAT_JSON_SCHEMA,
    schema_dialect=SchemaDialect.OPENAI,
    # NOTE: DeepSeek reasoning is accessed by selecting the `deepseek-reasoner`
    # MODEL, not by an effort knob — so no REASONING_EFFORT (which means
    # "effort control"). No effort param is sent to it.
    # It DOES do automatic context caching (reported back as cached tokens).
    features=_OPENAI_COMPAT_BASE | {_C.IMPLICIT_PROMPT_CACHING},
    config={"openai_strict": False},
)
_GROQ_CAPS = Capabilities(
    structured_output=StructuredMode.RESPONSE_FORMAT_JSON_SCHEMA,
    schema_dialect=SchemaDialect.OPENAI,
    features=_OPENAI_COMPAT_BASE,  # fast inference; no batch / explicit caching
    config={"openai_strict": False},
)
# Local servers accept the OpenAI wire format but json_schema enforcement and
# tool/vision support are model-dependent, so default to best-effort JSON and a
# minimal feature set. Override per entry if your local model does more.
_LOCAL_CAPS = Capabilities(
    structured_output=StructuredMode.JSON_OBJECT_BEST_EFFORT,
    schema_dialect=SchemaDialect.OPENAI,
    features=frozenset(
        {
            _C.MULTI_TURN,
            _C.SYSTEM_PROMPT,
            _C.SAMPLING_PARAMS,
            _C.STREAMING,
            _C.TOOL_CALLING,
        }
    ),
)
_ANTHROPIC_CAPS = Capabilities(
    structured_output=StructuredMode.FORCED_TOOL_USE,
    schema_dialect=SchemaDialect.ANTHROPIC,
    # NOTE: SAMPLING_PARAMS deliberately absent — support is per-MODEL (Opus
    # 4.7/4.8 & Fable reject temperature with a 400; Sonnet 4.6 / Haiku 4.5
    # accept it). Fail-closed at the provider level; a per-model policy can add
    # it back where valid.
    features=frozenset(
        {
            _C.MULTI_TURN,
            _C.SYSTEM_PROMPT,
            _C.STREAMING,
            _C.STRICT_SCHEMA_OUTPUT,
            _C.TOOL_CALLING,
            _C.PARALLEL_TOOL_USE,
            _C.VISION,
            _C.PDF_INPUT,
            _C.REASONING_EFFORT,
            _C.EXPLICIT_PROMPT_CACHING,  # low-level: inline cache_control breakpoints
            # Unified create-once/reference-many surface, backed here by replaying
            # the prefix with a cache_control breakpoint (no server resource).
            _C.REUSABLE_PROMPT_CACHE,
            _C.TOKEN_COUNTING,
            _C.BATCH,
        }
    ),
)
_GEMINI_CAPS = Capabilities(
    structured_output=StructuredMode.RESPONSE_SCHEMA,
    schema_dialect=SchemaDialect.GEMINI,
    features=frozenset(
        {
            _C.MULTI_TURN,
            _C.SYSTEM_PROMPT,
            _C.SAMPLING_PARAMS,
            _C.STREAMING,
            _C.STRICT_SCHEMA_OUTPUT,
            _C.TOOL_CALLING,
            _C.PARALLEL_TOOL_USE,
            _C.VISION,
            _C.PDF_INPUT,
            _C.AUDIO_INPUT,
            _C.REASONING_EFFORT,
            # Gemini's caller-controlled caching is out-of-band CachedContent (a
            # separate create-call + handle + TTL), NOT inline breakpoints, so it
            # does NOT carry EXPLICIT_PROMPT_CACHING (which means inline). Its
            # 2.5 models DO cache prefixes automatically -> IMPLICIT, and its
            # out-of-band CachedContent handle is wrapped as REUSABLE_PROMPT_CACHE
            # (create_cache -> CacheHandle -> chat(cache=...)).
            _C.IMPLICIT_PROMPT_CACHING,
            _C.REUSABLE_PROMPT_CACHE,
            _C.TOKEN_COUNTING,
            _C.BATCH,
        }
    ),
)


def default_registry() -> Registry:
    """The built-in provider set.

    Native adapters: Anthropic, Gemini. OpenAI-compatible base adapter: OpenAI,
    Mistral, DeepSeek, Groq, plus local Ollama and LM Studio servers.
    """
    return Registry(
        [
            # ── native ──────────────────────────────────────────────────────
            ProviderSpec(
                name="anthropic",
                kind=KIND_ANTHROPIC,
                capabilities=_ANTHROPIC_CAPS,
                api_key_env="ANTHROPIC_API_KEY",
                model_prefixes=("claude",),
                known_models=(
                    "claude-haiku-4-5",
                    "claude-sonnet-4-6",
                    "claude-opus-4-8",
                ),
            ),
            ProviderSpec(
                name="gemini",
                kind=KIND_GEMINI,
                capabilities=_GEMINI_CAPS,
                api_key_env="GEMINI_API_KEY",
                model_prefixes=("gemini",),
                known_models=(
                    "gemini-2.5-flash",
                    "gemini-3-flash",
                    "gemini-3.5-flash",
                ),
            ),
            # ── OpenAI-compatible (hosted) ──────────────────────────────────
            ProviderSpec(
                name="openai",
                kind=KIND_OPENAI,
                capabilities=_OPENAI_CAPS,
                api_key_env="OPENAI_API_KEY",
                base_url=None,  # SDK default: api.openai.com
                model_prefixes=("gpt-", "o1", "o3", "o4"),
            ),
            ProviderSpec(
                name="mistral",
                kind=KIND_OPENAI,
                capabilities=_MISTRAL_CAPS,
                api_key_env="MISTRAL_API_KEY",
                base_url="https://api.mistral.ai/v1",
                model_prefixes=("mistral", "codestral", "ministral", "magistral"),
            ),
            ProviderSpec(
                name="deepseek",
                kind=KIND_OPENAI,
                capabilities=_DEEPSEEK_CAPS,
                api_key_env="DEEPSEEK_API_KEY",
                base_url="https://api.deepseek.com/v1",
                model_prefixes=("deepseek",),
            ),
            ProviderSpec(
                name="groq",
                kind=KIND_OPENAI,
                capabilities=_GROQ_CAPS,
                api_key_env="GROQ_API_KEY",
                base_url="https://api.groq.com/openai/v1",
                # No prefixes: Groq serves llama/qwen/etc. that also run locally.
                # Select it explicitly (provider="groq" or "groq/<model>").
            ),
            # ── OpenAI-compatible (local, keyless) ──────────────────────────
            ProviderSpec(
                name="ollama",
                kind=KIND_OPENAI,
                capabilities=_LOCAL_CAPS,
                api_key_env=None,
                base_url="http://localhost:11434/v1",
                is_local=True,
            ),
            ProviderSpec(
                name="lmstudio",
                kind=KIND_OPENAI,
                capabilities=_LOCAL_CAPS,
                api_key_env=None,
                base_url="http://localhost:1234/v1",
                is_local=True,
            ),
        ]
    )
