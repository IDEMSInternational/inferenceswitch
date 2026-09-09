"""The thick client facade.

``LLMClient`` is the single entry point. It resolves a model string to a provider,
resolves that provider's API key, builds (and caches) the right adapter, and
forwards the uniform structured-output / text calls. Callers never touch an
adapter or an SDK directly, and never write a ``match provider`` block.

Key handling is intentionally minimal: the library reads keys from the
environment by default and takes plaintext keys from a caller-supplied
``key_provider`` otherwise. It does NOT own key storage or encryption — an app
with per-user BYOK keys passes a ``key_provider`` (or a per-call ``api_key``)
that decrypts on its side.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from .adapters import Adapter, build_adapter
from .capabilities import Capabilities, Capability
from .errors import (
    LLMSwitchboardError,
    MissingAPIKeyError,
    ProviderResolutionError,
    UnsupportedCapabilityError,
)
from .messages import (
    CacheHandle,
    Effort,
    LLMResponse,
    Message,
    Text,
    Tool,
    ToolChoice,
    ToolResult,
    ToolUse,
)
from .registry import LOCAL_PLACEHOLDER_KEY, ProviderSpec, Registry, default_registry

#: A key provider maps a resolved ProviderSpec to its plaintext API key.
KeyProvider = Callable[[ProviderSpec], str]


def env_key_provider(spec: ProviderSpec) -> str:
    """Default key source: read the provider's env var; local servers get a
    harmless placeholder (they don't authenticate)."""
    if spec.api_key_env is None:
        return LOCAL_PLACEHOLDER_KEY
    key = os.getenv(spec.api_key_env)
    if not key:
        raise MissingAPIKeyError(
            f"No API key for provider {spec.name!r}: set ${spec.api_key_env} "
            "or pass api_key=/key_provider."
        )
    return key


def _has_cache_markers(messages: list[Message], tools: list[Tool] | None) -> bool:
    """True if any content block or tool requests an explicit cache breakpoint.

    This is the *derived* requirement signal: a caller who marks a part with
    ``cache=True`` has, by that act, declared the request needs
    EXPLICIT_PROMPT_CACHING — no separate hand-declaration. One source of truth
    drives both the fail-closed guard and ``models_for`` filtering, so the marker
    and the dropdown can never disagree.
    """
    for message in messages:
        for part in message.parts():
            if isinstance(part, Text) and part.cache:
                return True
    return any(tool.cache for tool in tools or ())


@dataclass(frozen=True)
class Resolution:
    """The provider a request routed to, and the bare model to send it."""

    spec: ProviderSpec
    model: str


class ModelChoice(str):
    """One selectable option returned by :meth:`LLMClient.models_for`.

    It *is* a ``str``, so a curated option — ``"anthropic/claude-opus-4-8"`` —
    can be passed straight to :meth:`LLMClient.chat` as before.

    Providers with no curated catalog (Groq, the local servers, your own
    registry entries) still appear, as the bare provider name with
    :attr:`model` ``None``. Those cannot route on their own: the caller supplies
    the model, ``chat(provider=choice.provider, model=...)``. Check
    :attr:`needs_model` before routing a choice you did not curate.
    """

    provider: str
    model: str | None

    def __new__(cls, provider: str, model: str | None = None) -> ModelChoice:
        self = super().__new__(cls, f"{provider}/{model}" if model else provider)
        self.provider = provider
        self.model = model
        return self

    @property
    def needs_model(self) -> bool:
        """True when the caller must still supply a model name."""
        return self.model is None

    def __repr__(self) -> str:
        return f"ModelChoice({str.__str__(self)!r})"


class LLMClient:
    def __init__(
        self,
        registry: Registry | None = None,
        key_provider: KeyProvider | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.key_provider = key_provider or env_key_provider
        self._adapters: dict[tuple[str, str], Adapter] = {}

    # ── resolution ────────────────────────────────────────────────────────────

    def resolve(self, model: str, provider: str | None = None) -> Resolution:
        """Route a model string to exactly one provider.

        Order of precedence:
          1. Explicit ``provider=``.
          2. ``"provider/model"`` where the first segment is a registered provider.
          3. A bare model that matches exactly one provider's ``known_models``.
          4. A registered ``model_prefix`` (e.g. ``claude-*`` -> anthropic).
        No match — or an ambiguous one — is a hard error, never a silent default.
        """
        if provider is not None:
            return Resolution(self.registry.get(provider), model)

        if "/" in model:
            head, tail = model.split("/", 1)
            if head in self.registry:
                return Resolution(self.registry[head], tail)

        exact = [s for s in self.registry if model in s.known_models]
        if len(exact) == 1:
            return Resolution(exact[0], model)
        if len(exact) > 1:
            raise ProviderResolutionError(
                f"Model {model!r} is registered under multiple providers "
                f"({', '.join(s.name for s in exact)}); pass provider= explicitly."
            )

        prefix_matches = [
            s for s in self.registry
            if any(model.startswith(p) for p in s.model_prefixes)
        ]
        if len(prefix_matches) == 1:
            return Resolution(prefix_matches[0], model)
        if len(prefix_matches) > 1:
            raise ProviderResolutionError(
                f"Model {model!r} matches multiple provider prefixes "
                f"({', '.join(s.name for s in prefix_matches)}); pass provider= "
                "or use 'provider/model'."
            )

        raise ProviderResolutionError(
            f"Cannot route model {model!r} to any provider. Pass provider= "
            f"explicitly, or use 'provider/model'. Registered providers: "
            f"{', '.join(self.registry.names)}."
        )

    # ── capability introspection ──────────────────────────────────────────────

    def _spec(self, provider: str | None, model: str | None) -> ProviderSpec:
        if provider is not None:
            return self.registry.get(provider)
        if model is not None:
            return self.resolve(model).spec
        raise TypeError("Pass provider= or model=.")

    def capabilities(
        self, provider: str | None = None, *, model: str | None = None
    ) -> Capabilities:
        """The :class:`Capabilities` for a provider (by name) or a model (routed)."""
        return self._spec(provider, model).capabilities

    def supports(
        self,
        capability: Capability,
        provider: str | None = None,
        *,
        model: str | None = None,
    ) -> bool:
        """Whether the resolved provider advertises ``capability``.

        Presence reflects what the *provider* can do; for a capability llmswitchboard
        doesn't wrap with a first-class method yet, reach it via
        :meth:`raw_client`.
        """
        return self._spec(provider, model).capabilities.has(capability)

    def require(
        self,
        capability: Capability,
        provider: str | None = None,
        *,
        model: str | None = None,
    ) -> None:
        """Assert a capability, or raise :class:`UnsupportedCapabilityError`.

        The fail-loud gate for model-specific workflows — call it before
        attempting something only some providers can do, so an unsupported
        request stops here with a clear message instead of degrading silently
        or 400-ing deep in an SDK.
        """
        spec = self._spec(provider, model)
        if not spec.capabilities.has(capability):
            supported = ", ".join(sorted(c.value for c in spec.capabilities.features))
            raise UnsupportedCapabilityError(
                f"Provider {spec.name!r} does not support {capability.value!r}. "
                f"Supported: {supported or '(none)'}."
            )

    def raw_client(
        self,
        provider: str | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """The underlying provider SDK client — the escape hatch for features
        this library doesn't wrap yet, with routing/keys still handled here."""
        spec = self._spec(provider, model)
        return self._adapter(spec, api_key).raw_client

    # ── adapters ──────────────────────────────────────────────────────────────

    def _adapter(self, spec: ProviderSpec, api_key: str | None) -> Adapter:
        key = api_key or self.key_provider(spec)
        cache_key = (spec.name, key)
        adapter = self._adapters.get(cache_key)
        if adapter is None:
            adapter = build_adapter(spec, key)
            self._adapters[cache_key] = adapter
        return adapter

    # ── public calls ──────────────────────────────────────────────────────────

    def generate_structured_json(
        self,
        *,
        model: str,
        schema: dict,
        prompt: str,
        system: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        tool_name: str = "generate_json",
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Any:
        """Return schema-constrained JSON from whichever provider ``model`` routes
        to, handling the per-provider mechanism and schema dialect internally."""
        resolution = self.resolve(model, provider)
        adapter = self._adapter(resolution.spec, api_key)
        return adapter.generate_structured_json(
            model=resolution.model,
            prompt=prompt,
            schema=schema,
            system=system,
            tool_name=tool_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Return a plain text completion."""
        resolution = self.resolve(model, provider)
        adapter = self._adapter(resolution.spec, api_key)
        return adapter.generate_text(
            model=resolution.model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice = ToolChoice.AUTO,
        force_tool: str | None = None,
        system: str | None = None,
        effort: Effort | None = None,
        cache: CacheHandle | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """A normalized, optionally tool-using turn.

        Same call regardless of provider — the adapter maps tools, reasoning
        effort, and the message history into that provider's native mechanism and
        normalizes the reply (text, tool calls, stop reason, usage). Passing
        ``tools`` (or ``effort``) to a provider without the matching capability
        raises :class:`UnsupportedCapabilityError` rather than degrading silently.

        ``cache`` reuses a handle from :meth:`create_cache`: the cached prefix
        (system + messages + tools it was built from) is not resent, so pass only
        the dynamic tail in ``messages``.
        """
        resolution = self.resolve(model, provider)
        if tools:
            self.require(Capability.TOOL_CALLING, resolution.spec.name)
        if effort is not None:
            self.require(Capability.REASONING_EFFORT, resolution.spec.name)
        if _has_cache_markers(messages, tools):
            # Derived from the markers themselves — fail closed on a provider
            # whose caching isn't inline (Gemini's is out-of-band; OpenAI's is
            # automatic and unaddressable) rather than dropping the marker.
            self.require(Capability.EXPLICIT_PROMPT_CACHING, resolution.spec.name)
        if cache is not None:
            self.require(Capability.REUSABLE_PROMPT_CACHE, resolution.spec.name)
            if cache.provider != resolution.spec.name:
                raise LLMSwitchboardError(
                    f"Cache handle was created for provider {cache.provider!r} but "
                    f"this request routes to {resolution.spec.name!r}; a handle is "
                    "not portable across providers."
                )
            if tools or system:
                # Uniform contract across providers: the cached prefix supplies
                # the system prompt and tools (Gemini physically can't take them
                # again alongside a cache). Pass only the dynamic tail here.
                raise LLMSwitchboardError(
                    "With cache=, the cached prefix supplies the system prompt and "
                    "tools; pass only the dynamic tail in messages (not system=/tools=)."
                )
        adapter = self._adapter(resolution.spec, api_key)
        return adapter.chat(
            model=resolution.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            force_tool=force_tool,
            system=system,
            effort=effort,
            cache=cache,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def create_cache(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[Tool] | None = None,
        ttl: int | None = None,
        provider: str | None = None,
        api_key: str | None = None,
    ) -> CacheHandle:
        """Create a reusable server-side prompt cache and return a
        :class:`~llmswitchboard.CacheHandle` to pass to :meth:`chat` as ``cache=``.

        For a large static prefix (system prompt, few-shot context, tool set)
        reused across many requests: create the cache once, then reference the
        handle so the prefix is neither resent nor re-billed on each call. Routing
        to a provider without REUSABLE_PROMPT_CACHE raises
        :class:`UnsupportedCapabilityError`. The returned handle names a billable,
        TTL-bound resource the caller owns — reuse (and its cost) is explicit.
        """
        resolution = self.resolve(model, provider)
        self.require(Capability.REUSABLE_PROMPT_CACHE, resolution.spec.name)
        adapter = self._adapter(resolution.spec, api_key)
        return adapter.create_cache(
            model=resolution.model,
            messages=messages,
            system=system,
            tools=tools,
            ttl=ttl,
        )

    def delete_cache(self, handle: CacheHandle, *, api_key: str | None = None) -> None:
        """Release a cache from :meth:`create_cache`. No-op for a replay-backed
        handle (Anthropic — nothing server-side to free, so no key is needed);
        deletes the server resource for a server-backed one (Gemini)."""
        if handle.name is None:
            return
        spec = self.registry.get(handle.provider)
        self._adapter(spec, api_key).delete_cache(handle)

    def run_tools(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[Tool],
        handlers: dict[str, Callable[[dict], Any]],
        provider: str | None = None,
        api_key: str | None = None,
        system: str | None = None,
        effort: Effort | None = None,
        max_turns: int = 10,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Drive the full tool-use loop: call the model, run the requested tools
        via ``handlers`` (``{tool_name: fn(input) -> result}``), feed results
        back, and repeat until the model stops calling tools.

        Returns the final :class:`LLMResponse`. A handler exception, or a call to
        an unmapped tool, is returned to the model as an error tool result rather
        than raised — so the model can recover. Raises :class:`LLMSwitchboardError`
        only if ``max_turns`` is exceeded.
        """
        history = list(messages)
        for _ in range(max_turns):
            response = self.chat(
                model=model,
                messages=history,
                tools=tools,
                system=system,
                effort=effort,
                provider=provider,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not response.tool_calls:
                return response

            # Record the assistant's turn (text + the tool calls it made)...
            assistant_parts: list = [ToolUse(c.id, c.name, c.input) for c in response.tool_calls]
            if response.text:
                assistant_parts.insert(0, Text(response.text))
            history.append(Message(role="assistant", content=assistant_parts))

            # ...then run each tool and return the results in one user turn.
            results: list = []
            for call in response.tool_calls:
                handler = handlers.get(call.name)
                if handler is None:
                    results.append(
                        ToolResult(call.id, f"No handler for tool {call.name!r}.",
                                   is_error=True, name=call.name)
                    )
                    continue
                try:
                    output = handler(call.input)
                    results.append(ToolResult(call.id, str(output), name=call.name))
                except Exception as exc:  # noqa: BLE001 - report back to the model
                    results.append(
                        ToolResult(call.id, f"Tool error: {exc}", is_error=True, name=call.name)
                    )
            history.append(Message(role="user", content=results))

        raise LLMSwitchboardError(f"run_tools exceeded max_turns={max_turns} without finishing.")

    def providers_for(self, *capabilities: Capability) -> list[str]:
        """Provider names that support **all** of ``capabilities`` — the basis for
        'given this call, which providers can serve it?'."""
        needed = set(capabilities)
        return [s.name for s in self.registry if needed <= s.capabilities.features]

    def models_for(self, *capabilities: Capability) -> list[ModelChoice]:
        """Every option that can serve a call needing all of ``capabilities`` —
        hand these to a UI / .env / prompt for the user to pick from.

        Each entry is a :class:`ModelChoice`, i.e. a ``str``. Providers with a
        curated catalog contribute one routable ``provider/model`` per model,
        which passes straight back to :meth:`chat`. Providers without one
        contribute a single bare-provider entry with ``needs_model`` True — they
        qualify on capabilities but the caller must supply the model name, so
        the menu no longer silently hides Groq, the local servers, or an
        uncurated provider from your own registry::

            for choice in client.models_for(Capability.TOOL_CALLING):
                if choice.needs_model:
                    client.chat(provider=choice.provider, model=ask_user(), ...)
                else:
                    client.chat(model=choice, ...)

        Filter with ``[c for c in ... if not c.needs_model]`` for the
        pass-straight-through subset.
        """
        needed = set(capabilities)
        choices: list[ModelChoice] = []
        for spec in self.registry:
            if not needed <= spec.capabilities.features:
                continue
            if spec.known_models:
                choices.extend(ModelChoice(spec.name, m) for m in spec.known_models)
            else:
                choices.append(ModelChoice(spec.name))
        return choices

    def verify_key(
        self, provider: str, *, api_key: str | None = None
    ) -> tuple[bool, str]:
        """Smoke-test a provider's key via its cheapest authenticated call."""
        spec = self.registry.get(provider)
        adapter = self._adapter(spec, api_key)
        return adapter.verify_key()

    def known_models(self) -> list[str]:
        """Every curated model ID across registered providers, as
        ``provider/model`` so the result is unambiguously routable."""
        return [
            f"{spec.name}/{m}"
            for spec in self.registry
            for m in spec.known_models
        ]
