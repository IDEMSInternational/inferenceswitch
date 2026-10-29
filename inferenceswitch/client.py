"""The thick client facade.

``Client`` is the single entry point. It resolves a model string to a provider,
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
import time
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal, overload

from .adapters import Adapter, build_adapter
from .capabilities import Capabilities, Capability
from .cards import ModelCard, ModelCards, Reliability, Rung, default_cards
from .errors import (
    InferenceSwitchError,
    MissingAPIKeyError,
    ProviderResolutionError,
    UnsupportedCapabilityError,
)
from .messages import (
    CacheHandle,
    Effort,
    Response,
    Message,
    StreamCallback,
    StructuredResult,
    Text,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolRun,
    ToolUse,
    Usage,
)
from .registry import LOCAL_PLACEHOLDER_KEY, ProviderSpec, Registry, default_registry
from .selection import (
    AvailabilityCheck,
    Preference,
    ProviderAvailability,
    Selection,
    key_availability,
    make_request,
    select,
    static_availability,
)

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


def _streaming(on_stream: StreamCallback | None) -> dict[str, Any]:
    """``on_stream`` as adapter kwargs, empty when there is none, so an adapter
    written before streaming (a caller's own, or a test fake) still works."""
    return {} if on_stream is None else {"on_stream": on_stream}


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
    """One selectable option returned by :meth:`Client.models_for`.

    It *is* a ``str``, so a curated option — ``"anthropic/claude-opus-4-8"`` —
    can be passed straight to :meth:`Client.chat` as before.

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


class Client:
    def __init__(
        self,
        registry: Registry | None = None,
        key_provider: KeyProvider | None = None,
        *,
        cards: ModelCards | None = None,
        availability: AvailabilityCheck | Mapping[str, bool | Iterable[str]] | None = None,
    ) -> None:
        """``cards`` replaces the shipped model cards; to add to or override
        them, pass ``default_cards().merged(...)``. ``availability`` replaces the
        default check (keys for hosted providers, a probe for local servers)
        used by :meth:`select_model`: a callable, or a mapping as for
        :func:`~inferenceswitch.static_availability`."""
        self.registry = registry or default_registry()
        self.key_provider = key_provider or env_key_provider
        self.cards = default_cards() if cards is None else cards
        if availability is None:
            availability = key_availability(self.key_provider)
        elif isinstance(availability, Mapping):
            availability = static_availability(availability)
        self._availability_check: AvailabilityCheck = availability
        self._availability: dict[str, ProviderAvailability] = {}
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

    # ── model selection by rung ───────────────────────────────────────────────

    def select_model(
        self,
        rung: Rung | str,
        *,
        needs: Iterable[Capability | str] = (),
        structured_output: Reliability | str | None = None,
        min_context: int | None = None,
        local_only: bool = False,
        prefer: Preference | str = Preference.CHEAPEST,
        allow_higher: bool = True,
        providers: Iterable[str] | None = None,
        refresh: bool = False,
    ) -> Selection:
        """Pick an available model for a rung, hard requirements and a preference.

        Candidates are the model cards' routes. A route counts when its provider
        is available (see :meth:`availability`), and the card meets every
        requirement: rung ``rung`` or above (exactly ``rung`` with
        ``allow_higher=False``; never below), every capability in ``needs``,
        a ``structured_output`` grade at least as good as the one given, a
        context window of at least ``min_context`` tokens, and a local server
        when ``local_only``. The survivors are ranked by ``prefer``
        (``"local"``, ``"cheapest"`` or ``"fastest"``; see
        :class:`~inferenceswitch.Preference`).

        ``providers`` is an allow-list of registry provider names, applied to
        the routes before anything else: routes on other providers are
        excluded (and their availability is not checked), the explanation says
        the candidates were restricted, and ``to_dict()`` records the list.
        An unregistered name, or an empty list, raises :class:`ValueError`. ``None`` (the
        default) allows every provider.

        Returns a :class:`~inferenceswitch.Selection` whose ``model`` passes
        straight to :meth:`chat`, with the reason and every candidate
        considered (``to_dict()`` for a run record). Raises
        :class:`~inferenceswitch.NoModelAvailableError` naming what was
        missing when nothing fits. Strings are accepted for every enum, so a
        profile file can be passed through as read.
        """
        request = make_request(
            rung,
            needs=needs,
            structured_output=structured_output,
            min_context=min_context,
            local_only=local_only,
            prefer=prefer,
            allow_higher=allow_higher,
            providers=providers,
        )
        if refresh:
            self._availability.clear()
        return select(request, self.cards, self.registry, self._provider_availability)

    def availability(
        self, providers: Iterable[str] | None = None, *, refresh: bool = False
    ) -> list[ProviderAvailability]:
        """Whether each provider (default: every registered one) can be used here.

        Results are cached on the client, since a local check is a network
        probe; ``refresh=True`` checks again.
        """
        if refresh:
            self._availability.clear()
        names = self.registry.names if providers is None else list(providers)
        return [self._provider_availability(name) for name in names]

    def _provider_availability(self, provider: str) -> ProviderAvailability:
        cached = self._availability.get(provider)
        if cached is None:
            cached = self._availability_check(self.registry.get(provider))
            self._availability[provider] = cached
        return cached

    # ── capability introspection ──────────────────────────────────────────────

    def _spec(self, provider: str | None, model: str | None) -> ProviderSpec:
        if model is not None:
            return self.resolve(model, provider).spec
        if provider is not None:
            return self.registry.get(provider)
        raise TypeError("Pass provider= or model=.")

    def card(self, model: str, provider: str | None = None) -> ModelCard | None:
        """The model card for ``model`` (routed as for :meth:`resolve`), or
        ``None`` when no card lists it for that provider."""
        resolution = self.resolve(model, provider)
        return self.cards.find(resolution.spec.name, resolution.model)

    def _caps(self, resolution: Resolution) -> Capabilities:
        card = self.cards.find(resolution.spec.name, resolution.model)
        caps = resolution.spec.capabilities.for_model(resolution.model)
        return caps if card is None else card.capabilities(caps)

    def _temperature(self, resolution: Resolution, temperature: float) -> float | None:
        """``temperature``, or ``None`` (not sent) for a model that rejects
        sampling parameters, so the caller's default never causes a 400."""
        return temperature if self._caps(resolution).has(Capability.SAMPLING_PARAMS) else None

    def capabilities(
        self, provider: str | None = None, *, model: str | None = None
    ) -> Capabilities:
        """The :class:`Capabilities` for a provider (by name), or for a model.

        For a model with a card, the provider's defaults with the card's
        feature overrides applied; otherwise the provider's defaults.
        """
        if model is not None:
            return self._caps(self.resolve(model, provider))
        return self._spec(provider, None).capabilities

    def supports(
        self,
        capability: Capability,
        provider: str | None = None,
        *,
        model: str | None = None,
    ) -> bool:
        """Whether the resolved provider advertises ``capability``.

        Presence reflects what the *provider* can do; for a capability inferenceswitch
        doesn't wrap with a first-class method yet, reach it via
        :meth:`raw_client`.
        """
        return self.capabilities(provider, model=model).has(capability)

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
        caps = self.capabilities(provider, model=model)
        if not caps.has(capability):
            spec = self._spec(provider, model)
            subject = f"Provider {spec.name!r}"
            if model is not None:
                subject = f"Model {model!r} on provider {spec.name!r}"
            supported = ", ".join(sorted(c.value for c in caps.features))
            raise UnsupportedCapabilityError(
                f"{subject} does not support {capability.value!r}. "
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

    def generate_structured(
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
        cache_prefix: str | None = None,
        on_stream: StreamCallback | None = None,
    ) -> StructuredResult:
        """Schema-constrained JSON from whichever provider ``model`` routes to,
        with the call's token usage, stop reason and raw provider response.

        Same request as :meth:`generate_structured_json`; use this one when you
        need per-call tokens or cost. ``usage`` fields the provider doesn't
        report are ``None``.

        ``cache_prefix`` is static text sent before ``prompt``, in the same
        user turn: a reference document or run context that repeats across
        calls while ``prompt`` varies. Where the provider takes inline cache
        breakpoints (EXPLICIT_PROMPT_CACHING, Anthropic) it is marked for
        caching, which also caches the system prompt and schema above it;
        ``usage.cache_write_tokens`` / ``usage.cache_read_tokens`` show the
        write and the later hits. Elsewhere it is sent first so implicit
        prefix caching can match it. It is a hint, not a requirement: the
        model sees the same text either way, so it never raises. Providers
        cache only prefixes above a minimum size (about 1024 tokens on most
        Claude models), and smaller ones are sent uncached.

        ``on_stream`` receives each :class:`~inferenceswitch.StreamDelta` of
        thinking and text while the reply streams; see :meth:`chat`.
        """
        resolution = self.resolve(model, provider)
        adapter = self._adapter(resolution.spec, api_key)
        return adapter.generate_structured(
            model=resolution.model,
            prompt=prompt,
            schema=schema,
            system=system,
            tool_name=tool_name,
            temperature=self._temperature(resolution, temperature),
            max_tokens=max_tokens,
            cache_prefix=cache_prefix,
            **_streaming(on_stream),
        )

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
        cache_prefix: str | None = None,
        on_stream: StreamCallback | None = None,
    ) -> Any:
        """Return schema-constrained JSON from whichever provider ``model`` routes
        to, handling the per-provider mechanism and schema dialect internally.

        The ``value`` of :meth:`generate_structured`, without the usage.
        ``cache_prefix`` and ``on_stream`` work as they do there."""
        return self.generate_structured(
            model=model,
            schema=schema,
            prompt=prompt,
            system=system,
            provider=provider,
            api_key=api_key,
            tool_name=tool_name,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_prefix=cache_prefix,
            on_stream=on_stream,
        ).value

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
        on_stream: StreamCallback | None = None,
    ) -> str:
        """Return a plain text completion. ``on_stream`` works as on :meth:`chat`."""
        resolution = self.resolve(model, provider)
        adapter = self._adapter(resolution.spec, api_key)
        return adapter.generate_text(
            model=resolution.model,
            prompt=prompt,
            system=system,
            temperature=self._temperature(resolution, temperature),
            max_tokens=max_tokens,
            **_streaming(on_stream),
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
        on_stream: StreamCallback | None = None,
    ) -> Response:
        """A normalized, optionally tool-using turn.

        Same call regardless of provider — the adapter maps tools, reasoning
        effort, and the message history into that provider's native mechanism and
        normalizes the reply (text, tool calls, stop reason, usage). Passing
        ``tools`` (or ``effort``) to a provider without the matching capability
        raises :class:`UnsupportedCapabilityError` rather than degrading silently.

        ``cache`` reuses a handle from :meth:`create_cache`: the cached prefix
        (system + messages + tools it was built from) is not resent, so pass only
        the dynamic tail in ``messages``.

        ``on_stream(delta)`` is called with each
        :class:`~inferenceswitch.StreamDelta` while the reply streams: its
        thinking (``kind="thinking"``) where the provider shows it
        (:attr:`Capability.STREAMED_THINKING`), and its text
        (``kind="text"``). The response returned is the same as without it. A
        provider or model that shows no thinking sends none; that is never an
        error. On Anthropic, a call with ``effort`` and ``on_stream`` asks for
        summarized thinking, since some models omit it by default; on Gemini
        it asks for thought summaries. An exception from the callback
        propagates and ends the call.
        """
        resolution = self.resolve(model, provider)
        if tools:
            self.require(
                Capability.TOOL_CALLING, resolution.spec.name, model=resolution.model
            )
        if effort is not None:
            self.require(
                Capability.REASONING_EFFORT, resolution.spec.name, model=resolution.model
            )
        if _has_cache_markers(messages, tools):
            # Derived from the markers themselves — fail closed on a provider
            # whose caching isn't inline (Gemini's is out-of-band; OpenAI's is
            # automatic and unaddressable) rather than dropping the marker.
            self.require(
                Capability.EXPLICIT_PROMPT_CACHING, resolution.spec.name, model=resolution.model
            )
        if cache is not None:
            self.require(
                Capability.REUSABLE_PROMPT_CACHE, resolution.spec.name, model=resolution.model
            )
            if cache.provider != resolution.spec.name:
                raise InferenceSwitchError(
                    f"Cache handle was created for provider {cache.provider!r} but "
                    f"this request routes to {resolution.spec.name!r}; a handle is "
                    "not portable across providers."
                )
            if tools or system:
                # Uniform contract across providers: the cached prefix supplies
                # the system prompt and tools (Gemini physically can't take them
                # again alongside a cache). Pass only the dynamic tail here.
                raise InferenceSwitchError(
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
            temperature=self._temperature(resolution, temperature),
            max_tokens=max_tokens,
            **_streaming(on_stream),
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
        :class:`~inferenceswitch.CacheHandle` to pass to :meth:`chat` as ``cache=``.

        For a large static prefix (system prompt, few-shot context, tool set)
        reused across many requests: create the cache once, then reference the
        handle so the prefix is neither resent nor re-billed on each call. Routing
        to a provider without REUSABLE_PROMPT_CACHE raises
        :class:`UnsupportedCapabilityError`. The returned handle names a billable,
        TTL-bound resource the caller owns — reuse (and its cost) is explicit.
        """
        resolution = self.resolve(model, provider)
        self.require(
            Capability.REUSABLE_PROMPT_CACHE, resolution.spec.name, model=resolution.model
        )
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

    @overload
    def run_tools(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        handlers: dict[str, Callable[[dict], Any]],
        provider: str | None = None,
        api_key: str | None = None,
        system: str | None = None,
        effort: Effort | None = None,
        cache: CacheHandle | None = None,
        max_turns: int = 10,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        terminal_tools: Collection[str] | None = None,
        stop_when: Callable[[ToolCall, Any], bool] | None = None,
        on_response: Callable[[Response, float], None] | None = None,
        on_tool_call: Callable[[ToolCall, ToolResult, float], None] | None = None,
        on_stream: StreamCallback | None = None,
        return_run: Literal[False] = False,
    ) -> Response: ...

    @overload
    def run_tools(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        handlers: dict[str, Callable[[dict], Any]],
        provider: str | None = None,
        api_key: str | None = None,
        system: str | None = None,
        effort: Effort | None = None,
        cache: CacheHandle | None = None,
        max_turns: int = 10,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        terminal_tools: Collection[str] | None = None,
        stop_when: Callable[[ToolCall, Any], bool] | None = None,
        on_response: Callable[[Response, float], None] | None = None,
        on_tool_call: Callable[[ToolCall, ToolResult, float], None] | None = None,
        on_stream: StreamCallback | None = None,
        return_run: Literal[True],
    ) -> ToolRun: ...

    def run_tools(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        handlers: dict[str, Callable[[dict], Any]],
        provider: str | None = None,
        api_key: str | None = None,
        system: str | None = None,
        effort: Effort | None = None,
        cache: CacheHandle | None = None,
        max_turns: int = 10,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        terminal_tools: Collection[str] | None = None,
        stop_when: Callable[[ToolCall, Any], bool] | None = None,
        on_response: Callable[[Response, float], None] | None = None,
        on_tool_call: Callable[[ToolCall, ToolResult, float], None] | None = None,
        on_stream: StreamCallback | None = None,
        return_run: bool = False,
    ) -> Response | ToolRun:
        """Drive the full tool-use loop: call the model, run the requested tools
        via ``handlers`` (``{tool_name: fn(input) -> result}``), feed results
        back, and repeat until the model stops calling tools.

        Returns the final :class:`Response`. A handler exception, or a call to
        an unmapped tool, is returned to the model as an error tool result rather
        than raised — so the model can recover. Raises :class:`InferenceSwitchError`
        only if ``max_turns`` is exceeded.

        Optional, and without effect when omitted:

        * ``terminal_tools`` / ``stop_when`` end the loop early. After the
          model's tool calls in a round have all run, the loop stops if any of
          them succeeded and either named a tool in ``terminal_tools`` or made
          ``stop_when(call, output)`` true (``output`` is the handler's return
          value, before it is turned into a string). Calls that error, or have
          no handler, never stop the loop, so the model can retry them. The
          rest of the round still runs, so every call has a result and the
          history stays valid to continue from. The returned response is the
          one that made the stopping call.
        * ``on_response(response, seconds)`` is called after each model call,
          and ``on_tool_call(call, result, seconds)`` after each tool call
          (``seconds`` is 0.0 for a call with no handler), so a host can trace
          the loop as it runs. An exception from a hook propagates.
        * ``on_stream`` is passed to every model call, as on :meth:`chat`. Each
          call's deltas come before its ``on_response``, so a host showing
          "thinking so far" starts afresh after each ``on_response``.
        * ``cache`` is forwarded to every :meth:`chat` call, under the same
          rules: the provider needs REUSABLE_PROMPT_CACHE, and the cache's
          prefix supplies the system prompt and tools, so pass neither
          ``system`` nor ``tools`` with it.
        * ``return_run=True`` returns a :class:`ToolRun` instead: the final
          response, the full message history, usage summed over every model
          call, the number of calls, and the call that stopped the loop.
        """
        if tools is None and cache is None:
            raise ValueError("run_tools needs tools=, or a cache= whose prefix holds them.")
        terminal = frozenset(terminal_tools or ())
        history = list(messages)
        usage = Usage()
        for turn in range(1, max_turns + 1):
            started = time.perf_counter()
            response = self.chat(
                model=model,
                messages=history,
                tools=tools,
                system=system,
                effort=effort,
                cache=cache,
                provider=provider,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **_streaming(on_stream),
            )
            if on_response is not None:
                on_response(response, time.perf_counter() - started)
            usage = usage + response.usage
            if not response.tool_calls:
                if not return_run:
                    return response
                if response.text:
                    history.append(Message(role="assistant", content=response.text))
                return ToolRun(response, history, usage, turn)

            # Record the assistant's turn (text + the tool calls it made)...
            assistant_parts: list = [ToolUse(c.id, c.name, c.input) for c in response.tool_calls]
            if response.text:
                assistant_parts.insert(0, Text(response.text))
            history.append(Message(role="assistant", content=assistant_parts))

            # ...then run each tool and return the results in one user turn.
            results: list = []
            stopped_by: ToolCall | None = None
            for call in response.tool_calls:
                handler = handlers.get(call.name)
                if handler is None:
                    result = ToolResult(call.id, f"No handler for tool {call.name!r}.",
                                        is_error=True, name=call.name)
                    results.append(result)
                    if on_tool_call is not None:
                        on_tool_call(call, result, 0.0)
                    continue
                started = time.perf_counter()
                try:
                    output = handler(call.input)
                    result = ToolResult(call.id, str(output), name=call.name)
                except Exception as exc:  # noqa: BLE001 - report back to the model
                    result = ToolResult(call.id, f"Tool error: {exc}", is_error=True, name=call.name)
                elapsed = time.perf_counter() - started
                results.append(result)
                if on_tool_call is not None:
                    on_tool_call(call, result, elapsed)
                if result.is_error:
                    continue
                stops = call.name in terminal
                if stop_when is not None:
                    stops = stop_when(call, output) or stops
                if stops and stopped_by is None:
                    stopped_by = call
            history.append(Message(role="user", content=results))

            if stopped_by is not None:
                if not return_run:
                    return response
                return ToolRun(response, history, usage, turn, stopped_by)

        raise InferenceSwitchError(f"run_tools exceeded max_turns={max_turns} without finishing.")

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
