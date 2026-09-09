"""Adapter contract + shared helpers.

An adapter owns exactly one provider *kind* and translates the library's
uniform calls into that kind's SDK. Everything provider-specific (the structured
-output mechanism, schema dialect, key smoke-test) lives behind this interface,
so the client facade never branches on provider name.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

from ..errors import MissingDependencyError, UnsupportedCapabilityError
from ..messages import CacheHandle, Effort, Response, Message, Tool, ToolChoice
from ..registry import ProviderSpec


def split_system(messages: list[Message], system: str | None) -> tuple[str | None, list[Message]]:
    """Pull system text out of the message list and merge it with an explicit
    ``system=`` argument, returning ``(combined_system, non_system_messages)``.

    Providers put the system prompt in different places (a message vs a top-level
    field), so the adapters work from this single extracted string.
    """
    from ..messages import Text  # local import to avoid a cycle at module load

    system_chunks: list[str] = []
    if system:
        system_chunks.append(system)
    rest: list[Message] = []
    for message in messages:
        if message.role == "system":
            system_chunks.extend(
                p.text for p in message.parts() if isinstance(p, Text)
            )
        else:
            rest.append(message)
    combined = "\n\n".join(system_chunks) if system_chunks else None
    return combined, rest


def require_sdk(module: str, extra: str):
    """Import a provider SDK lazily, or raise a clear install hint."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"The {module!r} SDK is required for this provider but is not "
            f'installed. Install it with:  pip install "inferenceswitch[{extra}]"'
        ) from exc


class Adapter(ABC):
    """Drives one provider kind."""

    def __init__(self, spec: ProviderSpec, api_key: str, *, raw_client: Any = None) -> None:
        self.spec = spec
        self.capabilities = spec.capabilities
        self._api_key = api_key
        # ``raw_client`` lets callers/tests inject a pre-built SDK client (mirrors
        # the dependency-injection seam used for BYOK and offline tests).
        self._client = raw_client if raw_client is not None else self._build_client(api_key)

    @property
    def raw_client(self) -> Any:
        """The underlying provider SDK client.

        The escape hatch for provider-specific features inferenceswitch doesn't wrap
        yet — a workflow can call the native SDK directly while still getting
        this library's routing, key resolution, and adapter caching.
        """
        return self._client

    @abstractmethod
    def _build_client(self, api_key: str) -> Any: ...

    @abstractmethod
    def generate_structured_json(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict,
        system: str | None = None,
        tool_name: str = "generate_json",
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Any:
        """Return schema-constrained JSON as a native Python object."""

    @abstractmethod
    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Return a plain text completion."""

    @abstractmethod
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
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Response:
        """A normalized, optionally tool-using turn. Returns an :class:`Response`
        with any tool calls the model requested, unified across providers.

        ``cache`` references a reusable prompt cache created by
        :meth:`create_cache`; only providers with REUSABLE_PROMPT_CACHE honor it
        (the client gates this, so other adapters never receive a non-None cache)."""

    def create_cache(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[Tool] | None = None,
        ttl: int | None = None,
    ) -> CacheHandle:
        """Create a reusable server-side prompt cache and return a handle to it.

        Overridden only by adapters whose provider supports REUSABLE_PROMPT_CACHE;
        the base raises so a mis-routed call fails loudly instead of silently.
        """
        raise UnsupportedCapabilityError(
            f"Provider {self.spec.name!r} has no reusable prompt cache."
        )

    def delete_cache(self, handle: CacheHandle) -> None:
        """Release a cache created by :meth:`create_cache`. The base no-ops —
        replay-backed handles hold no server resource. Adapters whose handles are
        server-backed override this to delete the resource."""

    @abstractmethod
    def verify_key(self) -> tuple[bool, str]:
        """Cheapest possible authenticated call. Returns ``(ok, detail)`` and
        never raises — a bad key is an expected outcome, not an error."""
