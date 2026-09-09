"""Adapter registry: maps a provider ``kind`` to its adapter class."""
from __future__ import annotations

from ..registry import KIND_ANTHROPIC, KIND_GEMINI, KIND_OPENAI, ProviderSpec
from .anthropic import AnthropicAdapter
from .base import Adapter
from .gemini import GeminiAdapter
from .openai_compat import OpenAICompatibleAdapter

_ADAPTERS_BY_KIND: dict[str, type[Adapter]] = {
    KIND_OPENAI: OpenAICompatibleAdapter,
    KIND_ANTHROPIC: AnthropicAdapter,
    KIND_GEMINI: GeminiAdapter,
}


def build_adapter(spec: ProviderSpec, api_key: str, *, raw_client=None) -> Adapter:
    try:
        cls = _ADAPTERS_BY_KIND[spec.kind]
    except KeyError:
        raise ValueError(
            f"No adapter registered for provider kind {spec.kind!r} "
            f"(provider {spec.name!r})."
        ) from None
    return cls(spec, api_key, raw_client=raw_client)


__all__ = [
    "Adapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "OpenAICompatibleAdapter",
    "build_adapter",
]
