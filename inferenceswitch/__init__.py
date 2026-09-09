"""inferenceswitch — lightweight, import-only multi-provider LLM dispatch.

    from inferenceswitch import Client

    client = Client()
    data = client.generate_structured_json(
        model="claude-opus-4-8",          # or "gemini-3.5-flash", "groq/llama-3.3-70b", ...
        schema=MyModel.model_json_schema(),
        prompt="...",
        system="...",
    )

No proxy, no server, no dynamic provider loading — just a static registry and a
thick client that handles per-provider structured-output mechanics for you.
"""
from __future__ import annotations

from .adapters.anthropic import (
    CLAUDE_MAX_OUTPUT_TOKENS,
    UNKNOWN_MODEL_MAX_OUTPUT_TOKENS,
    claude_max_output_tokens,
)
from .adapters.gemini import (
    GEMINI_MAX_OUTPUT_TOKENS,
    UNKNOWN_GEMINI_MODEL_MAX_OUTPUT_TOKENS,
    gemini_max_output_tokens,
)
from .capabilities import Capabilities, Capability, SchemaDialect, StructuredMode
from .client import Client, ModelChoice, Resolution, env_key_provider
from .messages import (
    CacheHandle,
    Effort,
    Response,
    Message,
    StopReason,
    Text,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolUse,
    Usage,
    assistant,
    system,
    tool_result,
    user,
)
from .errors import (
    InferenceSwitchError,
    MissingAPIKeyError,
    MissingDependencyError,
    ProviderResolutionError,
    StructuredOutputError,
    UnsupportedCapabilityError,
)
from .registry import ProviderSpec, Registry, default_registry

__version__ = "0.1.0"

__all__ = [
    "Client",
    "ModelChoice",
    "Resolution",
    "env_key_provider",
    "Registry",
    "ProviderSpec",
    "default_registry",
    "Capabilities",
    "Capability",
    "StructuredMode",
    "SchemaDialect",
    "InferenceSwitchError",
    "ProviderResolutionError",
    "MissingAPIKeyError",
    "MissingDependencyError",
    "StructuredOutputError",
    "UnsupportedCapabilityError",
    # chat / tool-calling surface
    "Message",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolUse",
    "ToolResult",
    "Text",
    "CacheHandle",
    "Effort",
    "Response",
    "StopReason",
    "Usage",
    # Per-provider output caps (default = the model's own maximum). Editing a
    # table imposes a house ceiling process-wide; a per-call max_tokens still wins.
    "CLAUDE_MAX_OUTPUT_TOKENS",
    "UNKNOWN_MODEL_MAX_OUTPUT_TOKENS",
    "claude_max_output_tokens",
    "GEMINI_MAX_OUTPUT_TOKENS",
    "UNKNOWN_GEMINI_MODEL_MAX_OUTPUT_TOKENS",
    "gemini_max_output_tokens",
    "user",
    "assistant",
    "system",
    "tool_result",
    "__version__",
]
