"""Exception hierarchy for inferenceswitch.

All errors derive from :class:`InferenceSwitchError`, so a caller can catch the whole
library with one clause while still discriminating on the specific subclass. No
error ever carries an API key in its message — keys stay out of tracebacks by
design (one of the security properties that keeps this library "lighter than
liteLLM").
"""
from __future__ import annotations


class InferenceSwitchError(Exception):
    """Base class for every error raised by inferenceswitch."""


class ProviderResolutionError(InferenceSwitchError):
    """A model string could not be mapped to exactly one registered provider.

    Raised when a bare model name matches no provider (or more than one) and no
    explicit ``provider=`` was given. There is deliberately no silent fallback
    to a default provider — an unroutable request is a hard error.
    """


class MissingAPIKeyError(InferenceSwitchError):
    """The env var (or key provider) for the resolved provider yielded no key."""


class MissingDependencyError(InferenceSwitchError):
    """The provider's SDK is not installed.

    Provider SDKs are optional extras; this is raised (with the exact
    ``pip install`` hint) the first time an adapter needs an SDK that isn't
    importable, rather than at import time.
    """


class UnsupportedCapabilityError(InferenceSwitchError):
    """A workflow asked a provider for a capability it does not have.

    Raised by :meth:`Client.require` (and intended for callers to raise via
    their own ``supports`` checks) so unsupported model-specific features fail
    loudly and predictably rather than silently degrading.
    """


class UnsupportedSchemaError(InferenceSwitchError, ValueError):
    """A caller's JSON schema cannot be expressed in a provider's dialect.

    Raised before any request is sent, e.g. for a recursive schema or a
    free-form object on Anthropic structured outputs, which supports neither.
    """


class StructuredOutputError(InferenceSwitchError):
    """The model did not return usable schema-constrained JSON.

    Covers truncation (hit the output-token cap mid-object), a refusal, a model
    that did not call the structured-output tool, JSON that would not parse, and
    JSON that parsed but does not match the schema.
    ``context`` carries provider/model/stop-reason detail for logging.
    """

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.context = context or {}
