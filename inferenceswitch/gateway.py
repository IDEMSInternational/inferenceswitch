"""A gateway route: hosted providers' calls go to a deployment's gateway, with a
token in place of the key (#50).

A deployment that must not hand provider keys to the code it runs (a container per
person, say) gives that code a :class:`Gateway` instead. The gateway holds the real
keys: it takes a call made with the token, puts the key in, and forwards it to the
provider unchanged. The client sends each routed provider's calls to
``<url>/<provider><the path of the provider's own address>``::

    Client(gateway=Gateway(url="http://gateway:8001/relay", token=token))
    # anthropic -> http://gateway:8001/relay/anthropic/v1/messages
    # gemini    -> http://gateway:8001/relay/gemini/v1beta/models/...
    # openai    -> http://gateway:8001/relay/openai/v1/chat/completions
    # mistral   -> http://gateway:8001/relay/mistral/v1/chat/completions

The token goes where the provider's SDK puts its key, so every request and reply
is the provider's own. A routed provider's ``key_provider`` is never called, so no
real key is sent to the gateway, unless the deployment sets ``own_keys=True``:
then a per-call ``api_key=`` (a person's own key) goes to the gateway as it came.

A route is set only by the code that builds the :class:`~inferenceswitch.Client`.
Model strings and per-call arguments can't set or change one.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .registry import ProviderSpec


@dataclass(frozen=True)
class Gateway:
    """Where a deployment's hosted providers' calls go, and the token they carry."""

    #: The gateway's address, under which each provider has its own path
    #: (``<url>/<provider>``). ``http`` or ``https``, with no query, fragment or
    #: credentials.
    url: str
    #: Sent in place of each routed provider's key. Never shown in ``repr``.
    token: str = field(repr=False)
    #: The providers routed, by registry name; ``None`` routes every hosted one.
    #: A local server (``ollama``, ``lmstudio``) is never routed.
    providers: frozenset[str] | None = None
    #: Whether a per-call ``api_key=`` for a routed provider (a person's own key)
    #: goes to the gateway as it came. Off, such a call raises.
    own_keys: bool = False

    def __post_init__(self) -> None:
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"A gateway URL is http(s)://host[:port][/path], not {self.url!r}.")
        if parts.query or parts.fragment or parts.username or parts.password:
            raise ValueError("A gateway URL takes no query, fragment or credentials.")
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("A gateway needs a token.")
        if self.providers is not None:
            providers = self.providers
            if isinstance(providers, str) or not isinstance(providers, Iterable):
                raise TypeError("Gateway providers is a collection of provider names.")
            object.__setattr__(self, "providers", frozenset(providers))

    def routes(self, spec: ProviderSpec) -> bool:
        """Whether calls to ``spec`` go through this gateway."""
        if spec.is_local:
            return False
        return self.providers is None or spec.name in self.providers

    def address(self, spec: ProviderSpec) -> str:
        """Where ``spec``'s SDK is pointed: ``<url>/<provider>`` and the path of
        the provider's own address (``/v1`` for OpenAI, none for Anthropic)."""
        path = urlsplit(spec.address).path.rstrip("/")
        return f"{self.url.rstrip('/')}/{spec.name}{path}"
