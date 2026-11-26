"""A gateway route (#50): a client built with ``Gateway(url, token)`` sends each hosted
provider's calls to ``<url>/<provider><the provider's own path>`` with the token where the
key goes, and never sends a real key there.

The gateway here is a fake on 127.0.0.1 that answers as each provider does and records
every request. The environment holds "real" keys, credentials and base-URL overrides
throughout, so a test fails if any of them reaches the gateway or redirects a call."""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inferenceswitch import Client, Gateway, GatewayKeyError, default_registry

TOKEN = "gw-token-123"
REAL = "REAL-SECRET"
REPLY = "hello from the gateway"

#: Everything in the environment an SDK might send or be redirected by.
ENVIRONMENT = {
    "ANTHROPIC_API_KEY": f"{REAL}-anthropic",
    "ANTHROPIC_AUTH_TOKEN": f"{REAL}-anthropic-auth",
    "GEMINI_API_KEY": f"{REAL}-gemini",
    "GOOGLE_API_KEY": f"{REAL}-google",
    "OPENAI_API_KEY": f"{REAL}-openai",
    "OPENAI_ADMIN_KEY": f"{REAL}-openai-admin",
    "MISTRAL_API_KEY": f"{REAL}-mistral",
    "GOOGLE_GENAI_USE_VERTEXAI": "true",
}
BASE_URL_VARIABLES = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "GOOGLE_GEMINI_BASE_URL")

_ANTHROPIC_EVENTS = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": REPLY}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta",
                       "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 4}}),
    ("message_stop", {"type": "message_stop"}),
]
_OPENAI_REPLY = {
    "id": "c1", "object": "chat.completion", "created": 1, "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
}
_GEMINI_REPLY = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": REPLY}]},
                    "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4, "totalTokenCount": 9},
    "modelVersion": "gemini-3.8-flash",
}


class _Recorder(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    server: _Recorder

    def log_message(self, *args) -> None:
        pass

    def _record(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else ""
        seen = {
            "method": self.command,
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": body,
        }
        self.server.requests.append(seen)
        return seen

    def _send(self, status: int, content_type: str, payload: str) -> None:
        data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._record()
        self._send(404, "application/json", '{"error": "not here"}')

    def do_POST(self) -> None:
        path = self._record()["path"].split("?")[0]
        if path.endswith("/v1/messages"):
            events = "".join(
                f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in _ANTHROPIC_EVENTS
            )
            self._send(200, "text/event-stream", events)
        elif path.endswith("/chat/completions"):
            self._send(200, "application/json", json.dumps(_OPENAI_REPLY))
        elif path.endswith(":streamGenerateContent"):
            self._send(200, "text/event-stream", f"data: {json.dumps(_GEMINI_REPLY)}\r\n\r\n")
        elif path.endswith(":generateContent"):
            self._send(200, "application/json", json.dumps(_GEMINI_REPLY))
        else:
            self._send(404, "application/json", '{"error": "not here"}')


def _serve():
    server = _Recorder()
    threading.Thread(target=server.serve_forever, args=(0.02,), daemon=True).start()
    return server


@pytest.fixture
def gateway():
    server = _serve()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def hostile_environment(monkeypatch):
    """Real-looking keys in the environment, and every SDK's base-URL variable pointed at
    a trap that records anything sent to it."""
    trap = _serve()
    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    for name in BASE_URL_VARIABLES:
        monkeypatch.setenv(name, trap.url + ("/v1" if name.startswith("OPENAI") else ""))
    yield trap
    trap.shutdown()
    trap.server_close()
    assert trap.requests == [], "an SDK followed its base-URL environment variable"


def _no_real_key(server: _Recorder) -> None:
    for seen in server.requests:
        assert REAL not in json.dumps(seen), seen


def _refuse(spec):
    raise AssertionError(f"key_provider was asked for {spec.name}'s key")


def _client(server, **gateway):
    return Client(key_provider=_refuse, gateway=Gateway(url=server.url + "/relay", token=TOKEN, **gateway))


# ── each adapter, through the gateway ─────────────────────────────────────────


def test_anthropic_goes_to_the_gateway_with_the_token(gateway):
    pytest.importorskip("anthropic")
    text = _client(gateway).generate_text(model="anthropic/claude-sonnet-5-5", prompt="hi")
    assert text == REPLY
    (seen,) = gateway.requests
    assert seen["path"] == "/relay/anthropic/v1/messages"
    headers = seen["headers"]
    assert TOKEN in (headers.get("x-api-key"), headers.get("authorization", "").removeprefix("Bearer "))
    body = json.loads(seen["body"])
    assert body["model"] == "claude-sonnet-5-5"
    assert body["messages"][0]["role"] == "user" and "max_tokens" in body and body["stream"]
    _no_real_key(gateway)


def test_gemini_goes_to_the_gateway_with_the_token(gateway):
    pytest.importorskip("google.genai")
    text = _client(gateway).generate_text(model="gemini/gemini-3.8-flash", prompt="hi")
    assert text == REPLY
    (seen,) = gateway.requests
    assert re.fullmatch(r"/relay/gemini/v1\w*/models/gemini-3\.8-flash:generateContent", seen["path"])
    assert seen["headers"]["x-goog-api-key"] == TOKEN
    assert json.loads(seen["body"])["contents"][0]["parts"][0]["text"] == "hi"
    _no_real_key(gateway)


def test_a_streamed_gemini_call_keeps_the_route(gateway):
    pytest.importorskip("google.genai")
    client = Client(
        key_provider=_refuse,
        gateway=Gateway(url=gateway.url + "/relay", token=TOKEN),
        stream_silence={"gemini": 30.0},  # a per-call http_options must not drop the address
    )
    deltas: list = []
    text = client.generate_text(model="gemini/gemini-3.8-flash", prompt="hi", on_stream=deltas.append)
    assert text == REPLY and deltas
    (seen,) = gateway.requests
    assert re.match(r"/relay/gemini/v1\w*/models/gemini-3\.8-flash:streamGenerateContent", seen["path"])
    assert seen["headers"]["x-goog-api-key"] == TOKEN
    _no_real_key(gateway)


@pytest.mark.parametrize(
    ("model", "path"),
    [
        ("openai/gpt-4.1-mini", "/relay/openai/v1/chat/completions"),
        ("mistral/mistral-small-2603", "/relay/mistral/v1/chat/completions"),
        ("groq/llama-4", "/relay/groq/openai/v1/chat/completions"),
    ],
)
def test_openai_compatible_providers_go_to_the_gateway_with_the_token(gateway, model, path):
    pytest.importorskip("openai")
    text = _client(gateway).generate_text(model=model, prompt="hi")
    assert text == REPLY
    (seen,) = gateway.requests
    assert seen["path"] == path
    assert seen["headers"]["authorization"] == f"Bearer {TOKEN}"
    body = json.loads(seen["body"])
    assert body["model"] == model.split("/", 1)[1]
    assert body["messages"][-1] == {"role": "user", "content": "hi"}
    _no_real_key(gateway)


# ── keys ──────────────────────────────────────────────────────────────────────


def test_an_own_key_is_refused_unless_the_gateway_takes_them(gateway):
    pytest.importorskip("openai")
    with pytest.raises(GatewayKeyError):
        _client(gateway).generate_text(model="openai/gpt-4.1-mini", prompt="hi", api_key="sk-own")
    with pytest.raises(GatewayKeyError):
        _client(gateway).verify_key("anthropic", api_key="sk-ant-own")
    assert gateway.requests == []


def test_an_own_key_goes_through_as_it_came_where_allowed(gateway):
    pytest.importorskip("openai")
    client = _client(gateway, own_keys=True)
    assert client.generate_text(model="openai/gpt-4.1-mini", prompt="hi", api_key="sk-own") == REPLY
    (seen,) = gateway.requests
    assert seen["headers"]["authorization"] == "Bearer sk-own"
    assert TOKEN not in json.dumps(seen)


def test_routed_providers_are_available_without_a_key(gateway):
    client = _client(gateway, providers={"anthropic"})
    available = {a.provider: a.available for a in client.availability(["anthropic", "openai"])}
    assert available == {"anthropic": True, "openai": False}


# ── what is routed, and what is not ───────────────────────────────────────────


def _address(client, provider):
    return client._adapter(default_registry().get(provider), None).base_url


def test_only_the_named_providers_are_routed(gateway):
    client = Client(gateway=Gateway(url="http://gw/relay", token=TOKEN, providers=["gemini"]))
    assert _address(client, "gemini") == "http://gw/relay/gemini"
    assert _address(client, "anthropic") == "https://api.anthropic.com"


def test_local_servers_are_never_routed():
    client = Client(gateway=Gateway(url="http://gw/relay", token=TOKEN))
    assert _address(client, "ollama") == "http://localhost:11434/v1"
    with pytest.raises(ValueError):
        Client(gateway=Gateway(url="http://gw", token=TOKEN, providers={"ollama"}))


@pytest.mark.parametrize(
    ("provider", "official"),
    [
        ("anthropic", "https://api.anthropic.com"),
        ("gemini", "https://generativelanguage.googleapis.com"),
        ("openai", "https://api.openai.com/v1"),
    ],
)
def test_without_a_gateway_every_sdk_is_given_the_official_address(provider, official):
    """The base-URL variables set by ``hostile_environment`` don't apply."""
    pytest.importorskip({"anthropic": "anthropic", "gemini": "google.genai"}.get(provider, "openai"))
    adapter = Client()._adapter(default_registry().get(provider), "k")
    assert adapter.base_url == official
    if provider == "gemini":
        api = adapter.raw_client._api_client
        assert api._http_options.base_url.rstrip("/") == official and not api.vertexai
    else:
        assert str(adapter.raw_client.base_url).rstrip("/") == official


def test_a_model_string_cannot_name_a_gateway():
    from inferenceswitch import ProviderResolutionError

    with pytest.raises((ProviderResolutionError, KeyError)):
        Client().resolve("http://evil.example/anthropic/claude-sonnet-5-5")


# ── the Gateway itself ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    ["ftp://gw", "gw:8001", "http://", "http://gw?x=1", "http://gw#f", "http://user:pw@gw"],
)
def test_a_gateway_url_must_be_plain_http(url):
    with pytest.raises(ValueError):
        Gateway(url=url, token=TOKEN)


def test_a_gateway_needs_a_token_and_never_shows_it():
    with pytest.raises(ValueError):
        Gateway(url="http://gw", token="")
    assert TOKEN not in repr(Gateway(url="http://gw", token=TOKEN))


def test_the_address_keeps_the_providers_own_path():
    reg = default_registry()
    gw = Gateway(url="http://gw:8001/hosting/relay/", token=TOKEN)
    assert gw.address(reg["anthropic"]) == "http://gw:8001/hosting/relay/anthropic"
    assert gw.address(reg["openai"]) == "http://gw:8001/hosting/relay/openai/v1"
    assert gw.address(reg["gemini"]) == "http://gw:8001/hosting/relay/gemini"
    assert gw.address(reg["openrouter"]) == "http://gw:8001/hosting/relay/openrouter/api/v1"
