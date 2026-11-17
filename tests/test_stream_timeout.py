"""A stream that goes silent fails in bounded time when asked to (waypoint-anastom#144).

``Client(stream_silence={provider: seconds})`` sets, per provider, how long a streamed call may
go without a byte. Each provider is run with the setting on, against a local server that
accepts the request and then says nothing, and with it off, where the SDK's own default
stands. The error is the SDK's own, which anastom and WayPoint recognise as a lost connection."""
from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import pytest

from inferenceswitch import Client
from inferenceswitch.adapters.anthropic import AnthropicAdapter
from inferenceswitch.adapters.gemini import GeminiAdapter
from inferenceswitch.adapters.openai_compat import OpenAICompatibleAdapter
from inferenceswitch.registry import default_registry


@pytest.fixture
def silent_server():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    held: list[socket.socket] = []

    def accept():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            held.append(conn)
            conn.recv(65536)  # take the request, then say nothing

    threading.Thread(target=accept, daemon=True).start()
    yield f"http://127.0.0.1:{server.getsockname()[1]}"
    server.close()
    for conn in held:
        conn.close()


def spec(name):
    return default_registry().get(name)


def test_the_client_hands_each_provider_its_own_setting():
    client = Client(stream_silence={"anthropic": 120.0, "ollama": None})
    assert client._adapter(spec("anthropic"), "k").stream_silence == 120.0
    assert client._adapter(spec("ollama"), "k").stream_silence is None
    assert client._adapter(spec("openai"), "k").stream_silence is None
    assert Client()._adapter(spec("anthropic"), "k").stream_silence is None


# ── Anthropic ─────────────────────────────────────────────────────────────────


def test_anthropic_off_keeps_the_sdks_default():
    pytest.importorskip("anthropic")
    adapter = AnthropicAdapter(spec("anthropic"), "sk-ant-fake")
    assert adapter._client.timeout.read == 600


def test_anthropic_on_silent_stream_raises(monkeypatch, silent_server):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", silent_server)
    adapter = AnthropicAdapter(spec("anthropic"), "sk-ant-fake", stream_silence=1.0)
    assert adapter._client.timeout.read == 1.0
    client = adapter._client.with_options(max_retries=0)
    messages = [{"role": "user", "content": "hi"}]
    started = time.monotonic()
    with (
        pytest.raises(anthropic.APITimeoutError),
        client.messages.stream(model="x", max_tokens=5, messages=messages) as stream,
    ):
        stream.get_final_message()
    assert time.monotonic() - started < 10


# ── OpenAI-compatible ─────────────────────────────────────────────────────────


def test_openai_off_keeps_the_sdks_default():
    pytest.importorskip("openai")
    adapter = OpenAICompatibleAdapter(spec("openai"), "sk-fake")
    assert adapter._streaming_client() is adapter._client
    assert adapter._client.timeout.read == 600


def test_openai_on_silent_stream_raises(monkeypatch, silent_server):
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_BASE_URL", silent_server + "/v1")
    adapter = OpenAICompatibleAdapter(spec("openai"), "sk-fake", stream_silence=1.0)
    client = adapter._streaming_client().with_options(max_retries=0)
    started = time.monotonic()
    with pytest.raises(openai.APITimeoutError):
        for _ in client.chat.completions.create(
            model="x", messages=[{"role": "user", "content": "hi"}], stream=True
        ):
            pass
    assert time.monotonic() - started < 10


def test_a_local_openai_compatible_server_takes_its_own_setting():
    pytest.importorskip("openai")
    long = OpenAICompatibleAdapter(spec("ollama"), "x", stream_silence=900.0)
    assert long._streaming_client().timeout.read == 900.0
    assert OpenAICompatibleAdapter(spec("ollama"), "x")._streaming_client().timeout.read == 600


# ── Gemini ────────────────────────────────────────────────────────────────────


def _fake_gemini(seen):
    def stream(*, model, contents, config):
        seen.append(config)
        return iter(())

    return SimpleNamespace(models=SimpleNamespace(generate_content_stream=stream))


@pytest.mark.parametrize(("setting", "expected"), [(None, None), (1.0, 1000)])
def test_gemini_passes_the_setting_to_a_streamed_call_only(setting, expected):
    pytest.importorskip("google.genai")
    from inferenceswitch.errors import InferenceSwitchError

    seen: list = []
    adapter = GeminiAdapter(
        spec("gemini"), "x", raw_client=_fake_gemini(seen), stream_silence=setting
    )
    types = pytest.importorskip("google.genai").types
    with pytest.raises(InferenceSwitchError):  # the fake stream is empty
        adapter._generate(
            lambda d: None, model="m", contents="hi", config=types.GenerateContentConfig()
        )
    options = seen[0].http_options
    assert (options.timeout if options else None) == expected


def test_gemini_on_silent_stream_raises(monkeypatch, silent_server):
    pytest.importorskip("google.genai")
    import httpx

    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", silent_server)
    adapter = GeminiAdapter(spec("gemini"), "fake", stream_silence=1.0)
    types = pytest.importorskip("google.genai").types
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        adapter._generate(
            lambda delta: None,
            model="gemini-3.8-flash",
            contents="hi",
            config=types.GenerateContentConfig(),
        )
    assert time.monotonic() - started < 10
