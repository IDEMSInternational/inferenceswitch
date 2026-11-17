"""A stream that goes silent fails in bounded time (waypoint-anastom#144), per provider.

Each test points the real SDK at a local server that accepts the request and then says
nothing, with the timeout shortened to a second. The error is the SDK's own, which is what
anastom and WayPoint already recognise as a lost connection."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from inferenceswitch.adapters import anthropic as anthropic_mod
from inferenceswitch.adapters import base as base_mod
from inferenceswitch.adapters import gemini as gemini_mod
from inferenceswitch.adapters import openai_compat as openai_mod
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


def test_one_limit_is_shared_by_the_providers():
    assert anthropic_mod.STREAM_READ_TIMEOUT == base_mod.STREAM_READ_TIMEOUT == 120.0
    assert openai_mod.STREAM_READ_TIMEOUT == gemini_mod.STREAM_READ_TIMEOUT == 120.0


def test_anthropic_reads_with_the_stream_timeout_not_the_sdks_ten_minutes():
    pytest.importorskip("anthropic")
    adapter = anthropic_mod.AnthropicAdapter.__new__(anthropic_mod.AnthropicAdapter)
    client = adapter._build_client("sk-ant-fake")
    assert client.timeout.read == anthropic_mod.STREAM_READ_TIMEOUT
    assert client.timeout.connect == 5.0


def test_anthropic_silent_stream_raises(monkeypatch, silent_server):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setattr(anthropic_mod, "STREAM_READ_TIMEOUT", 1.0)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", silent_server)
    adapter = anthropic_mod.AnthropicAdapter.__new__(anthropic_mod.AnthropicAdapter)
    client = adapter._build_client("sk-ant-fake").with_options(max_retries=0)
    messages = [{"role": "user", "content": "hi"}]
    started = time.monotonic()
    with (
        pytest.raises(anthropic.APITimeoutError),
        client.messages.stream(model="x", max_tokens=5, messages=messages) as stream,
    ):
        stream.get_final_message()
    assert time.monotonic() - started < 10


def test_openai_silent_stream_raises(monkeypatch, silent_server):
    openai = pytest.importorskip("openai")
    monkeypatch.setattr(openai_mod, "STREAM_READ_TIMEOUT", 1.0)
    monkeypatch.setenv("OPENAI_BASE_URL", silent_server + "/v1")
    adapter = openai_mod.OpenAICompatibleAdapter(default_registry().get("openai"), "sk-fake")
    client = adapter._streaming_client().with_options(max_retries=0)
    started = time.monotonic()
    with pytest.raises(openai.APITimeoutError):
        for _ in client.chat.completions.create(
            model="x", messages=[{"role": "user", "content": "hi"}], stream=True
        ):
            pass
    assert time.monotonic() - started < 10


def test_openai_leaves_a_local_server_alone():
    pytest.importorskip("openai")
    adapter = openai_mod.OpenAICompatibleAdapter(default_registry().get("ollama"), "x")
    assert adapter._streaming_client() is adapter._client


def test_gemini_silent_stream_raises(monkeypatch, silent_server):
    pytest.importorskip("google.genai")
    import httpx

    monkeypatch.setattr(gemini_mod, "STREAM_READ_TIMEOUT", 1.0)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", silent_server)
    adapter = gemini_mod.GeminiAdapter(default_registry().get("gemini"), "fake")
    started = time.monotonic()
    types = gemini_mod.require_sdk("google.genai", "gemini").types
    with pytest.raises(httpx.ReadTimeout):
        adapter._generate(
            lambda delta: None,
            model="gemini-3.8-flash",
            contents="hi",
            config=types.GenerateContentConfig(),
        )
    assert time.monotonic() - started < 10
