"""A stream that goes silent fails in bounded time (waypoint-anastom#144)."""
from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("anthropic")

from inferenceswitch.adapters import anthropic as adapter_mod
from inferenceswitch.adapters.anthropic import AnthropicAdapter


def test_the_client_reads_with_the_stream_timeout_not_the_sdks_ten_minutes():
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    client = adapter._build_client("sk-ant-fake")
    assert client.timeout.read == adapter_mod.STREAM_READ_TIMEOUT
    assert client.timeout.connect == 5.0


def test_a_silent_server_raises_instead_of_hanging(monkeypatch):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    held = []

    def accept():
        conn, _ = server.accept()
        held.append(conn)
        conn.recv(65536)  # take the request, then say nothing

    threading.Thread(target=accept, daemon=True).start()
    monkeypatch.setattr(adapter_mod, "STREAM_READ_TIMEOUT", 1.0)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.getsockname()[1]}")
    import anthropic

    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    client = adapter._build_client("sk-ant-fake").with_options(max_retries=0)
    started = time.monotonic()
    messages = [{"role": "user", "content": "hi"}]
    with (
        pytest.raises(anthropic.APITimeoutError),
        client.messages.stream(model="x", max_tokens=5, messages=messages) as stream,
    ):
        stream.get_final_message()
    assert time.monotonic() - started < 10
    server.close()
