"""on_stream: thinking and text handed over while a reply streams.

Offline. Each adapter is driven with a fake SDK client that replays a recorded
shape of stream: Anthropic content-block events, Gemini ``GenerateContentResponse``
chunks, OpenAI ``ChatCompletionChunk``s (built from the SDKs' own types where the
adapter reads them back). The assertions are that the deltas arrive in order,
and that what each call returns is the same with a callback as without.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from inferenceswitch import (
    Capability,
    Client,
    Effort,
    StreamDelta,
    Tool,
    user,
)
from inferenceswitch.adapters.anthropic import AnthropicAdapter
from inferenceswitch.adapters.gemini import GeminiAdapter
from inferenceswitch.adapters.openai_compat import OpenAICompatibleAdapter
from inferenceswitch.registry import default_registry

ADD = Tool("add", "adds", {"type": "object", "properties": {"a": {"type": "integer"}}})


def _collect():
    deltas: list[StreamDelta] = []
    return deltas, deltas.append


def _joined(deltas: list[StreamDelta], kind: str) -> str:
    return "".join(d.text for d in deltas if d.kind == kind)


# ── Anthropic ─────────────────────────────────────────────────────────────────


def _delta(kind: str, **fields):
    return SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type=kind, **fields))


class _FakeAnthropic:
    """A stand-in anthropic.Anthropic. Each ``messages.stream`` call replays the
    next scripted (events, final message) pair; iterating the stream yields the
    events the way the SDK's MessageStream does, raw events among its own."""

    def __init__(self, *replies):
        self.calls: list[dict] = []
        self.iterated: list[bool] = []
        self.replies = list(replies)
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        events, message = self.replies.pop(0)
        iterated = self.iterated
        iterated.append(False)

        class _Stream:
            def __enter__(inner):
                return inner

            def __exit__(inner, *exc):
                return False

            def __iter__(inner):
                iterated[-1] = True
                return iter(events)

            def get_final_message(inner):
                return message

        return _Stream()


def _message(content, stop_reason="end_turn"):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )


THINKING_EVENTS = [
    SimpleNamespace(type="message_start"),
    SimpleNamespace(type="content_block_start", index=0),
    _delta("thinking_delta", thinking="Let me "),
    SimpleNamespace(type="thinking", thinking="Let me ", snapshot="Let me "),
    _delta("thinking_delta", thinking="add."),
    _delta("signature_delta", signature="sig"),
    SimpleNamespace(type="content_block_stop", index=0),
    _delta("text_delta", text="It is "),
    SimpleNamespace(type="text", text="It is ", snapshot="It is "),
    _delta("text_delta", text="3."),
    SimpleNamespace(type="message_stop"),
]
THOUGHT_MESSAGE = _message(
    [
        SimpleNamespace(type="thinking", thinking="Let me add.", signature="sig"),
        SimpleNamespace(type="text", text="It is 3."),
    ]
)


def _anthropic(*replies):
    fake = _FakeAnthropic(*replies)
    return AnthropicAdapter(default_registry().get("anthropic"), "", raw_client=fake), fake


def test_anthropic_chat_streams_thinking_then_text():
    adapter, fake = _anthropic((THINKING_EVENTS, THOUGHT_MESSAGE))
    deltas, on_stream = _collect()

    resp = adapter.chat(
        model="claude-opus-4-8", messages=[user("1+2?")], effort=Effort.HIGH, on_stream=on_stream
    )

    assert deltas == [
        StreamDelta("thinking", "Let me "),
        StreamDelta("thinking", "add."),
        StreamDelta("text", "It is "),
        StreamDelta("text", "3."),
    ]
    assert resp.text == "It is 3."
    assert resp.usage.output_tokens == 20
    # Summarized thinking is asked for, so a model that omits it by default
    # still has something to stream.
    assert fake.calls[0]["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_anthropic_without_callback_sends_and_returns_as_before():
    adapter, fake = _anthropic((THINKING_EVENTS, THOUGHT_MESSAGE), (THINKING_EVENTS, THOUGHT_MESSAGE))
    plain = adapter.chat(model="claude-opus-4-8", messages=[user("1+2?")], effort=Effort.HIGH)
    streamed = adapter.chat(
        model="claude-opus-4-8", messages=[user("1+2?")], effort=Effort.HIGH, on_stream=lambda d: None
    )

    assert fake.iterated == [False, True]
    assert fake.calls[0]["thinking"] == {"type": "adaptive"}
    assert {k: v for k, v in fake.calls[1].items() if k != "thinking"} == {
        k: v for k, v in fake.calls[0].items() if k != "thinking"
    }
    assert (plain.text, plain.tool_calls, plain.stop_reason, plain.usage) == (
        streamed.text, streamed.tool_calls, streamed.stop_reason, streamed.usage
    )


def test_anthropic_without_effort_leaves_thinking_unset():
    adapter, fake = _anthropic(([_delta("text_delta", text="hi")], _message([SimpleNamespace(type="text", text="hi")])))
    deltas, on_stream = _collect()
    adapter.chat(model="claude-opus-4-8", messages=[user("hi")], on_stream=on_stream)
    assert "thinking" not in fake.calls[0]
    assert deltas == [StreamDelta("text", "hi")]


def test_anthropic_generate_text_streams():
    adapter, _ = _anthropic((THINKING_EVENTS, THOUGHT_MESSAGE))
    deltas, on_stream = _collect()
    assert adapter.generate_text(model="claude-opus-4-8", prompt="1+2?", on_stream=on_stream) == "It is 3."
    assert _joined(deltas, "thinking") == "Let me add."
    assert _joined(deltas, "text") == "It is 3."


def test_anthropic_cached_chat_streams():
    adapter, fake = _anthropic((THINKING_EVENTS, THOUGHT_MESSAGE))
    handle = adapter.create_cache(model="claude-opus-4-8", system="BIG", messages=[])
    deltas, on_stream = _collect()
    adapter.chat(
        model="claude-opus-4-8", messages=[user("q")], cache=handle, effort=Effort.LOW,
        on_stream=on_stream,
    )
    assert _joined(deltas, "thinking") == "Let me add."
    assert fake.calls[0]["thinking"]["display"] == "summarized"


ANSWER = {"answer": 3, "why": "sum"}
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}, "why": {"type": "string"}},
    "required": ["answer", "why"],
}


def test_anthropic_structured_output_config_streams_the_json_as_text():
    text = json.dumps(ANSWER)
    events = [_delta("text_delta", text=text[:9]), _delta("text_delta", text=text[9:])]
    adapter, _ = _anthropic((events, _message([SimpleNamespace(type="text", text=text)])))
    deltas, on_stream = _collect()

    result = adapter.generate_structured(
        model="claude-opus-4-8", prompt="1+2?", schema=SCHEMA, on_stream=on_stream
    )

    assert result.value == ANSWER
    assert _joined(deltas, "text") == text


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_wrapped_tool_reply_is_unwrapped_with_or_without_streaming(stream):
    """The non-strict tool path (models before structured outputs) sometimes
    nests the answer under one stray key. The adapter unwraps it; streaming
    reads the same final message, so it unwraps it the same way, and the
    tool's input (input_json_delta) is not streamed as text."""
    wrapped = {"$PARAMETER_NAME": ANSWER}
    events = [
        _delta("text_delta", text="Calling."),
        _delta("input_json_delta", partial_json=json.dumps(wrapped)),
    ]
    message = _message(
        [
            SimpleNamespace(type="text", text="Calling."),
            SimpleNamespace(type="tool_use", id="t1", name="generate_json", input=wrapped),
        ],
        stop_reason="tool_use",
    )
    adapter, fake = _anthropic((events, message))
    deltas, on_stream = _collect()

    result = adapter.generate_structured(
        model="claude-opus-4-1", prompt="1+2?", schema=SCHEMA,
        on_stream=on_stream if stream else None,
    )

    assert fake.calls[0]["tools"][0]["name"] == "generate_json"
    assert result.value == ANSWER
    assert deltas == ([StreamDelta("text", "Calling.")] if stream else [])


def test_a_callback_exception_propagates():
    adapter, _ = _anthropic((THINKING_EVENTS, THOUGHT_MESSAGE))

    def boom(delta):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        adapter.chat(model="claude-opus-4-8", messages=[user("q")], on_stream=boom)


# ── through the Client, and run_tools ─────────────────────────────────────────


def _client_with(adapter) -> Client:
    client = Client(key_provider=lambda spec: "k")
    client._adapters[(adapter.spec.name, "k")] = adapter
    return client


def test_run_tools_streams_every_model_call_before_its_on_response():
    tool_message = _message(
        [
            SimpleNamespace(type="thinking", thinking="Use add.", signature="s"),
            SimpleNamespace(type="tool_use", id="t1", name="add", input={"a": 1}),
        ],
        stop_reason="tool_use",
    )
    adapter, fake = _anthropic(
        ([_delta("thinking_delta", thinking="Use add.")], tool_message),
        (THINKING_EVENTS, THOUGHT_MESSAGE),
    )
    log: list = []

    run = _client_with(adapter).run_tools(
        model="claude-opus-4-8",
        messages=[user("1+2?")],
        tools=[ADD],
        handlers={"add": lambda inp: inp["a"] + 2},
        effort=Effort.MEDIUM,
        on_stream=lambda d: log.append((d.kind, d.text)),
        on_response=lambda r, s: log.append(("response", r.text)),
        on_tool_call=lambda c, r, s: log.append(("tool", c.name)),
        return_run=True,
    )

    assert log == [
        ("thinking", "Use add."),
        ("response", ""),
        ("tool", "add"),
        ("thinking", "Let me "),
        ("thinking", "add."),
        ("text", "It is "),
        ("text", "3."),
        ("response", "It is 3."),
    ]
    assert run.response.text == "It is 3."
    assert all(call["thinking"]["display"] == "summarized" for call in fake.calls)


def test_client_generate_structured_json_passes_the_callback():
    text = json.dumps(ANSWER)
    adapter, _ = _anthropic(
        ([_delta("text_delta", text=text)], _message([SimpleNamespace(type="text", text=text)]))
    )
    deltas, on_stream = _collect()
    value = _client_with(adapter).generate_structured_json(
        model="claude-opus-4-8", prompt="q", schema=SCHEMA, on_stream=on_stream
    )
    assert value == ANSWER
    assert _joined(deltas, "text") == text


class _OldAdapter(AnthropicAdapter):
    """An adapter written before streaming: its methods take no on_stream."""

    def generate_text(self, *, model, prompt, system=None, temperature=0.1, max_tokens=None):
        return "old"


def test_client_does_not_pass_a_missing_callback_to_the_adapter():
    adapter = _OldAdapter(default_registry().get("anthropic"), "", raw_client=_FakeAnthropic())
    assert _client_with(adapter).generate_text(model="claude-opus-4-8", prompt="q") == "old"


# ── Gemini ────────────────────────────────────────────────────────────────────


def _gemini_chunk(*parts, finish=None, usage=None):
    from google.genai import types

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                finish_reason=finish,
            )
        ],
        usage_metadata=usage,
    )


class _FakeGemini:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls: list[dict] = []
        self.models = SimpleNamespace(
            generate_content=self._unexpected, generate_content_stream=self._stream
        )

    def _unexpected(self, **kwargs):
        raise AssertionError("a call with on_stream should stream")

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.chunks)


def _gemini(chunks):
    fake = _FakeGemini(chunks)
    return GeminiAdapter(default_registry().get("gemini"), "", raw_client=fake), fake


def _gemini_thought_stream(*answer_parts):
    from google.genai import types

    return [
        _gemini_chunk(types.Part(text="Adding ", thought=True)),
        _gemini_chunk(types.Part(text="1 and 2.", thought=True)),
        *[_gemini_chunk(p) for p in answer_parts[:-1]],
        _gemini_chunk(
            answer_parts[-1],
            finish="STOP",
            usage=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=5, candidates_token_count=7, thoughts_token_count=4
            ),
        ),
    ]


def test_gemini_chat_streams_thought_summaries_and_merges_the_reply():
    from google.genai import types

    adapter, fake = _gemini(
        _gemini_thought_stream(types.Part(text="It is "), types.Part(text="3."))
    )
    deltas, on_stream = _collect()

    resp = adapter.chat(
        model="gemini-3.8-flash", messages=[user("1+2?")], effort=Effort.LOW, on_stream=on_stream
    )

    assert deltas == [
        StreamDelta("thinking", "Adding "),
        StreamDelta("thinking", "1 and 2."),
        StreamDelta("text", "It is "),
        StreamDelta("text", "3."),
    ]
    # Thought parts are reasoning, not the answer.
    assert resp.text == "It is 3."
    assert (resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.reasoning_tokens) == (5, 7, 4)
    thinking = fake.calls[0]["config"].thinking_config
    assert thinking.include_thoughts is True
    assert thinking.thinking_budget == 1024  # the effort's budget is kept


def test_gemini_streamed_tool_call_decodes():
    from google.genai import types

    call = types.Part(function_call=types.FunctionCall(name="add", args={"a": 1}))
    adapter, _ = _gemini(_gemini_thought_stream(call))
    resp = adapter.chat(
        model="gemini-3.8-flash", messages=[user("1+2?")], tools=[ADD], on_stream=lambda d: None
    )
    assert [(c.name, c.input) for c in resp.tool_calls] == [("add", {"a": 1})]
    assert resp.text == ""


def test_gemini_structured_streams_and_parses_only_the_answer():
    from google.genai import types

    text = json.dumps(ANSWER)
    adapter, _ = _gemini(
        _gemini_thought_stream(types.Part(text=text[:7]), types.Part(text=text[7:]))
    )
    deltas, on_stream = _collect()
    result = adapter.generate_structured(
        model="gemini-3.8-flash", prompt="1+2?", schema=SCHEMA, on_stream=on_stream
    )
    assert result.value == ANSWER
    assert _joined(deltas, "thinking") == "Adding 1 and 2."
    assert _joined(deltas, "text") == text


def test_gemini_generate_text_streams():
    from google.genai import types

    adapter, _ = _gemini(_gemini_thought_stream(types.Part(text="3")))
    deltas, on_stream = _collect()
    assert adapter.generate_text(model="gemini-3.8-flash", prompt="1+2?", on_stream=on_stream) == "3"
    assert _joined(deltas, "thinking") == "Adding 1 and 2."


# ── OpenAI-compatible ─────────────────────────────────────────────────────────


def _chunk(delta: dict | None = None, finish=None, usage=None):
    from openai.types.chat import ChatCompletionChunk

    choices = []
    if delta is not None:
        choices = [{"index": 0, "delta": delta, "finish_reason": finish}]
    data = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": choices}
    if usage is not None:
        data["usage"] = usage
    return ChatCompletionChunk.model_validate(data)


class _FakeOpenAI:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs.get("stream") is True
        return iter(self.chunks)


def _openai(provider, chunks):
    fake = _FakeOpenAI(chunks)
    return OpenAICompatibleAdapter(default_registry().get(provider), "", raw_client=fake), fake


USAGE = {"prompt_tokens": 9, "completion_tokens": 11, "total_tokens": 20}


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_openai_compatible_reasoning_streams_as_thinking(field):
    adapter, fake = _openai(
        "deepseek",
        [
            _chunk({"role": "assistant", field: "Adding "}),
            _chunk({field: "1 and 2."}),
            _chunk({"content": "It is "}),
            _chunk({"content": "3."}, finish="stop"),
            _chunk(usage=USAGE),
        ],
    )
    deltas, on_stream = _collect()

    resp = adapter.chat(model="deepseek-reasoner", messages=[user("1+2?")], on_stream=on_stream)

    assert deltas == [
        StreamDelta("thinking", "Adding "),
        StreamDelta("thinking", "1 and 2."),
        StreamDelta("text", "It is "),
        StreamDelta("text", "3."),
    ]
    assert resp.text == "It is 3."
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (9, 11)
    assert fake.calls[0]["stream_options"] == {"include_usage": True}


def test_openai_streamed_tool_calls_are_reassembled():
    adapter, _ = _openai(
        "openai",
        [
            _chunk({"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "add", "arguments": '{"a"'}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}),
            _chunk({}, finish="tool_calls"),
            _chunk(usage=USAGE),
        ],
    )
    deltas, on_stream = _collect()
    resp = adapter.chat(model="gpt-5.5", messages=[user("1+2?")], tools=[ADD], on_stream=on_stream)
    assert [(c.id, c.name, c.input) for c in resp.tool_calls] == [("call_1", "add", {"a": 1})]
    assert deltas == []  # OpenAI streams no reasoning, and there was no text


def test_openai_compatible_structured_streams():
    text = json.dumps(ANSWER)
    adapter, _ = _openai(
        "deepseek",
        [_chunk({"role": "assistant", "content": text[:5]}), _chunk({"content": text[5:]}, finish="stop")],
    )
    deltas, on_stream = _collect()
    result = adapter.generate_structured(
        model="deepseek-chat", prompt="1+2?", schema=SCHEMA, on_stream=on_stream
    )
    assert result.value == ANSWER
    assert _joined(deltas, "text") == text


def test_streamed_thinking_capability():
    registry = default_registry()
    for name in ("anthropic", "gemini", "deepseek"):
        assert registry.get(name).capabilities.has(Capability.STREAMED_THINKING)
    for name in ("openai", "mistral"):
        assert not registry.get(name).capabilities.has(Capability.STREAMED_THINKING)


class _ClosableStream:
    """A provider's open response: it stays open (and the model keeps generating) until closed."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self):
        self.closed = True


def test_an_on_stream_that_raises_closes_the_openai_compatible_stream():
    """A host's Stop raises from ``on_stream``: the request must end there, not run on (and
    bill) until the response is collected."""
    stream = _ClosableStream([_chunk({"reasoning": "Adding "}), _chunk({"content": "3."})])
    fake = _FakeOpenAI([])
    fake._create = lambda **kwargs: stream  # type: ignore[method-assign]
    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=fake._create))
    adapter = OpenAICompatibleAdapter(default_registry().get("deepseek"), "", raw_client=fake)

    def stop(delta):
        raise RuntimeError("stopped")

    with pytest.raises(RuntimeError, match="stopped"):
        adapter.chat(model="deepseek-reasoner", messages=[user("q")], on_stream=stop)
    assert stream.closed


def test_a_finished_openai_compatible_stream_is_closed_too():
    stream = _ClosableStream([_chunk({"content": "3."}, finish="stop"), _chunk(usage=USAGE)])
    fake = _FakeOpenAI([])
    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: stream))
    adapter = OpenAICompatibleAdapter(default_registry().get("deepseek"), "", raw_client=fake)

    resp = adapter.chat(model="deepseek-reasoner", messages=[user("q")], on_stream=lambda d: None)

    assert resp.text == "3." and stream.closed


# ── what a stopped call was billed: partial_usage ─────────────────────────────

from inferenceswitch import Usage, partial_usage  # noqa: E402
from inferenceswitch.cards import Cost  # noqa: E402


class _Stop(Exception):
    """A host's Stop, raised from ``on_stream``."""


def _stop_after(n: int):
    seen: list[StreamDelta] = []

    def on_stream(delta):
        seen.append(delta)
        if len(seen) >= n:
            raise _Stop()

    return on_stream


def _usage_events(output_so_far: int):
    start = SimpleNamespace(
        input_tokens=12, cache_read_input_tokens=100, cache_creation_input_tokens=50, output_tokens=1
    )
    return [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=start)),
        _delta("thinking_delta", thinking="Let me "),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=output_so_far)),
        _delta("thinking_delta", thinking="add."),
        _delta("text_delta", text="3."),
    ]


def test_a_stopped_anthropic_call_hands_back_the_counts_message_start_gave():
    adapter, _ = _anthropic((_usage_events(40), THOUGHT_MESSAGE))
    with pytest.raises(_Stop) as stopped:
        adapter.chat(model="claude-opus-4-8", messages=[user("q")], on_stream=_stop_after(1))
    usage = partial_usage(stopped.value)
    # the prompt's three counts are the real ones; the output is what message_start said (1)
    assert usage == Usage(
        input_tokens=12, output_tokens=1, cache_read_tokens=100, cache_write_tokens=50, partial=True
    )


def test_a_stopped_anthropic_call_has_the_output_message_delta_reported_so_far():
    adapter, _ = _anthropic((_usage_events(40), THOUGHT_MESSAGE))
    with pytest.raises(_Stop) as stopped:
        adapter.chat(model="claude-opus-4-8", messages=[user("q")], on_stream=_stop_after(2))
    usage = partial_usage(stopped.value)
    assert (usage.input_tokens, usage.output_tokens, usage.partial) == (12, 40, True)
    # priced as any usage is, from the card's prices
    usd, unpriced = Cost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75).usd(usage)
    assert usd == pytest.approx((12 * 3 + 40 * 15 + 100 * 0.3 + 50 * 3.75) / 1e6)
    assert unpriced == ()


def test_a_stopped_gemini_call_hands_back_the_last_usage_metadata_seen():
    from google.genai import types

    def meta(prompt, out):
        return types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, candidates_token_count=out, cached_content_token_count=2
        )

    adapter, _ = _gemini(
        [
            _gemini_chunk(types.Part(text="Adding ", thought=True), usage=meta(30, 3)),
            _gemini_chunk(types.Part(text="1 and 2.", thought=True), usage=meta(30, 9)),
            _gemini_chunk(types.Part(text="3."), finish="STOP", usage=meta(30, 12)),
        ]
    )
    with pytest.raises(_Stop) as stopped:
        adapter.chat(model="gemini-3.8-flash", messages=[user("q")], on_stream=_stop_after(2))
    usage = partial_usage(stopped.value)
    # prompt_token_count counts the cached 2 too; input_tokens is the uncached rest
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (28, 9, 2)
    assert usage.partial


def test_a_stopped_openai_compatible_call_says_it_has_no_counts():
    adapter, _ = _openai(
        "deepseek",
        [_chunk({"reasoning": "Adding "}), _chunk({"content": "3."}), _chunk(usage=USAGE)],
    )
    with pytest.raises(_Stop) as stopped:
        adapter.chat(model="deepseek-reasoner", messages=[user("q")], on_stream=_stop_after(1))
    usage = partial_usage(stopped.value)
    assert usage == Usage(partial=True)  # billed, and not known by how much: nothing invented


def test_an_openai_compatible_call_stopped_after_its_usage_chunk_has_the_real_counts():
    adapter, _ = _openai(
        "deepseek",
        [_chunk(usage=USAGE), _chunk({"content": "3."})],
    )
    with pytest.raises(_Stop) as stopped:
        adapter.chat(model="deepseek-reasoner", messages=[user("q")], on_stream=_stop_after(1))
    usage = partial_usage(stopped.value)
    assert (usage.input_tokens, usage.output_tokens, usage.partial) == (9, 11, True)


def test_an_error_that_did_not_come_out_of_a_stream_has_no_partial_usage():
    assert partial_usage(RuntimeError("no")) is None


def test_partial_usage_survives_adding_calls_and_marks_the_sum():
    total = Usage(input_tokens=5, output_tokens=2) + Usage(input_tokens=3, partial=True)
    assert total.partial and total.input_tokens == 8
