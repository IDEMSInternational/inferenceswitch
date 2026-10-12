"""run_tools: stop conditions, hooks, history and usage out, cache pass-through.

Scripted, no SDK: ``chat`` (or the adapter behind it) is replaced by a fake that
plays back a fixed list of responses and records what it was sent.
"""
from __future__ import annotations

import pytest

from inferenceswitch import (
    CacheHandle,
    Client,
    InferenceSwitchError,
    Message,
    Response,
    StopReason,
    Text,
    Tool,
    ToolCall,
    ToolResult,
    ToolRun,
    ToolUse,
    UnsupportedCapabilityError,
    Usage,
    user,
)

TOOLS = [Tool("add", "adds", {"type": "object"}), Tool("decide", "final answer", {"type": "object"})]


def _calls(*calls: ToolCall, text: str = "", usage: Usage | None = None) -> Response:
    return Response(text=text, tool_calls=list(calls), stop_reason=StopReason.TOOL_USE,
                    usage=usage or Usage())


def _done(text: str = "done", usage: Usage | None = None) -> Response:
    return Response(text=text, stop_reason=StopReason.END_TURN, usage=usage or Usage())


def _script(monkeypatch, client: Client, *responses: Response) -> list[dict]:
    """Make ``client.chat`` return ``responses`` in order; return the kwargs of
    each call, with ``messages`` snapshotted (the loop extends the list)."""
    sent: list[dict] = []
    queue = list(responses)

    def fake_chat(**kw) -> Response:
        sent.append({**kw, "messages": list(kw["messages"])})
        return queue.pop(0)

    monkeypatch.setattr(client, "chat", fake_chat)
    return sent


def _handlers(log: list | None = None) -> dict:
    def add(inp):
        if log is not None:
            log.append(("add", inp))
        return inp["a"] + inp["b"]

    def decide(inp):
        if log is not None:
            log.append(("decide", inp))
        if inp.get("bad"):
            raise ValueError("invalid decision")
        return {"accepted": inp["choice"]}

    return {"add": add, "decide": decide}


# ── existing callers ──────────────────────────────────────────────────────────


def test_without_new_arguments_behaviour_is_unchanged(monkeypatch):
    client = Client()
    sent = _script(
        monkeypatch, client,
        _calls(ToolCall("1", "add", {"a": 1, "b": 2}), ToolCall("2", "decide", {"choice": "x"})),
        _done("all done"),
    )
    final = client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                             handlers=_handlers(), system="sys")

    # The plain final Response, as before; a terminal-looking tool doesn't stop it.
    assert type(final) is Response and final.text == "all done"
    assert len(sent) == 2
    # Each chat call gets the same arguments as before; cache=None is chat's default.
    assert sent[0] == {
        "model": "m", "messages": [user("go")], "tools": TOOLS, "system": "sys",
        "effort": None, "cache": None, "provider": None, "api_key": None,
        "temperature": 0.1, "max_tokens": None,
    }
    assert sent[1]["messages"][1:] == [
        Message("assistant", [ToolUse("1", "add", {"a": 1, "b": 2}),
                              ToolUse("2", "decide", {"choice": "x"})]),
        Message("user", [ToolResult("1", "3", name="add"),
                         ToolResult("2", "{'accepted': 'x'}", name="decide")]),
    ]


def test_without_new_arguments_max_turns_still_raises(monkeypatch):
    client = Client()
    _script(monkeypatch, client, *[_calls(ToolCall(str(i), "add", {"a": 1, "b": 1}))
                                   for i in range(3)])
    with pytest.raises(InferenceSwitchError, match="max_turns=3"):
        client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                         handlers=_handlers(), max_turns=3)


def test_caller_messages_are_not_mutated(monkeypatch):
    client = Client()
    _script(monkeypatch, client, _calls(ToolCall("1", "add", {"a": 1, "b": 1})), _done())
    messages = [user("go")]
    client.run_tools(model="m", messages=messages, tools=TOOLS, handlers=_handlers())
    assert messages == [user("go")]


def test_tools_or_cache_is_required():
    with pytest.raises(ValueError):
        Client().run_tools(model="m", messages=[user("go")], handlers={})


# ── stop conditions ───────────────────────────────────────────────────────────


def test_terminal_tool_stops_after_the_round(monkeypatch):
    client = Client()
    log: list = []
    stopping = _calls(ToolCall("1", "decide", {"choice": "x"}),
                      ToolCall("2", "add", {"a": 1, "b": 1}))
    sent = _script(monkeypatch, client,
                   _calls(ToolCall("0", "add", {"a": 2, "b": 2})), stopping, _done())
    final = client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                             handlers=_handlers(log), terminal_tools={"decide"})

    assert final is stopping              # the response that made the stopping call
    assert len(sent) == 2                 # no model call after the terminal tool
    # The rest of the round still ran, so every call has a result.
    assert log == [("add", {"a": 2, "b": 2}), ("decide", {"choice": "x"}),
                   ("add", {"a": 1, "b": 1})]


def test_a_failing_terminal_call_does_not_stop(monkeypatch):
    client = Client()
    sent = _script(monkeypatch, client,
                   _calls(ToolCall("1", "decide", {"bad": True})),
                   _calls(ToolCall("2", "decide", {"choice": "y"})),
                   _done())
    run = client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                           handlers=_handlers(), terminal_tools=["decide"], return_run=True)
    assert len(sent) == 2                 # the model got the error and retried
    assert run.stopped_by == ToolCall("2", "decide", {"choice": "y"})
    assert run.messages[2] == Message("user", [ToolResult(
        "1", "Tool error: invalid decision", is_error=True, name="decide")])


def test_an_unmapped_terminal_tool_does_not_stop(monkeypatch):
    client = Client()
    sent = _script(monkeypatch, client, _calls(ToolCall("1", "decide", {})), _done())
    client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                     handlers={}, terminal_tools={"decide"})
    assert len(sent) == 2


def test_stop_when_sees_the_raw_output(monkeypatch):
    client = Client()
    seen: list = []

    def stop_when(call, output):
        seen.append((call.name, output))
        return call.name == "add" and output > 10

    sent = _script(monkeypatch, client,
                   _calls(ToolCall("1", "add", {"a": 1, "b": 2})),
                   _calls(ToolCall("2", "add", {"a": 9, "b": 9}),
                          ToolCall("3", "decide", {"choice": "z"})),
                   _done())
    run = client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                           handlers=_handlers(), stop_when=stop_when, return_run=True)
    assert len(sent) == 2
    assert run.stopped_by == ToolCall("2", "add", {"a": 9, "b": 9})
    # Evaluated after every successful handler, with the unstringified value.
    assert seen == [("add", 3), ("add", 18), ("decide", {"accepted": "z"})]


def test_stop_on_the_last_allowed_turn_does_not_raise(monkeypatch):
    client = Client()
    _script(monkeypatch, client, _calls(ToolCall("1", "decide", {"choice": "x"})))
    final = client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                             handlers=_handlers(), terminal_tools={"decide"}, max_turns=1)
    assert final.tool_calls[0].name == "decide"


# ── hooks ─────────────────────────────────────────────────────────────────────


def test_hooks_see_every_response_and_call_in_order(monkeypatch):
    client = Client()
    first = _calls(ToolCall("1", "add", {"a": 1, "b": 2}), ToolCall("2", "nope", {}),
                   ToolCall("3", "decide", {"bad": True}))
    last = _done()
    _script(monkeypatch, client, first, last)
    events: list = []
    client.run_tools(
        model="m", messages=[user("go")], tools=TOOLS, handlers=_handlers(),
        on_response=lambda r, s: events.append(("response", r, s)),
        on_tool_call=lambda c, r, s: events.append(("call", c, r, s)),
    )

    assert [e[0] for e in events] == ["response", "call", "call", "call", "response"]
    assert events[0][1] is first and events[4][1] is last
    assert all(isinstance(e[-1], float) and e[-1] >= 0 for e in events)
    _, call, result, _ = events[1]
    assert call.id == "1" and result == ToolResult("1", "3", name="add")
    _, call, result, seconds = events[2]
    assert result.is_error and "No handler" in result.content and seconds == 0.0
    _, call, result, _ = events[3]
    assert result == ToolResult("3", "Tool error: invalid decision", is_error=True, name="decide")


def test_hook_exceptions_propagate(monkeypatch):
    client = Client()
    _script(monkeypatch, client, _calls(ToolCall("1", "add", {"a": 1, "b": 2})), _done())

    def boom(*_):
        raise RuntimeError("trace store down")

    with pytest.raises(RuntimeError, match="trace store down"):
        client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                         handlers=_handlers(), on_tool_call=boom)


# ── history and usage out ─────────────────────────────────────────────────────


def test_return_run_gives_history_usage_and_turns(monkeypatch):
    client = Client()
    _script(
        monkeypatch, client,
        _calls(ToolCall("1", "add", {"a": 1, "b": 2}), text="Let me add.",
               usage=Usage(input_tokens=10, output_tokens=5, cache_read_tokens=8)),
        _done("It's 3.", usage=Usage(input_tokens=20, output_tokens=7)),
    )
    run = client.run_tools(model="m", messages=[user("1+2?")], tools=TOOLS,
                           handlers=_handlers(), return_run=True)

    assert isinstance(run, ToolRun)
    assert run.response.text == "It's 3."
    assert run.turns == 2 and run.stopped_by is None
    assert run.usage == Usage(input_tokens=30, output_tokens=12, cache_read_tokens=8)
    assert run.messages == [
        user("1+2?"),
        Message("assistant", [Text("Let me add."), ToolUse("1", "add", {"a": 1, "b": 2})]),
        Message("user", [ToolResult("1", "3", name="add")]),
        Message("assistant", "It's 3."),
    ]


def test_history_continues_a_conversation(monkeypatch):
    client = Client()
    sent = _script(monkeypatch, client, _done("first"), _done("second"))
    run = client.run_tools(model="m", messages=[user("a")], tools=TOOLS,
                           handlers={}, return_run=True)
    client.run_tools(model="m", messages=[*run.messages, user("b")], tools=TOOLS, handlers={})
    assert sent[1]["messages"] == [user("a"), Message("assistant", "first"), user("b")]


def test_return_run_when_stopped_ends_on_the_tool_results(monkeypatch):
    client = Client()
    _script(monkeypatch, client, _calls(ToolCall("1", "decide", {"choice": "x"}),
                                        usage=Usage(input_tokens=4)))
    run = client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                           handlers=_handlers(), terminal_tools={"decide"}, return_run=True)
    assert run.turns == 1 and run.usage == Usage(input_tokens=4)
    assert run.messages[-1] == Message(
        "user", [ToolResult("1", "{'accepted': 'x'}", name="decide")])


def test_empty_final_text_adds_no_assistant_turn(monkeypatch):
    client = Client()
    _script(monkeypatch, client, _done(""))
    run = client.run_tools(model="m", messages=[user("go")], tools=TOOLS,
                           handlers={}, return_run=True)
    assert run.messages == [user("go")]


def test_usage_sum_keeps_unreported_fields_none():
    total = Usage(input_tokens=1) + Usage(input_tokens=2, reasoning_tokens=3) + Usage()
    assert total == Usage(input_tokens=3, reasoning_tokens=3)


# ── cache pass-through ────────────────────────────────────────────────────────


class _FakeAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.caches: list = []

    def chat(self, *, cache=None, **kw):
        self.caches.append(cache)
        return self.responses.pop(0)


def test_cache_is_forwarded_on_every_round(monkeypatch):
    client = Client()
    adapter = _FakeAdapter([_calls(ToolCall("1", "add", {"a": 1, "b": 1})), _done()])
    monkeypatch.setattr(client, "_adapter", lambda spec, api_key: adapter)
    handle = CacheHandle(provider="anthropic", model="claude-opus-4-8",
                         system="SYS", tools=tuple(TOOLS))

    client.run_tools(model="claude-opus-4-8", messages=[user("go")],
                     handlers=_handlers(), cache=handle)
    assert adapter.caches == [handle, handle]


def test_cache_follows_chats_rules(monkeypatch):
    client = Client()
    monkeypatch.setattr(client, "_adapter", lambda spec, api_key: _FakeAdapter([_done()]))
    handle = CacheHandle(provider="anthropic", model="claude-opus-4-8")

    # The prefix owns tools and system, as with chat(cache=).
    with pytest.raises(InferenceSwitchError, match="cached prefix"):
        client.run_tools(model="claude-opus-4-8", messages=[user("go")], tools=TOOLS,
                         handlers={}, cache=handle)
    # Gated by capability: OpenAI has no reusable cache handle.
    with pytest.raises(UnsupportedCapabilityError):
        client.run_tools(model="openai/gpt-4o", messages=[user("go")], handlers={},
                         cache=CacheHandle(provider="openai", model="gpt-4o"))
