"""Image input: the ``Image`` part, what each adapter sends for it, and the
VISION gate.

Offline — adapters are driven with fake SDK clients that record the request,
so every assertion is about what inferenceswitch *sends*.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace as NS

import pytest

from inferenceswitch import (
    Client,
    Image,
    InferenceSwitchError,
    Message,
    Text,
    Tool,
    ToolResult,
    UnsupportedCapabilityError,
    user,
)
from inferenceswitch.adapters.anthropic import AnthropicAdapter, encode_anthropic_messages
from inferenceswitch.adapters.gemini import GeminiAdapter, encode_gemini_contents
from inferenceswitch.adapters.openai_compat import (
    OpenAICompatibleAdapter,
    encode_openai_messages,
)
from inferenceswitch.registry import default_registry

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00fake image bytes"
PNG_B64 = base64.b64encode(PNG).decode("ascii")
IMAGE = Image(PNG, "image/png")
SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}
VALUE = {"text": "Solve x^2 = 4."}

ANTHROPIC_IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
}
OPENAI_IMAGE = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}
GEMINI_IMAGE = {"inline_data": {"mime_type": "image/png", "data": PNG}}


# ── fakes ─────────────────────────────────────────────────────────────────────


class _FakeAnthropic:
    def __init__(self, text=json.dumps(VALUE)):
        self.calls: list[dict] = []
        self.text = text
        self.messages = NS(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        message = NS(
            content=[NS(type="text", text=self.text)],
            stop_reason="end_turn",
            stop_details=None,
            usage=None,
        )

        class _Stream:
            def __enter__(inner):
                return inner

            def __exit__(inner, *exc):
                return False

            def get_final_message(inner):
                return message

        return _Stream()


class _FakeOpenAI:
    def __init__(self, text=json.dumps(VALUE)):
        self.calls: list[dict] = []
        self.text = text
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return NS(
            choices=[NS(message=NS(content=self.text, tool_calls=None), finish_reason="stop")],
            usage=None,
        )


class _FakeGenai:
    def __init__(self, text=json.dumps(VALUE)):
        self.calls: list[dict] = []
        self.text = text
        self.models = NS(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return NS(
            candidates=[NS(content=NS(parts=[]), finish_reason=NS(name="STOP"))],
            text=self.text,
            usage_metadata=None,
        )


def _client(provider: str, fake) -> Client:
    """A client whose ``provider`` adapter drives ``fake``."""
    spec = default_registry().get(provider)
    adapter_cls = {
        "anthropic": AnthropicAdapter,
        "gemini": GeminiAdapter,
    }.get(provider, OpenAICompatibleAdapter)
    client = Client(key_provider=lambda spec: "k")
    client._adapters[(provider, "k")] = adapter_cls(spec, "k", raw_client=fake)
    return client


# ── the Image part ────────────────────────────────────────────────────────────


def test_bytes_and_base64_are_the_same_image():
    from_b64 = Image(PNG_B64, "image/png")
    assert from_b64.to_bytes() == IMAGE.to_bytes() == PNG
    assert from_b64.to_base64() == IMAGE.to_base64() == PNG_B64
    assert IMAGE.data_url() == f"data:image/png;base64,{PNG_B64}"


def test_unsupported_media_type_is_refused():
    with pytest.raises(ValueError, match="image/bmp"):
        Image(PNG, "image/bmp")


def test_a_string_that_is_not_base64_is_refused():
    with pytest.raises(ValueError, match="base64"):
        Image("not base64!", "image/png")


def test_from_file_guesses_the_type(tmp_path):
    path = tmp_path / "exercise.jpg"
    path.write_bytes(b"\xff\xd8\xff jpeg")
    image = Image.from_file(path)
    assert image.media_type == "image/jpeg"
    assert image.to_bytes() == b"\xff\xd8\xff jpeg"
    with pytest.raises(ValueError, match="media_type"):
        Image.from_file(tmp_path / "exercise.unknownext")


def test_user_puts_images_before_text():
    assert user("read it", images=[IMAGE]) == Message("user", [IMAGE, Text("read it")])
    assert user("plain") == Message("user", "plain")


# ── encode: one image, three native shapes ────────────────────────────────────


def test_encode_anthropic_image_block():
    out = encode_anthropic_messages([user("read it", images=[IMAGE])])
    assert out == [
        {"role": "user", "content": [ANTHROPIC_IMAGE, {"type": "text", "text": "read it"}]}
    ]


def test_encode_openai_image_url_part():
    out = encode_openai_messages([user("read it", images=[IMAGE])])
    assert out == [
        {"role": "user", "content": [OPENAI_IMAGE, {"type": "text", "text": "read it"}]}
    ]


def test_encode_openai_keeps_text_only_turns_as_strings():
    assert encode_openai_messages([user("hi")]) == [{"role": "user", "content": "hi"}]


def test_encode_openai_image_beside_a_tool_result():
    # A tool result is its own "tool" message; the image stays in the user turn.
    message = Message("user", [ToolResult("call_1", "done", name="t"), IMAGE])
    assert encode_openai_messages([message]) == [
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        {"role": "user", "content": [OPENAI_IMAGE]},
    ]


def test_encode_gemini_inline_data():
    out = encode_gemini_contents([user("read it", images=[IMAGE])])
    assert out == [{"role": "user", "parts": [GEMINI_IMAGE, {"text": "read it"}]}]


# ── chat and run_tools, through the client ────────────────────────────────────


def test_chat_sends_an_image_to_claude():
    fake = _FakeAnthropic(text="Solve x^2 = 4.")
    client = _client("anthropic", fake)
    response = client.chat(model="claude-sonnet-5", messages=[user("read it", images=[IMAGE])])
    assert response.text == "Solve x^2 = 4."
    assert fake.calls[0]["messages"][0]["content"][0] == ANTHROPIC_IMAGE


def test_chat_sends_an_image_to_grok():
    fake = _FakeOpenAI(text="Solve x^2 = 4.")
    client = _client("xai", fake)
    response = client.chat(model="grok-4.6", messages=[user("read it", images=[IMAGE])])
    assert response.text == "Solve x^2 = 4."
    assert fake.calls[0]["model"] == "grok-4.6"
    assert fake.calls[0]["messages"] == [
        {"role": "user", "content": [OPENAI_IMAGE, {"type": "text", "text": "read it"}]}
    ]


def test_run_tools_carries_the_image_on_every_turn():
    fake = _FakeAnthropic(text="done")
    client = _client("anthropic", fake)
    tool = Tool("noop", "does nothing", {"type": "object", "properties": {}})
    run = client.run_tools(
        model="claude-sonnet-5",
        messages=[user("read it", images=[IMAGE])],
        tools=[tool],
        handlers={"noop": lambda _: "ok"},
        return_run=True,
    )
    assert run.response.text == "done"
    assert fake.calls[0]["messages"][0]["content"][0] == ANTHROPIC_IMAGE
    assert run.messages[0].content[0] is IMAGE


# ── structured calls and generate_text ────────────────────────────────────────


def test_structured_anthropic_image_after_the_cache_prefix():
    fake = _FakeAnthropic()
    client = _client("anthropic", fake)
    result = client.generate_structured(
        model="claude-sonnet-5",
        schema=SCHEMA,
        prompt="Transcribe it.",
        cache_prefix="Instructions. " * 20,
        images=[IMAGE],
    )
    assert result.value == VALUE
    content = fake.calls[0]["messages"][0]["content"]
    # Prefix (cached), then the image, then the prompt: the breakpoint stays
    # above the image, so the prefix is read back for the next image.
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[1] == ANTHROPIC_IMAGE
    assert content[2] == {"type": "text", "text": "Transcribe it."}


def test_structured_xai_image():
    fake = _FakeOpenAI()
    client = _client("xai", fake)
    value = client.generate_structured_json(
        model="grok-4.6", schema=SCHEMA, prompt="Transcribe it.", system="sys", images=[IMAGE]
    )
    assert value == VALUE
    sent = fake.calls[0]
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [OPENAI_IMAGE, {"type": "text", "text": "Transcribe it."}]},
    ]
    assert sent["response_format"]["type"] == "json_schema"
    assert "temperature" not in sent  # xAI: no sampling params sent


def test_structured_gemini_image():
    fake = _FakeGenai()
    client = _client("gemini", fake)
    value = client.generate_structured_json(
        model="gemini-3.8-flash", schema=SCHEMA, prompt="Transcribe it.", images=[IMAGE]
    )
    assert value == VALUE
    assert fake.calls[0]["contents"] == [
        {"role": "user", "parts": [GEMINI_IMAGE, {"text": "Transcribe it."}]}
    ]


def test_generate_text_with_an_image():
    claude = _FakeAnthropic(text="Solve x^2 = 4.")
    assert _client("anthropic", claude).generate_text(
        model="claude-sonnet-5", prompt="Transcribe it.", images=[IMAGE]
    ) == "Solve x^2 = 4."
    assert claude.calls[0]["messages"][0]["content"][0] == ANTHROPIC_IMAGE

    grok = _FakeOpenAI(text="Solve x^2 = 4.")
    _client("xai", grok).generate_text(model="grok-4.6", prompt="Transcribe it.", images=[IMAGE])
    assert grok.calls[0]["messages"][0]["content"][0] == OPENAI_IMAGE

    gemini = _FakeGenai(text="Solve x^2 = 4.")
    _client("gemini", gemini).generate_text(
        model="gemini-3.8-flash", prompt="Transcribe it.", images=[IMAGE]
    )
    assert gemini.calls[0]["contents"][0]["parts"][0] == GEMINI_IMAGE


def test_without_images_the_request_is_unchanged():
    fake = _FakeOpenAI()
    _client("xai", fake).generate_structured_json(model="grok-4.6", schema=SCHEMA, prompt="p")
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "p"}]


def test_no_images_kwarg_reaches_an_adapter_when_there_are_none():
    # An adapter written before images (a caller's own) still works.
    client = Client(key_provider=lambda spec: "k")

    class _Old:
        def generate_text(self, *, model, prompt, system, temperature, max_tokens):
            return "ok"

    client._adapters[("xai", "k")] = _Old()
    assert client.generate_text(model="grok-4.6", prompt="p") == "ok"


# ── the VISION gate ───────────────────────────────────────────────────────────


class _Unreachable:
    """An adapter that fails the test if a call ever reaches it."""

    def __getattr__(self, name):
        raise AssertionError(f"{name} reached the adapter; the VISION gate should have refused")


def test_image_to_a_model_without_vision_is_refused_before_any_request():
    client = Client(key_provider=lambda spec: "k")
    client._adapters[("deepseek", "k")] = _Unreachable()
    # DeepSeek has no VISION (only the deepseek-flash card adds it).
    with pytest.raises(UnsupportedCapabilityError, match="vision"):
        client.chat(model="deepseek-v4-pro", messages=[user("read it", images=[IMAGE])])
    with pytest.raises(UnsupportedCapabilityError, match="vision"):
        client.generate_structured(
            model="deepseek-v4-pro", schema=SCHEMA, prompt="p", images=[IMAGE]
        )
    with pytest.raises(UnsupportedCapabilityError, match="vision"):
        client.generate_text(model="deepseek-v4-pro", prompt="p", images=[IMAGE])
    with pytest.raises(UnsupportedCapabilityError, match="vision"):
        client.run_tools(
            model="deepseek-v4-pro",
            messages=[user("read it", images=[IMAGE])],
            tools=[Tool("noop", "", {"type": "object", "properties": {}})],
            handlers={"noop": lambda _: "ok"},
        )


def test_a_model_card_can_take_vision_away():
    # xAI reads images, but the non-reasoning Grok's card turns VISION off.
    client = Client(key_provider=lambda spec: "k")
    client._adapters[("xai", "k")] = _Unreachable()
    with pytest.raises(UnsupportedCapabilityError, match="grok-4.20"):
        client.chat(
            model="grok-4.20-0309-non-reasoning", messages=[user("x", images=[IMAGE])]
        )


def test_an_image_outside_a_user_message_is_refused():
    client = Client(key_provider=lambda spec: "k")
    client._adapters[("anthropic", "k")] = _Unreachable()
    with pytest.raises(InferenceSwitchError, match="user message"):
        client.chat(model="claude-sonnet-5", messages=[Message("assistant", [IMAGE])])
