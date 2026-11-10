import pytest

from api.models.anthropic import Message, MessagesRequest
from config.settings import Settings
from core.anthropic.sse import format_sse_event
from api.provider_fallback import stream_with_provider_fallback


class FakeProvider:
    def __init__(self, events):
        self.events = events

    def preflight_stream(self, request, *, thinking_enabled=None):
        return None

    async def stream_response(
        self, request, input_tokens=0, *, request_id=None, thinking_enabled=None
    ):
        for event in self.events:
            yield event


def _event(name, payload):
    return format_sse_event(name, payload)


@pytest.mark.asyncio
async def test_fallback_discards_failed_provider_events_before_first_content():
    failed = FakeProvider(
        [
            _event("message_start", {"type": "message_start"}),
            _event(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": "timeout"},
                },
            ),
        ]
    )
    backup = FakeProvider(
        [
            _event("message_start", {"type": "message_start"}),
            _event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            ),
            _event("message_stop", {"type": "message_stop"}),
        ]
    )
    settings = Settings(
        model="nvidia_nim/primary",
        model_opus="nvidia_nim/primary",
        model_sonnet="open_router/backup",
        model_haiku="deepseek/third",
    )
    request = MessagesRequest(
        model="primary",
        messages=[Message(role="user", content="hello")],
        max_tokens=20,
    )
    providers = {"nvidia_nim": failed, "open_router": backup}
    seen = []

    async for event in stream_with_provider_fallback(
        request,
        primary_provider_id="nvidia_nim",
        primary_model="primary",
        settings=settings,
        provider_getter=lambda provider_id: providers[provider_id],
        input_tokens=1,
        request_id="test",
        thinking_enabled=False,
    ):
        seen.append(event)

    joined = "".join(seen)
    assert "Hello" in joined
    assert "timeout" not in joined
    assert joined.count("event: message_start") == 1


@pytest.mark.asyncio
async def test_stream_does_not_switch_provider_after_content_starts():
    first = FakeProvider(
        [
            _event("message_start", {"type": "message_start"}),
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Partial"},
                },
            ),
            _event(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": "later failure"},
                },
            ),
        ]
    )
    backup = FakeProvider(
        [
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Wrong provider"},
                },
            )
        ]
    )
    settings = Settings(
        model="nvidia_nim/primary",
        model_opus="nvidia_nim/primary",
        model_sonnet="open_router/backup",
    )
    request = MessagesRequest(
        model="primary",
        messages=[Message(role="user", content="hello")],
        max_tokens=20,
    )
    providers = {"nvidia_nim": first, "open_router": backup}
    seen = []

    async for event in stream_with_provider_fallback(
        request,
        primary_provider_id="nvidia_nim",
        primary_model="primary",
        settings=settings,
        provider_getter=lambda provider_id: providers[provider_id],
        input_tokens=1,
        request_id="test",
        thinking_enabled=False,
    ):
        seen.append(event)

    joined = "".join(seen)
    assert "Partial" in joined
    assert "later failure" in joined
    assert "Wrong provider" not in joined
