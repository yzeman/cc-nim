"""Retry a failed provider before any assistant content reaches the client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from loguru import logger

from config.settings import Settings
from core.anthropic.sse import format_sse_event
from providers.base import BaseProvider
from .models.anthropic import MessagesRequest


CLOUD_PROVIDER_IDS = frozenset({"nvidia_nim", "open_router", "deepseek"})
FIRST_CONTENT_TIMEOUT_SECONDS = 20
NVIDIA_FIRST_CONTENT_TIMEOUT_SECONDS = 60
ProviderGetter = Callable[[str], BaseProvider]


def _provider_candidates(
    settings: Settings, primary_provider_id: str, primary_model: str
) -> list[tuple[str, str]]:
    """Keep the selected provider first, then use other configured cloud providers."""
    candidates = [(primary_provider_id, primary_model)]
    configured = (
        settings.model_opus,
        settings.model_sonnet,
        settings.model_haiku,
        settings.model,
    )
    seen = {primary_provider_id}
    for value in configured:
        if not value or "/" not in value:
            continue
        provider_id, model_name = value.split("/", 1)
        if (
            provider_id in CLOUD_PROVIDER_IDS
            and provider_id not in seen
            and model_name
        ):
            candidates.append((provider_id, model_name))
            seen.add(provider_id)
    return candidates


def _event_details(raw_event: str) -> tuple[str | None, dict[str, Any] | None]:
    event_name: str | None = None
    data_lines: list[str] = []
    for line in raw_event.splitlines():
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            data_lines.append(line.partition(":")[2].strip())
    if not data_lines:
        return event_name, None
    try:
        payload = json.loads("\\n".join(data_lines))
    except (TypeError, ValueError):
        return event_name, None
    return event_name, payload if isinstance(payload, dict) else None


def _is_error_event(event_name: str | None, payload: dict[str, Any] | None) -> bool:
    return event_name == "error" or bool(payload and payload.get("type") == "error")


def _is_first_content_event(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    delta = payload.get("delta")
    if isinstance(delta, dict) and delta.get("type") in {
        "text_delta",
        "thinking_delta",
        "input_json_delta",
    }:
        return True
    block = payload.get("content_block")
    return isinstance(block, dict) and block.get("type") == "tool_use"


async def stream_with_provider_fallback(
    request: MessagesRequest,
    *,
    primary_provider_id: str,
    primary_model: str,
    settings: Settings,
    provider_getter: ProviderGetter,
    input_tokens: int,
    request_id: str,
    thinking_enabled: bool,
) -> AsyncIterator[str]:
    """Stream from the first provider that starts producing content.

    Events are buffered until the first content delta. A provider that errors or
    produces no content within the first-token deadline is skipped. Once content
    starts, the stream is committed to that provider to avoid mixing answers.
    """
    candidates = _provider_candidates(settings, primary_provider_id, primary_model)
    last_uncommitted_events: list[str] = []

    for index, (provider_id, model_name) in enumerate(candidates):
        buffered: list[str] = []
        committed = False
        stream = None
        try:
            provider = provider_getter(provider_id)
            provider_request = request.model_copy(
                update={"model": model_name}, deep=True
            )
            provider.preflight_stream(
                provider_request, thinking_enabled=thinking_enabled
            )
            stream = provider.stream_response(
                provider_request,
                input_tokens=input_tokens,
                request_id=f"{request_id}_{provider_id}",
                thinking_enabled=thinking_enabled,
            ).__aiter__()
            first_content_timeout = (
                NVIDIA_FIRST_CONTENT_TIMEOUT_SECONDS
                if provider_id == "nvidia_nim"
                else FIRST_CONTENT_TIMEOUT_SECONDS
            )
            deadline = asyncio.get_running_loop().time() + first_content_timeout

            while not committed:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("Provider produced no content before first-token deadline")
                try:
                    async with asyncio.timeout(remaining):
                        raw_event = await anext(stream)
                except StopAsyncIteration:
                    break

                event_name, payload = _event_details(raw_event)
                if _is_error_event(event_name, payload):
                    buffered.append(raw_event)
                    last_uncommitted_events = buffered
                    break

                buffered.append(raw_event)
                if _is_first_content_event(payload):
                    committed = True
                    for pending_event in buffered:
                        yield pending_event
                    buffered.clear()

            if committed:
                async for raw_event in stream:
                    event_name, payload = _event_details(raw_event)
                    if _is_error_event(event_name, payload):
                        yield raw_event
                        return
                    yield raw_event
                return

            last_uncommitted_events = buffered
            if index < len(candidates) - 1:
                logger.warning(
                    "PROVIDER_FALLBACK: provider={} failed before first content; trying next provider={}",
                    provider_id,
                    candidates[index + 1][0],
                )
                continue

        except Exception as exc:
            if committed:
                logger.error(
                    "PROVIDER_STREAM_FAILED_AFTER_CONTENT: provider={} exc_type={}",
                    provider_id,
                    type(exc).__name__,
                )
                yield format_sse_event(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": "The selected provider stopped after beginning its response.",
                        },
                    },
                )
                return
            last_uncommitted_events = buffered
            logger.warning(
                "PROVIDER_FALLBACK: provider={} failed before first content exc_type={}",
                provider_id,
                type(exc).__name__,
            )
            if index < len(candidates) - 1:
                continue
        finally:
            if stream is not None:
                await stream.aclose()

    if last_uncommitted_events:
        for pending_event in last_uncommitted_events:
            yield pending_event
    else:
        yield format_sse_event(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "All configured cloud providers failed before producing a response.",
                },
            },
        )
