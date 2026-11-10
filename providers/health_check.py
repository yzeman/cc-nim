"""Safe live health checks for configured cloud model providers."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from api.models.anthropic import Message, MessagesRequest
from config.provider_catalog import PROVIDER_CATALOG
from config.settings import Settings, get_settings
from providers.exceptions import AuthenticationError
from providers.registry import build_provider_config, create_provider


CLOUD_PROVIDER_IDS = ("nvidia_nim", "open_router", "deepseek")
HEALTH_CHECK_TIMEOUT_SECONDS = 20


def _configured_cloud_models(settings: Settings) -> dict[str, str]:
    """Return one configured model per cloud provider, without exposing credentials."""
    configured = (
        settings.model_opus,
        settings.model_sonnet,
        settings.model_haiku,
        settings.model,
    )
    result: dict[str, str] = {}
    for value in configured:
        if not value or "/" not in value:
            continue
        provider_id, model_name = value.split("/", 1)
        if provider_id in CLOUD_PROVIDER_IDS and model_name and provider_id not in result:
            result[provider_id] = model_name
    return result


def _safe_provider_error(payload: dict[str, Any] | None) -> str:
    """Classify common upstream failures without exposing raw provider messages."""
    error = (payload or {}).get("error", {})
    if not isinstance(error, dict):
        return "Provider returned an API error."
    kind = str(error.get("type", "")).lower()
    message = str(error.get("message", "")).lower()
    combined = f"{kind} {message}"
    if any(term in combined for term in ("401", "403", "authentication", "unauthorized", "invalid api key")):
        return "API key was rejected."
    if any(term in combined for term in ("410", "model not found", "model unavailable", "retired")):
        return "Selected model is unavailable or retired."
    if any(term in combined for term in ("429", "rate limit", "too many requests")):
        return "Provider rate limit was reached."
    if any(term in combined for term in ("timeout", "timed out")):
        return "Provider did not respond before the timeout."
    if any(term in combined for term in ("503", "502", "overloaded", "capacity")):
        return "Provider is temporarily overloaded."
    return "Provider returned an API error."


def _parse_sse_event(raw_event: str) -> tuple[str | None, dict[str, Any] | None]:
    """Parse one complete Anthropic SSE event."""
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
        payload = json.loads("\n".join(data_lines))
    except (TypeError, ValueError):
        return event_name, None
    return event_name, payload if isinstance(payload, dict) else None


async def check_cloud_provider(
    provider_id: str,
    model_name: str,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Make a tiny real streaming request and return a credential-safe result."""
    settings = settings or get_settings()
    started = time.monotonic()
    provider = None
    result: dict[str, Any] = {
        "provider_id": provider_id,
        "model": model_name,
        "status": "failed",
        "reason": "The provider did not return a response.",
        "latency_ms": None,
        "first_token_ms": None,
        "text_chunks": 0,
    }

    if provider_id not in CLOUD_PROVIDER_IDS or provider_id not in PROVIDER_CATALOG:
        result["reason"] = "This cloud provider is not supported by the health check."
        return result

    try:
        # Validate that the configured key exists without ever logging or returning it.
        descriptor = PROVIDER_CATALOG[provider_id]
        build_provider_config(descriptor, settings)
    except AuthenticationError:
        result["status"] = "not_configured"
        result["reason"] = "API key is missing in the app environment."
        return result
    except Exception:
        result["reason"] = "Provider configuration could not be loaded."
        return result

    try:
        provider = create_provider(provider_id, settings)
        request = MessagesRequest(
            model=model_name,
            max_tokens=24,
            stream=True,
            messages=[Message(role="user", content="Reply with exactly: OK")],
            thinking={"enabled": False, "type": "disabled"},
            tools=[],
        )
        text_parts: list[str] = []
        async with asyncio.timeout(HEALTH_CHECK_TIMEOUT_SECONDS):
            async for raw_event in provider.stream_response(
                request,
                input_tokens=8,
                request_id=f"health_{uuid.uuid4().hex[:10]}",
                thinking_enabled=False,
            ):
                event_name, payload = _parse_sse_event(raw_event)
                if event_name == "error" or (payload and payload.get("type") == "error"):
                    error_type = (
                        (payload or {}).get("error", {}).get("type", "provider_error")
                        if isinstance((payload or {}).get("error", {}), dict)
                        else "provider_error"
                    )
                    result["reason"] = _safe_provider_error(payload)
                    return result

                delta = (payload or {}).get("delta", {})
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    text = delta.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                        result["text_chunks"] += 1
                        if result["first_token_ms"] is None:
                            result["first_token_ms"] = round(
                                (time.monotonic() - started) * 1000
                            )

                if event_name == "message_stop":
                    break

        if not text_parts:
            result["reason"] = "Connected, but the model returned no answer text."
            return result

        result["status"] = "working"
        result["reason"] = "A real answer was received through the streaming connection."
        result["latency_ms"] = round((time.monotonic() - started) * 1000)
        result["streaming"] = True
        return result

    except TimeoutError:
        result["reason"] = (
            f"No answer arrived within {HEALTH_CHECK_TIMEOUT_SECONDS} seconds."
        )
        return result
    except Exception as exc:
        # Exception messages can contain request details; expose only the class.
        result["reason"] = f"Connection/request failed ({type(exc).__name__})."
        return result
    finally:
        if provider is not None:
            try:
                await provider.cleanup()
            except Exception:
                pass


async def check_configured_cloud_providers(
    *, settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Check each configured cloud provider once, sequentially."""
    settings = settings or get_settings()
    models = _configured_cloud_models(settings)
    results: list[dict[str, Any]] = []
    for provider_id in CLOUD_PROVIDER_IDS:
        model_name = models.get(provider_id)
        if model_name is None:
            results.append(
                {
                    "provider_id": provider_id,
                    "model": None,
                    "status": "not_assigned",
                    "reason": "No model is currently assigned to this provider.",
                    "latency_ms": None,
                    "first_token_ms": None,
                    "text_chunks": 0,
                }
            )
            continue
        results.append(
            await check_cloud_provider(provider_id, model_name, settings=settings)
        )
    return results
