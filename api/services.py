"""Application services for the Claude-compatible API."""

from __future__ import annotations

import json
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

from .model_router import ModelRouter
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .provider_fallback import stream_with_provider_fallback
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion (not native Messages).
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim"})


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )



async def anthropic_json_response(
    body: AsyncIterator[str], *, fallback_model: str
) -> JSONResponse:
    """Collect Anthropic SSE events into the JSON response required by stream=false."""
    message: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    tool_json: dict[int, str] = {}
    stop_reason: str | None = None
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    async for raw_event in body:
        event_name: str | None = None
        data_lines: list[str] = []
        for line in raw_event.splitlines():
            if line.startswith("event:"):
                event_name = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                data_lines.append(line.partition(":")[2].strip())
        if not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue

        if event_name == "error" or payload.get("type") == "error":
            error = payload.get("error") or {}
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {
                        "type": error.get("type", "api_error"),
                        "message": error.get("message", "All providers failed."),
                    },
                },
            )
        if event_name == "message_start":
            message = payload.get("message") or {}
            start_usage = message.get("usage") or {}
            usage["input_tokens"] = int(start_usage.get("input_tokens") or 0)
        elif event_name == "content_block_start":
            index = payload.get("index")
            block = payload.get("content_block")
            if isinstance(index, int) and isinstance(block, dict):
                blocks[index] = dict(block)
                if block.get("type") == "tool_use":
                    tool_json[index] = ""
        elif event_name == "content_block_delta":
            index = payload.get("index")
            delta = payload.get("delta") or {}
            if not isinstance(index, int) or not isinstance(delta, dict):
                continue
            block = blocks.get(index)
            if block is None:
                continue
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = block.get("text", "") + str(delta.get("text", ""))
            elif delta_type == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + str(delta.get("thinking", ""))
            elif delta_type == "signature_delta":
                block["signature"] = block.get("signature", "") + str(delta.get("signature", ""))
            elif delta_type == "input_json_delta":
                tool_json[index] = tool_json.get(index, "") + str(delta.get("partial_json", ""))
        elif event_name == "message_delta":
            delta = payload.get("delta") or {}
            stop_reason = delta.get("stop_reason")
            final_usage = payload.get("usage") or {}
            usage["output_tokens"] = int(final_usage.get("output_tokens") or 0)
            usage["input_tokens"] = int(final_usage.get("input_tokens") or usage["input_tokens"])

    if not message.get("id"):
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Provider stream ended without a complete Anthropic message.",
                },
            },
        )

    content: list[dict[str, Any]] = []
    for index in sorted(blocks):
        block = blocks[index]
        if block.get("type") == "tool_use":
            try:
                block["input"] = json.loads(tool_json.get(index, "") or "{}")
            except (TypeError, ValueError):
                block["input"] = {}
        content.append(block)

    return JSONResponse(
        content={
            "id": message["id"],
            "type": "message",
            "role": "assistant",
            "model": message.get("model") or fallback_model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter

    def create_message(self, request_data: MessagesRequest) -> object:
        """Create a message response or streaming response."""
        try:
            _require_non_empty_messages(request_data.messages)

            routed = self._model_router.resolve_messages_request(request_data)
            if routed.resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                tool_err = openai_chat_upstream_server_tool_error(
                    routed.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    raise InvalidRequestError(tool_err)

            if self._settings.enable_web_server_tools and is_web_server_tool_request(
                routed.request
            ):
                input_tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info("Optimization: Handling Anthropic web server tool")
                egress = WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
                    allowed_schemes=self._settings.web_fetch_allowed_scheme_set(),
                )
                return anthropic_sse_streaming_response(
                    stream_web_server_tool_response(
                        routed.request,
                        input_tokens=input_tokens,
                        web_fetch_egress=egress,
                        verbose_client_errors=self._settings.log_api_error_tracebacks,
                    ),
                )

            optimized = try_optimizations(routed.request, self._settings)
            if optimized is not None:
                return optimized
            logger.debug("No optimization matched, routing to provider")

            request_id = f"req_{uuid.uuid4().hex[:12]}"
            logger.info(
                "API_REQUEST: request_id={} provider={} model={} messages={}",
                request_id,
                routed.resolved.provider_id,
                routed.request.model,
                len(routed.request.messages),
            )
            if self._settings.log_raw_api_payloads:
                logger.debug(
                    "FULL_PAYLOAD [{}]: {}", request_id, routed.request.model_dump()
                )

            input_tokens = self._token_counter(
                routed.request.messages, routed.request.system, routed.request.tools
            )
            return anthropic_sse_streaming_response(
                stream_with_provider_fallback(
                    routed.request,
                    primary_provider_id=routed.resolved.provider_id,
                    primary_model=routed.resolved.provider_model,
                    settings=self._settings,
                    provider_getter=self._provider_getter,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    thinking_enabled=routed.resolved.thinking_enabled,
                ),
            )

        except ProviderError:
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info(
                    "COUNT_TOKENS: request_id={} model={} messages={} input_tokens={}",
                    request_id,
                    routed.request.model,
                    len(routed.request.messages),
                    tokens,
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as e:
                _log_unexpected_service_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e
