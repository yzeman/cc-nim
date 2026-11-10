"""Cloud provider diagnostics command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.health_check import check_configured_cloud_providers

if TYPE_CHECKING:
    from messaging.handler import ClaudeMessageHandler
    from messaging.models import IncomingMessage


_PROVIDER_LABELS = {
    "nvidia_nim": "NVIDIA NIM",
    "open_router": "OpenRouter",
    "deepseek": "DeepSeek",
}


async def handle_providers_command(
    handler: ClaudeMessageHandler, incoming: IncomingMessage
) -> None:
    """Run a real short streaming request against each configured cloud provider."""
    ctx = handler.get_render_ctx()
    waiting_id = await handler.platform.queue_send_message(
        incoming.chat_id,
        handler.format_status("🩺", "Checking cloud providers", "This may take up to a minute..."),
        fire_and_forget=False,
        message_thread_id=incoming.message_thread_id,
    )

    if waiting_id:
        handler.record_outgoing_message(
            incoming.platform, incoming.chat_id, waiting_id, "command"
        )

    results = await check_configured_cloud_providers()
    lines = ["🩺 " + ctx.bold("Cloud provider health")]
    for result in results:
        provider = _PROVIDER_LABELS.get(result["provider_id"], result["provider_id"])
        status = result["status"]
        if status == "working":
            icon = "✅"
            label = "Working"
            details = (
                f"model {result['model']}; first text in "
                f"{result['first_token_ms']} ms"
            )
            if result.get("latency_ms") is not None:
                details += f", finished in {result['latency_ms']} ms"
            details += f", {result['text_chunks']} text chunk(s)"
        elif status == "not_configured":
            icon = "⚪"
            label = "Not configured"
            details = result["reason"]
        elif status == "not_assigned":
            icon = "⚪"
            label = "Not assigned"
            details = result["reason"]
        else:
            icon = "❌"
            label = "Failed"
            details = result["reason"]
            if result.get("model"):
                details = f"model {result['model']}; {details}"

        lines.append(
            f"{icon} {ctx.bold(ctx.escape_text(provider))} — "
            f"{ctx.escape_text(label)}\n"
            f"   {ctx.escape_text(details)}"
        )

    lines.append("")
    lines.append(
        ctx.escape_text(
            "This test sends a tiny request and checks that answer text arrives "
            "through the streaming connection. It does not use your coding workspace."
        )
    )
    lines.append(
        ctx.escape_text(
            "Local providers are not tested here; they remain ready for later setup."
        )
    )
    report = "\n".join(lines)

    if waiting_id:
        await handler.platform.queue_edit_message(
            incoming.chat_id,
            waiting_id,
            report,
            parse_mode=handler._parse_mode(),
            fire_and_forget=False,
        )
    else:
        msg_id = await handler.platform.queue_send_message(
            incoming.chat_id,
            report,
            parse_mode=handler._parse_mode(),
            fire_and_forget=False,
            message_thread_id=incoming.message_thread_id,
        )
        handler.record_outgoing_message(
            incoming.platform, incoming.chat_id, msg_id, "command"
        )
