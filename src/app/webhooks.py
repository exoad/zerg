from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
import discord

from src.core.agents import AgentConfig

logger = logging.getLogger(__name__)

class WebhookManager:
    def __init__(
        self,
        bot: discord.Client,
        allowed_channel_id: int,
        default_webhook_url: str | None = None,
    ) -> None:
        self.bot = bot
        self.allowed_channel_id = allowed_channel_id
        self.default_webhook_url = default_webhook_url
        self._session: aiohttp.ClientSession | None = None
        self._cache: dict[int, discord.Webhook] = {}
        self._default_webhook: discord.Webhook | None = None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send_agent_message(
        self,
        channel_id: int,
        agent: AgentConfig,
        content: str,
        embed_payloads: list[dict[str, Any]] | None = None,
    ) -> int | None:
        if channel_id != self.allowed_channel_id:
            logger.warning(
                "Blocked outbound webhook message to non-allowed channel %s (allowed: %s)",
                channel_id,
                self.allowed_channel_id,
            )
            return None

        if not content.strip() and not embed_payloads:
            return None

        webhook, thread = await self._resolve_webhook(channel_id)
        if webhook is None:
            logger.warning("No webhook available for channel %s", channel_id)
            return None

        kwargs: dict[str, object] = {
            "username": agent.name[:80],
            "avatar_url": agent.avatar_url,
            "wait": True,
            "allowed_mentions": discord.AllowedMentions.none(),
        }
        if content.strip():
            clipped = content.strip()
            if len(clipped) > 2000:
                clipped = clipped[:1997] + "..."
            kwargs["content"] = clipped
        if embed_payloads:
            kwargs["embeds"] = [self._payload_to_embed(payload) for payload in embed_payloads[:10]]
        if thread is not None:
            kwargs["thread"] = thread

        try:
            for attempt in range(2):
                try:
                    message: discord.WebhookMessage = await webhook.send(**kwargs)
                    logger.info(
                        "Delivered webhook message for %s to channel %s (message_id=%s)",
                        agent.name,
                        channel_id,
                        message.id,
                    )
                    return message.id
                except discord.HTTPException as exc:
                    status = getattr(exc, "status", None)
                    can_retry = attempt == 0 and status in {429, 500, 502, 503, 504}
                    if can_retry:
                        logger.warning(
                            "Webhook send transient failure for %s (status=%s). Retrying once.",
                            agent.name,
                            status,
                        )
                        await asyncio.sleep(1.0)
                        continue
                    logger.warning(
                        "Failed to send webhook message for %s in channel %s (status=%s): %s",
                        agent.name,
                        channel_id,
                        status,
                        exc,
                    )
                    return None
        except Exception as exc:
            logger.warning("Unexpected webhook send failure for %s in channel %s: %s", agent.name, channel_id, exc)
            return None

    async def _resolve_webhook(self, channel_id: int) -> tuple[discord.Webhook | None, discord.Thread | None]:
        if self.default_webhook_url:
            if self._default_webhook is None:
                session = await self._get_session()
                self._default_webhook = discord.Webhook.from_url(
                    self.default_webhook_url,
                    session=session,
                )
            return self._default_webhook, None

        channel = self.bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except discord.HTTPException:
                return None, None

        thread: discord.Thread | None = None
        base_channel: discord.TextChannel | None = None

        if isinstance(channel, discord.Thread):
            thread = channel
            if isinstance(channel.parent, discord.TextChannel):
                base_channel = channel.parent
        elif isinstance(channel, discord.TextChannel):
            base_channel = channel

        if base_channel is None:
            return None, None

        cached = self._cache.get(base_channel.id)
        if cached is not None:
            return cached, thread

        try:
            hooks = await base_channel.webhooks()
        except discord.Forbidden:
            logger.warning("Missing permission to list webhooks in channel %s", base_channel.id)
            return None, thread

        webhook = next((h for h in hooks if h.name == "Council Relay" and h.token), None)
        if webhook is None:
            try:
                webhook = await base_channel.create_webhook(name="Council Relay")
            except discord.Forbidden:
                logger.warning("Missing permission to create webhooks in channel %s", base_channel.id)
                return None, thread

        self._cache[base_channel.id] = webhook
        return webhook, thread

    @staticmethod
    def _payload_to_embed(payload: dict[str, Any]) -> discord.Embed:
        color = int(payload.get("color", 0x5865F2))
        embed = discord.Embed(
            title=str(payload.get("title", ""))[:256] or None,
            description=str(payload.get("description", ""))[:4096] or None,
            color=discord.Color(color),
        )

        fields = payload.get("fields", [])
        if isinstance(fields, list):
            for field in fields[:25]:
                if not isinstance(field, dict):
                    continue
                name = str(field.get("name", ""))[:256] or "Field"
                value = str(field.get("value", ""))[:1024] or "n/a"
                inline = bool(field.get("inline", False))
                embed.add_field(name=name, value=value, inline=inline)

        footer = str(payload.get("footer", "")).strip()
        if footer:
            embed.set_footer(text=footer[:2048])

        return embed
