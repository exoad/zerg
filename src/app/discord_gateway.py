from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from src.app.webhooks import WebhookManager
from src.config.settings import Settings
from src.core.orchestrator import MIN_SESSION_TURNS, CouncilOrchestrator, SessionRoster
from src.llm.model_catalog import ModelCatalog


logger = logging.getLogger(__name__)


def wire_bot(
    bot: commands.Bot,
    settings: Settings,
    orchestrator: CouncilOrchestrator,
    webhook_manager: WebhookManager,
    model_catalog: ModelCatalog,
) -> None:
    orchestrator.set_emitter(webhook_manager.send_agent_message)

    def _zerg_embed(title: str, description: str, *, color: discord.Color | None = None) -> discord.Embed:
        embed = discord.Embed(
            title=title[:256],
            description=description[:4096],
            color=color or discord.Color.blurple(),
        )
        return embed

    def _format_model_label(provider: str, model: str) -> str:
        cleaned_provider = provider.lower().strip()
        cleaned_model = model.strip()
        if not cleaned_model:
            return cleaned_provider
        if "/" in cleaned_model:
            return cleaned_model
        return f"{cleaned_provider}/{cleaned_model}"

    def _build_roster_embed(roster: SessionRoster) -> discord.Embed:
        description = (
            f"Assignments for epoch {roster.epoch}. "
            f"Unique models: {'yes' if roster.unique_models else 'best-effort fallback'}"
        )
        if roster.catalog_errors:
            description += "\nCatalog returned warnings for one or more providers."

        embed = discord.Embed(
            title=f"Council Roster ({len(roster.entries)} agents)",
            description=description[:4096],
            color=discord.Color.blurple(),
        )

        reserved_fields = 1 if roster.catalog_errors else 0
        max_agent_fields = max(0, 25 - reserved_fields)
        for entry in roster.entries[:max_agent_fields]:
            model_label = _format_model_label(entry.provider, entry.model)
            value = f"Provider: {entry.provider}\nModel: {model_label}\nSource: {entry.source}"
            embed.add_field(name=entry.agent_name[:256], value=value[:1024], inline=True)

        if roster.catalog_errors:
            warning = "; ".join(
                f"{provider}: {error}"
                for provider, error in sorted(roster.catalog_errors.items())
            )
            embed.add_field(name="Catalog Warnings", value=(warning[:1021] + "...") if len(warning) > 1024 else warning, inline=False)

        return embed

    def _is_allowed_channel(channel_id: int | None) -> bool:
        return channel_id == settings.allowed_channel_id

    @bot.tree.command(name="council", description="Start a council session in the locked channel")
    @app_commands.describe(
        topic="Topic for the council debate",
        max_turns="Override default max turns for this session"
    )
    async def council(interaction: discord.Interaction, topic: str, max_turns: int | None = None) -> None:
        if not _is_allowed_channel(interaction.channel_id):
            logger.info("Ignored /council outside allowed channel: %s", interaction.channel_id)
            return

        channel_id = interaction.channel_id
        if channel_id is None:
            await interaction.response.send_message(
                embed=_zerg_embed("Council Error", "Could not resolve channel id.", color=discord.Color.red()),
                ephemeral=True,
            )
            return

        if await orchestrator.has_active_session(channel_id):
            await interaction.response.send_message(
                embed=_zerg_embed(
                    "Council Busy",
                    "A council session is already active in this channel.",
                    color=discord.Color.orange(),
                ),
            )
            return

        try:
            session = await orchestrator.start_session(
                channel_id=channel_id,
                starter_user_id=interaction.user.id,
                starter_name=interaction.user.display_name,
                topic=topic,
                max_turns=max_turns,
            )
        except Exception as exc:
            logger.exception("Failed to start council session")
            error_msg = str(exc).replace("\n", " ").replace("```", "").strip()
            if error_msg.startswith("{") or error_msg.startswith("["):
                error_msg = "Internal startup error (see logs for details)."
            if len(error_msg) > 200:
                error_msg = error_msg[:197].rstrip() + "..."
            await interaction.response.send_message(
                embed=_zerg_embed("Council Error", f"Could not start council: {error_msg}", color=discord.Color.red()),
                ephemeral=True,
            )
            return

        requested_turns = max_turns if max_turns is not None else settings.council_max_turns
        start_desc = (
            "Send follow-up messages in this channel to steer the agents in real time.\n"
            f"Max turns this session: {session.max_turns}"
        )
        if requested_turns < MIN_SESSION_TURNS:
            start_desc += f"\nRequested max turns ({requested_turns}) was clamped to minimum {MIN_SESSION_TURNS}."

        embeds: list[discord.Embed] = [
            _zerg_embed(
                "Council Started",
                start_desc,
                color=discord.Color.green(),
            )
        ]

        roster = await orchestrator.get_session_roster(channel_id)
        if roster is not None:
            embeds.append(_build_roster_embed(roster))

        await interaction.response.send_message(embeds=embeds)

    @bot.event
    async def on_ready() -> None:
        logger.info("Logged in as %s", bot.user)
        try:
            await bot.tree.sync()
            logger.info("Slash commands synced")
        except Exception:
            logger.exception("Failed to sync slash commands")

    @bot.tree.command(name="models", description="Fetch available model IDs for a provider")
    @app_commands.describe(provider="Provider name: groq, mistral, cerebras, sambanova, google, or all")
    async def models(interaction: discord.Interaction, provider: str = "sambanova") -> None:
        if not _is_allowed_channel(interaction.channel_id):
            logger.info("Ignored /models outside allowed channel: %s", interaction.channel_id)
            return

        target = provider.lower().strip()
        if target == "all":
            results = await model_catalog.fetch_models_all()
            lines: list[str] = ["Model discovery results:"]
            for result in results:
                if result.error:
                    lines.append(f"- {result.provider}: {result.error}")
                    continue
                preview = ", ".join(result.models[:6])
                suffix = "" if len(result.models) <= 6 else f" ... (+{len(result.models) - 6} more)"
                lines.append(f"- {result.provider}: {preview}{suffix}")
            await interaction.response.send_message(
                embed=_zerg_embed("Model Discovery", "\n".join(lines)[:4000]),
            )
            return

        result = await model_catalog.fetch_models(target)
        if result.error:
            await interaction.response.send_message(
                embed=_zerg_embed(
                    "Model Discovery Error",
                    f"Could not fetch models for {target}: {result.error}",
                    color=discord.Color.red(),
                ),
            )
            return

        lines = [f"Models for {target} ({len(result.models)}):"]
        lines.extend(f"- {model_id}" for model_id in result.models[:20])
        if len(result.models) > 20:
            lines.append(f"... and {len(result.models) - 20} more")
        await interaction.response.send_message(
            embed=_zerg_embed("Model Discovery", "\n".join(lines)[:4000]),
        )

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot:
            return

        if not _is_allowed_channel(message.channel.id):
            return

        if not await orchestrator.has_active_session(message.channel.id):
            return

        if not await orchestrator.is_starter(message.channel.id, message.author.id):
            return

        accepted = await orchestrator.add_human_steer(
            channel_id=message.channel.id,
            user_id=message.author.id,
            user_name=message.author.display_name,
            text=message.content,
        )
        if accepted:
            try:
                await message.reply(
                    embed=_zerg_embed(
                        "Steer Received",
                        "Agents are adapting to your latest instruction.",
                        color=discord.Color.green(),
                    ),
                    mention_author=False,
                )
            except discord.HTTPException:
                pass

    async def _close_webhooks() -> None:
        await webhook_manager.close()
        await model_catalog.close()

    original_close = bot.close

    async def wrapped_close() -> None:
        await _close_webhooks()
        await original_close()

    bot.close = wrapped_close  # type: ignore[assignment]
