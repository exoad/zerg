from __future__ import annotations

import logging
from pathlib import Path

import discord
from discord.ext import commands

from src.app.discord_gateway import wire_bot
from src.app.webhooks import WebhookManager
from src.config.settings import get_settings
from src.core.agents import load_agents
from src.core.model_assignment import ModelAssignmentManager
from src.core.orchestrator import CouncilOrchestrator
from src.llm.rate_limits import RateLimitManager
from src.llm.router import LLMRouter
from src.llm.model_catalog import ModelCatalog


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_bot() -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True
    return commands.Bot(command_prefix=commands.when_mentioned, intents=intents)


def main() -> None:
    settings = get_settings()
    _configure_logging(settings.log_level)
    if not settings.discord_bot_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is required. Set it in .env")
    agents_path = Path(__file__).resolve().parents[1] / "config" / "agents.yaml"
    registry = load_agents(agents_path)
    rate_limits = RateLimitManager(
        groq_rpm=settings.groq_requests_per_min_budget,
        mistral_rpm=settings.mistral_requests_per_min_budget,
        cerebras_rpm=settings.cerebras_requests_per_min_budget,
        sambanova_rpm=settings.sambanova_requests_per_min_budget,
        google_rpm=settings.google_requests_per_min_budget,
    )
    router = LLMRouter(settings, rate_limits)
    model_catalog = ModelCatalog(settings)
    assignment_manager = ModelAssignmentManager(model_catalog)
    orchestrator = CouncilOrchestrator(settings, registry, router, assignment_manager)
    bot = _build_bot()
    webhook_manager = WebhookManager(
        bot,
        allowed_channel_id=settings.allowed_channel_id,
        default_webhook_url=settings.default_discord_webhook_url,
    )
    wire_bot(bot, settings, orchestrator, webhook_manager, model_catalog)
    bot.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
