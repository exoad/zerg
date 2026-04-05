from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    discord_bot_token: str
    default_discord_webhook_url: str | None
    allowed_channel_id: int

    groq_api_key: str
    mistral_api_key: str
    cerebras_api_key: str
    sambanova_api_key: str
    google_api_key: str

    groq_api_base: str
    mistral_api_base: str
    cerebras_api_base: str
    sambanova_api_base: str
    google_api_base: str
    google_models_api_base: str

    council_max_turns: int
    consensus_threshold: float
    consensus_min_recent_messages: int
    consensus_min_unique_agents: int
    consensus_min_turns: int
    consensus_pass_count_required: int
    consensus_lexical_weight: float
    consensus_stance_weight: float
    consensus_freshness_weight: float
    consensus_disagreement_guard_enabled: bool
    agent_response_max_tokens: int
    agent_summarize_max_tokens: int
    agent_response_timeout_sec: int
    message_delay_seconds: float
    agent_debug_embeds_enabled: bool

    allow_human_interrupt: bool
    steer_preemption_mode: str

    max_context_events: int
    max_context_chars: int

    scheduler_mode: str

    groq_requests_per_min_budget: int
    mistral_requests_per_min_budget: int
    cerebras_requests_per_min_budget: int
    sambanova_requests_per_min_budget: int
    google_requests_per_min_budget: int
    provider_retry_max_attempts: int
    provider_backoff_base_seconds: int

    force_finalize_timeout_minutes: int
    consensus_model: str

    log_level: str


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        discord_bot_token=os.getenv("DISCORD_BOT_TOKEN", "").strip(),
        default_discord_webhook_url=os.getenv("DEFAULT_DISCORD_WEBHOOK_URL", "").strip() or None,
        allowed_channel_id=_get_int("DISCORD_ALLOWED_CHANNEL_ID", 1490122012574748814),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        mistral_api_key=os.getenv("MISTRAL_API_KEY", "").strip(),
        cerebras_api_key=os.getenv("CEREBRAS_API_KEY", "").strip(),
        sambanova_api_key=os.getenv("SAMBANOVA_API_KEY", "").strip(),
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        groq_api_base=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").strip(),
        mistral_api_base=os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1").strip(),
        cerebras_api_base=os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1").strip(),
        sambanova_api_base=os.getenv("SAMBANOVA_API_BASE", "https://api.sambanova.ai/v1").strip(),
        google_api_base=os.getenv("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai").strip(),
        google_models_api_base=os.getenv("GOOGLE_MODELS_API_BASE", "https://generativelanguage.googleapis.com/v1beta").strip(),
        council_max_turns=_get_int("COUNCIL_MAX_TURNS", 20),
        consensus_threshold=_get_float("CONSENSUS_THRESHOLD", 1.01),
        consensus_min_recent_messages=_get_int("CONSENSUS_MIN_RECENT_MESSAGES", 5),
        consensus_min_unique_agents=_get_int("CONSENSUS_MIN_UNIQUE_AGENTS", 3),
        consensus_min_turns=_get_int("CONSENSUS_MIN_TURNS", 6),
        consensus_pass_count_required=_get_int("CONSENSUS_PASS_COUNT_REQUIRED", 0),
        consensus_lexical_weight=_get_float("CONSENSUS_LEXICAL_WEIGHT", 1.0),
        consensus_stance_weight=_get_float("CONSENSUS_STANCE_WEIGHT", 0.0),
        consensus_freshness_weight=_get_float("CONSENSUS_FRESHNESS_WEIGHT", 0.0),
        consensus_disagreement_guard_enabled=_get_bool("CONSENSUS_DISAGREEMENT_GUARD_ENABLED", False),
        agent_response_max_tokens=_get_int("AGENT_RESPONSE_MAX_TOKENS", 1500),
        agent_summarize_max_tokens=_get_int("AGENT_SUMMARIZE_MAX_TOKENS", 768),
        agent_response_timeout_sec=_get_int("AGENT_RESPONSE_TIMEOUT_SEC", 35),
        message_delay_seconds=_get_float("MESSAGE_DELAY_SECONDS", 2.0),
        agent_debug_embeds_enabled=_get_bool("AGENT_DEBUG_EMBEDS_ENABLED", True),
        allow_human_interrupt=_get_bool("ALLOW_HUMAN_INTERRUPT", True),
        steer_preemption_mode=os.getenv("STEER_PREEMPTION_MODE", "cancel_if_possible_else_mark_stale").strip(),
        max_context_events=_get_int("MAX_CONTEXT_EVENTS", 18),
        max_context_chars=_get_int("MAX_CONTEXT_CHARS", 18000),
        scheduler_mode=os.getenv("SCHEDULER_MODE", "weighted_round_robin").strip(),
        groq_requests_per_min_budget=_get_int("GROQ_REQUESTS_PER_MIN_BUDGET", 30),
        mistral_requests_per_min_budget=_get_int("MISTRAL_REQUESTS_PER_MIN_BUDGET", 12),
        cerebras_requests_per_min_budget=_get_int("CEREBRAS_REQUESTS_PER_MIN_BUDGET", 20),
        sambanova_requests_per_min_budget=_get_int("SAMBANOVA_REQUESTS_PER_MIN_BUDGET", 15),
        google_requests_per_min_budget=_get_int("GOOGLE_REQUESTS_PER_MIN_BUDGET", 8),
        provider_retry_max_attempts=_get_int("PROVIDER_RETRY_MAX_ATTEMPTS", 3),
        provider_backoff_base_seconds=_get_int("PROVIDER_BACKOFF_BASE_SECONDS", 1),
        force_finalize_timeout_minutes=_get_int("FORCE_FINALIZE_TIMEOUT_MINUTES", 120),
        consensus_model=os.getenv("CONSENSUS_MODEL", "groq/llama-3.1-8b-instant").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
    )
