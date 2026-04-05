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
    agent_min_cooldown_seconds: float
    global_max_messages_per_minute: int
    agent_min_words: int
    agent_max_words: int
    attack_intensity: str
    taunt_intensity: str
    allow_profanity: bool
    enforce_taunt_profanity: bool
    discussion_mode_auto: bool
    default_discussion_mode: str
    objective_profanity_policy: str
    objective_simple_min_words: int
    objective_simple_max_words: int
    objective_min_words: int
    objective_max_words: int
    objective_simple_max_corrections: int
    debatable_min_words: int
    debatable_max_words: int
    adversarial_min_words: int
    adversarial_max_words: int
    duplicate_similarity_threshold: float
    rebuttal_targeting_enabled: bool
    stance_retry_enabled: bool
    steer_bootstrap_required_agents: int
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
    # Intentionally env-driven (secrets/private routing only).
    discord_bot_token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    default_discord_webhook_url = os.getenv("DEFAULT_DISCORD_WEBHOOK_URL", "").strip() or None
    allowed_channel_id = _get_int("DISCORD_ALLOWED_CHANNEL_ID", 1490122012574748814)

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    mistral_api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    cerebras_api_key = os.getenv("CEREBRAS_API_KEY", "").strip()
    sambanova_api_key = os.getenv("SAMBANOVA_API_KEY", "").strip()
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    # Optional but useful runtime override.
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    # All non-secret/non-private settings are code-defined defaults.
    return Settings(
        discord_bot_token=discord_bot_token,
        default_discord_webhook_url=default_discord_webhook_url,
        allowed_channel_id=allowed_channel_id,
        groq_api_key=groq_api_key,
        mistral_api_key=mistral_api_key,
        cerebras_api_key=cerebras_api_key,
        sambanova_api_key=sambanova_api_key,
        google_api_key=google_api_key,
        groq_api_base="https://api.groq.com/openai/v1",
        mistral_api_base="https://api.mistral.ai/v1",
        cerebras_api_base="https://api.cerebras.ai/v1",
        sambanova_api_base="https://api.sambanova.ai/v1",
        google_api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        google_models_api_base="https://generativelanguage.googleapis.com/v1beta",
        council_max_turns=12,
        consensus_threshold=0.82,
        consensus_min_recent_messages=5,
        consensus_min_unique_agents=3,
        consensus_min_turns=6,
        consensus_pass_count_required=0,
        consensus_lexical_weight=1.0,
        consensus_stance_weight=0.0,
        consensus_freshness_weight=0.0,
        consensus_disagreement_guard_enabled=False,
        agent_response_max_tokens=1500,
        agent_summarize_max_tokens=768,
        agent_response_timeout_sec=35,
        message_delay_seconds=2.0,
        agent_min_cooldown_seconds=6.0,
        global_max_messages_per_minute=24,
        agent_min_words=100,
        agent_max_words=200,
        attack_intensity="high",
        taunt_intensity="medium",
        allow_profanity=True,
        enforce_taunt_profanity=True,
        discussion_mode_auto=True,
        default_discussion_mode="debatable",
        objective_profanity_policy="discourage",
        objective_simple_min_words=5,
        objective_simple_max_words=40,
        objective_min_words=70,
        objective_max_words=150,
        objective_simple_max_corrections=1,
        debatable_min_words=90,
        debatable_max_words=190,
        adversarial_min_words=100,
        adversarial_max_words=220,
        duplicate_similarity_threshold=0.58,
        rebuttal_targeting_enabled=True,
        stance_retry_enabled=True,
        steer_bootstrap_required_agents=5,
        agent_debug_embeds_enabled=True,
        allow_human_interrupt=True,
        steer_preemption_mode="cancel_if_possible_else_mark_stale",
        max_context_events=30,
        max_context_chars=18000,
        scheduler_mode="weighted_round_robin",
        groq_requests_per_min_budget=30,
        mistral_requests_per_min_budget=12,
        cerebras_requests_per_min_budget=20,
        sambanova_requests_per_min_budget=15,
        google_requests_per_min_budget=8,
        provider_retry_max_attempts=3,
        provider_backoff_base_seconds=1,
        force_finalize_timeout_minutes=120,
        consensus_model="groq/llama-3.1-8b-instant",
        log_level=log_level,
    )
