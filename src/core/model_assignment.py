from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from src.core.agents import AgentConfig
from src.llm.model_catalog import ModelCatalog


logger = logging.getLogger(__name__)

# Provider preference ranking based on free-tier generosity (April 2026).
# Higher index = preferred. Used to bias randomization toward generous providers.
# Groq/Cerebras: 14,400 RPD, 30 RPM
# Mistral: 60 RPM, 1B tokens/month
# SambaNova: 20 RPM
# Google excluded: 10 RPM, 20 RPD (too restrictive for multi-agent workloads)
PROVIDER_GENEROSITY: dict[str, int] = {
    "groq": 5,
    "cerebras": 5,
    "mistral": 4,
    "sambanova": 3,
}

# Curated pools of known-good, free-tier models per provider.
# These are verified working models with generous rate limits.
# The assignment system draws from these first, falling back to the live catalog.
# Model IDs are bare names (no provider prefix) — normalization happens at assignment time.
CURATED_MODEL_POOLS: dict[str, list[str]] = {
    "groq": [
        "llama-3.1-8b-instant",        # Fast, reliable, 14,400 RPD
        "llama-3.3-70b-versatile",     # High quality, 1,000 RPD
        "qwen-3-32b",                  # Strong reasoning, 60 RPM
    ],
    "cerebras": [
        "llama3.1-8b",                 # Fast, reliable
        "llama-3.3-70b-instruct",      # High quality
        "qwen-3-32b",                  # Strong reasoning
    ],
    "mistral": [
        "mistral-small-latest",        # Good balance
        "ministral-8b-latest",         # Fast, good for analysis
        "mistral-medium-latest",       # Higher quality when needed
    ],
    "sambanova": [
        "Meta-Llama-3.1-8B-Instruct",  # Reliable
        "Meta-Llama-3.3-70B-Instruct", # High quality
        "Qwen2.5-72B-Instruct",        # Strong reasoning
    ],
}


@dataclass(frozen=True)
class AssignedModel:
    provider: str
    model: str
    source: str  # "curated", "catalog", or "fallback"


@dataclass(frozen=True)
class AssignmentSet:
    by_agent: dict[str, AssignedModel]
    catalog_errors: dict[str, str]
    used_unique_models: bool
    generated_at: datetime


class ModelAssignmentManager:
    """Builds per-session model assignments for active agents.

    Policy:
    - Prefer curated pools of known-good models per provider.
    - Fall back to live provider catalogs when curated pools are exhausted.
    - Keep model IDs unique across active agents whenever possible.
    - Allow provider reuse.
    - If both curated and catalog are insufficient, fill with configured defaults.
    """

    def __init__(self, model_catalog: ModelCatalog) -> None:
        self._model_catalog = model_catalog
        self._rng = random.SystemRandom()

    @staticmethod
    def _normalize_model(provider: str, model: str) -> str:
        cleaned_provider = provider.lower().strip()
        cleaned_model = model.strip()
        if not cleaned_model:
            pool = CURATED_MODEL_POOLS.get(cleaned_provider)
            cleaned_model = pool[0] if pool else "llama-3.1-8b-instant"
        if "/" in cleaned_model:
            return cleaned_model
        return f"{cleaned_provider}/{cleaned_model}"

    @staticmethod
    def _is_eligible_catalog_model(model_id: str) -> bool:
        lowered = model_id.lower()
        blocked_fragments = (
            "embedding",
            "embed",
            "tts",
            "text-to-speech",
            "audio",
            "transcription",
            "speech",
            "rerank",
            "reranker",
            "moderation",
            "voxtral",
            "whisper",
        )
        return not any(fragment in lowered for fragment in blocked_fragments)

    def _build_curated_pool(self, used_models: set[str]) -> list[tuple[str, str]]:
        """Build a pool of curated models, excluding already-used ones."""
        pool: list[tuple[str, str]] = []
        for provider, models in CURATED_MODEL_POOLS.items():
            for model_name in models:
                normalized = self._normalize_model(provider, model_name)
                if normalized and normalized not in used_models:
                    pool.append((provider, normalized))
        return pool

    def _build_catalog_pool(
        self, results: list, used_models: set[str]
    ) -> tuple[list[tuple[str, str]], int]:
        """Build a pool from the live catalog, excluding already-used models."""
        pool: list[tuple[str, str]] = []
        seen_models: set[str] = set(used_models)
        skipped_ineligible = 0
        for result in results:
            for model in result.models:
                normalized_model = self._normalize_model(result.provider, model)
                if not normalized_model or "/" not in normalized_model:
                    continue
                if not self._is_eligible_catalog_model(normalized_model):
                    skipped_ineligible += 1
                    continue
                if normalized_model in seen_models:
                    continue
                seen_models.add(normalized_model)
                pool.append((result.provider, normalized_model))
        return pool, skipped_ineligible

    @staticmethod
    def _score_pool_item(item: tuple[str, str]) -> tuple[int, int]:
        """Score a pool item for sorting (higher = preferred)."""
        provider = item[0].lower()
        model_id = item[1].lower()
        prov_score = PROVIDER_GENEROSITY.get(provider, 0)

        eff_score = 0

        # Penalize expensive/large models
        if any(term in model_id for term in ("pro", "large", "405b", "ultra", "opus")):
            eff_score -= 5

        # Reward known small/fast models
        if any(term in model_id for term in ("flash", "8b", "small", "instant", "haiku", "mini")):
            eff_score += 5

        return (prov_score, eff_score)

    async def assign(self, agents: list[AgentConfig]) -> AssignmentSet:
        if not agents:
            return AssignmentSet(
                by_agent={},
                catalog_errors={},
                used_unique_models=True,
                generated_at=datetime.now(timezone.utc),
            )

        results = await self._model_catalog.fetch_models_all()
        catalog_errors = {
            result.provider: result.error
            for result in results
            if result.error
        }

        used_models: set[str] = set()
        by_agent: dict[str, AssignedModel] = {}
        active_names = [agent.name for agent in agents]

        # Phase 1: Assign from curated pools (per-provider, randomized, uniqueness-aware)
        for agent_name in active_names:
            agent_config = next(a for a in agents if a.name == agent_name)
            provider = agent_config.provider.lower()

            # Get available curated models for this provider
            provider_curated = [
                (p, m) for p, m in self._build_curated_pool(used_models)
                if p == provider
            ]

            if provider_curated:
                self._rng.shuffle(provider_curated)
                provider_curated.sort(key=self._score_pool_item, reverse=True)
                chosen_provider, chosen_model = provider_curated[0]
                by_agent[agent_name] = AssignedModel(
                    provider=chosen_provider,
                    model=chosen_model,
                    source="curated",
                )
                used_models.add(chosen_model)

        # Phase 2: Fill remaining agents from live catalog
        missing_agents = [a for a in agents if a.name not in by_agent]
        if missing_agents:
            catalog_pool, skipped_ineligible = self._build_catalog_pool(results, used_models)

            if skipped_ineligible >= 3:
                logger.info("Skipped %s ineligible catalog models during assignment", skipped_ineligible)

            self._rng.shuffle(catalog_pool)
            catalog_pool.sort(key=self._score_pool_item, reverse=True)

            for agent_name in [a.name for a in missing_agents]:
                if not catalog_pool:
                    break
                provider, model = catalog_pool.pop()
                by_agent[agent_name] = AssignedModel(
                    provider=provider,
                    model=model,
                    source="catalog",
                )
                used_models.add(model)

        # Phase 3: Fallback to configured defaults for any remaining agents
        missing_agents = [a for a in agents if a.name not in by_agent]
        fallback_unique = True

        # First, try to keep unique models with defaults that are not yet used.
        unused_defaults = [
            agent
            for agent in missing_agents
            if self._normalize_model(agent.provider, agent.model) not in used_models
        ]
        for agent in unused_defaults:
            normalized_model = self._normalize_model(agent.provider, agent.model)
            by_agent[agent.name] = AssignedModel(
                provider=agent.provider,
                model=normalized_model,
                source="fallback",
            )
            used_models.add(normalized_model)

        # If uniqueness still cannot be satisfied, degrade gracefully using defaults.
        for agent in missing_agents:
            if agent.name in by_agent:
                continue
            fallback_unique = False
            normalized_model = self._normalize_model(agent.provider, agent.model)
            by_agent[agent.name] = AssignedModel(
                provider=agent.provider,
                model=normalized_model,
                source="fallback",
            )

        # Log assignment summary
        for agent_name, assignment in by_agent.items():
            logger.info(
                "Assigned %s -> %s (source=%s)",
                agent_name,
                assignment.model,
                assignment.source,
            )

        return AssignmentSet(
            by_agent=by_agent,
            catalog_errors=catalog_errors,
            used_unique_models=fallback_unique,
            generated_at=datetime.now(timezone.utc),
        )
