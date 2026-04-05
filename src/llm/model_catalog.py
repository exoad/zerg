from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from src.config.settings import Settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderModelsResult:
    provider: str
    models: list[str]
    error: str | None = None


class ModelCatalog:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch_models(self, provider: str) -> ProviderModelsResult:
        normalized = self._normalize_provider(provider)
        if normalized == "google":
            return await self._fetch_google_models()

        config = self._provider_config(normalized)
        if config is None:
            return ProviderModelsResult(provider=normalized, models=[], error="unsupported provider")

        base_url, api_key = config
        if not api_key:
            return ProviderModelsResult(provider=normalized, models=[], error="missing API key")

        endpoint = base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            session = await self._get_session()
            async with session.get(endpoint, headers=headers, timeout=20) as response:
                if response.status >= 400:
                    text = await response.text()
                    short = text[:200].replace("\n", " ")
                    return ProviderModelsResult(
                        provider=normalized,
                        models=[],
                        error=f"HTTP {response.status}: {short}",
                    )

                payload: dict[str, Any] = await response.json()
        except Exception as exc:
            return ProviderModelsResult(provider=normalized, models=[], error=str(exc))

        data = payload.get("data", [])
        model_ids: list[str] = []
        filtered_count = 0
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and "id" in entry:
                    candidate = str(entry["id"])
                    if self._is_non_chat_model(candidate):
                        filtered_count += 1
                        continue
                    model_ids.append(self._qualify_model_id(normalized, candidate))

        if filtered_count >= 3:
            logger.info("Filtered %s non-chat models from %s catalog response", filtered_count, normalized)

        model_ids = sorted(set(model_ids))
        if not model_ids:
            return ProviderModelsResult(provider=normalized, models=[], error="no models returned")

        return ProviderModelsResult(provider=normalized, models=model_ids)

    async def fetch_models_all(self) -> list[ProviderModelsResult]:
        providers = ["groq", "mistral", "cerebras", "sambanova", "google"]
        results: list[ProviderModelsResult] = []
        for provider in providers:
            results.append(await self.fetch_models(provider))
        return results

    async def _fetch_google_models(self) -> ProviderModelsResult:
        api_key = self.settings.google_api_key
        if not api_key:
            return ProviderModelsResult(provider="google", models=[], error="missing API key")

        endpoint = self.settings.google_models_api_base.rstrip("/") + f"/models?key={api_key}"

        try:
            session = await self._get_session()
            async with session.get(endpoint, timeout=20) as response:
                if response.status >= 400:
                    text = await response.text()
                    short = text[:200].replace("\n", " ")
                    return ProviderModelsResult(
                        provider="google",
                        models=[],
                        error=f"HTTP {response.status}: {short}",
                    )
                payload: dict[str, Any] = await response.json()
        except Exception as exc:
            return ProviderModelsResult(provider="google", models=[], error=str(exc))

        raw_models = payload.get("models", [])
        model_ids: list[str] = []
        filtered_count = 0
        if isinstance(raw_models, list):
            for entry in raw_models:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                if name.startswith("models/"):
                    name = name.split("models/", 1)[1]
                if self._is_non_chat_model(name):
                    filtered_count += 1
                    continue
                model_ids.append(f"gemini/{name}")

        if filtered_count >= 3:
            logger.info("Filtered %s non-chat models from google catalog response", filtered_count)

        model_ids = sorted(set(model_ids))
        if not model_ids:
            return ProviderModelsResult(provider="google", models=[], error="no models returned")

        return ProviderModelsResult(provider="google", models=model_ids)

    @staticmethod
    def _qualify_model_id(provider: str, model_id: str) -> str:
        cleaned = model_id.strip()
        if not cleaned:
            return cleaned
        if "/" in cleaned:
            return cleaned
        return f"{provider}/{cleaned}"

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.lower().strip()
        if key in {"gemini", "google", "google_ai_studio", "google-ai-studio", "googleai"}:
            return "google"
        return key

    def _provider_config(self, provider: str) -> tuple[str, str] | None:
        if provider == "groq":
            return (self.settings.groq_api_base, self.settings.groq_api_key)
        if provider == "mistral":
            return (self.settings.mistral_api_base, self.settings.mistral_api_key)
        if provider == "cerebras":
            return (self.settings.cerebras_api_base, self.settings.cerebras_api_key)
        if provider == "sambanova":
            return (self.settings.sambanova_api_base, self.settings.sambanova_api_key)
        return None

    @staticmethod
    def _is_non_chat_model(model_id: str) -> bool:
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
        return any(fragment in lowered for fragment in blocked_fragments)