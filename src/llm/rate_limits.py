from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field


class WindowRateLimiter:
    def __init__(self, max_requests_per_minute: int) -> None:
        self.max_requests = max(1, max_requests_per_minute)
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                wait_seconds = 60.0 - (now - self._timestamps[0])

            await asyncio.sleep(max(0.05, wait_seconds))


@dataclass
class ProviderPolicy:
    name: str
    rpm_budget: int
    limiter: WindowRateLimiter
    cooldown_until: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RateLimitManager:
    def __init__(
        self,
        groq_rpm: int,
        mistral_rpm: int,
        cerebras_rpm: int,
        sambanova_rpm: int,
        google_rpm: int,
    ) -> None:
        self._providers: dict[str, ProviderPolicy] = {
            "groq": ProviderPolicy("groq", groq_rpm, WindowRateLimiter(groq_rpm)),
            "mistral": ProviderPolicy("mistral", mistral_rpm, WindowRateLimiter(mistral_rpm)),
            "cerebras": ProviderPolicy("cerebras", cerebras_rpm, WindowRateLimiter(cerebras_rpm)),
            "sambanova": ProviderPolicy("sambanova", sambanova_rpm, WindowRateLimiter(sambanova_rpm)),
            "google": ProviderPolicy("google", google_rpm, WindowRateLimiter(google_rpm)),
        }

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.lower().strip()
        if key in {"gemini", "google", "google_ai_studio", "google-ai-studio", "googleai"}:
            return "google"
        return key

    def _get(self, provider: str) -> ProviderPolicy:
        key = self._normalize_provider(provider)
        if key not in self._providers:
            # Unknown providers default to conservative pacing.
            self._providers[key] = ProviderPolicy(key, 10, WindowRateLimiter(10))
        return self._providers[key]

    async def acquire(self, provider: str) -> None:
        policy = self._get(provider)
        while True:
            async with policy.lock:
                now = time.monotonic()
                if now < policy.cooldown_until:
                    sleep_for = policy.cooldown_until - now
                else:
                    sleep_for = 0.0

            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
                continue

            await policy.limiter.acquire()
            return

    async def apply_retry_after(self, provider: str, retry_after_seconds: float) -> None:
        policy = self._get(provider)
        async with policy.lock:
            cooldown = max(0.0, retry_after_seconds)
            policy.cooldown_until = max(policy.cooldown_until, time.monotonic() + cooldown)
