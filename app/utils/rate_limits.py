"""
RATE LIMITING & BUDGET CONTROL

This is what stands between a working benchmark and a leaderboard full of
zeros. Free tiers are far tighter than they look:

    mistral  magistral-small   0.03 rps  = one request every 33 SECONDS
    mistral  mistral-large     0.07 rps  = one request every 14 seconds
    groq     (all models)      30 RPM    = one every 2 seconds
    nvidia   (all models)      ~40 RPM   = one every 1.5 seconds
    gemini   flash             ~15 RPM   = one every 4 seconds

Firing requests back-to-back against these limits produces 429s, which score
0 and look identical to "the model is bad". Three mechanisms here:

  1. PACING       - wait the required gap before every call, per model.
  2. ADAPTIVE     - on a 429, permanently slow that model down for the rest of
                    the session (and honour any retry-after header).
  3. BUDGET       - hard cap on total requests per provider, so a run can never
                    silently burn a day's quota or a credit balance.

Limits live in rate_limits.json so they can be corrected without code changes.
"""

import json
import time
import asyncio
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger("ratelimit")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "rate_limits.json"

_FALLBACK = {
    "groq": {"default_rps": 0.45, "daily_request_budget": 900, "models": {}},
    "nvidia": {"default_rps": 0.55, "daily_request_budget": 800, "models": {}},
    "gemini": {"default_rps": 0.15, "daily_request_budget": 1200, "models": {}},
    "mistral": {"default_rps": 0.3, "daily_request_budget": 3000, "models": {}},
}


def _load_config() -> dict:
    try:
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        cfg.pop("_comment", None)
        logger.info(f"rate limits loaded from {_CONFIG_PATH.name}")
        return cfg
    except Exception as e:
        logger.warning(f"could not read rate_limits.json ({e}); using built-in defaults")
        return _FALLBACK


class RateLimiter:
    """One shared instance per process. Safe across asyncio tasks."""

    def __init__(self):
        self.cfg = _load_config()
        self._last_call: dict[str, float] = {}     # key -> monotonic timestamp
        self._locks: dict[str, asyncio.Lock] = {}
        self._penalty: dict[str, float] = {}       # key -> extra seconds after 429
        self._used: dict[str, int] = {}            # provider -> requests made
        self._blocked: set[str] = set()            # providers over budget

    # ── config lookup ───────────────────────────────────────────────────

    def _provider_cfg(self, provider: str) -> dict:
        return self.cfg.get(provider, _FALLBACK.get(provider, {"default_rps": 0.3,
                                                               "daily_request_budget": 500,
                                                               "models": {}}))

    def rps_for(self, provider: str, model: str) -> float:
        """
        Requests-per-second allowed for this exact model.

        Model overrides are matched by PREFIX, so a single entry for
        "mistral-large" covers mistral-large-latest, mistral-large-2512, etc.
        The longest matching prefix wins, so "mistral-small-2603" beats the
        more general "mistral-small".
        """
        pcfg = self._provider_cfg(provider)
        models = pcfg.get("models", {})

        name = model.split("/")[-1].lower()
        best_key, best_len = None, -1
        for key in models:
            if name.startswith(key.lower()) and len(key) > best_len:
                best_key, best_len = key, len(key)

        return float(models[best_key]) if best_key else float(pcfg.get("default_rps", 0.3))

    def interval_for(self, provider: str, model: str) -> float:
        rps = max(self.rps_for(provider, model), 0.001)
        key = f"{provider}:{model}"
        return (1.0 / rps) + self._penalty.get(key, 0.0)

    # ── budget ──────────────────────────────────────────────────────────

    def budget_for(self, provider: str) -> int:
        return int(self._provider_cfg(provider).get("daily_request_budget", 500))

    def used(self, provider: str) -> int:
        return self._used.get(provider, 0)

    def remaining(self, provider: str) -> int:
        return max(self.budget_for(provider) - self.used(provider), 0)

    def is_exhausted(self, provider: str) -> bool:
        return provider in self._blocked or self.remaining(provider) <= 0

    def reset_provider(self, provider: str) -> None:
        """Called at the start of a provider run."""
        self._used[provider] = 0
        self._blocked.discard(provider)

    # ── the main gate ───────────────────────────────────────────────────

    async def acquire(self, provider: str, model: str) -> bool:
        """
        Wait until it's safe to call this model. Returns False if the
        provider's request budget is spent (caller should stop).
        """
        if self.is_exhausted(provider):
            if provider not in self._blocked:
                self._blocked.add(provider)
                logger.warning(f"{provider}: request budget exhausted "
                               f"({self.used(provider)}/{self.budget_for(provider)})")
            return False

        key = f"{provider}:{model}"
        lock = self._locks.setdefault(key, asyncio.Lock())

        async with lock:
            gap = self.interval_for(provider, model)
            last = self._last_call.get(key)
            if last is not None:
                wait = gap - (time.monotonic() - last)
                if wait > 0:
                    if wait > 5:
                        logger.info(f"pacing {provider}/{model}: waiting {wait:.1f}s "
                                    f"(limit {self.rps_for(provider, model)} rps)")
                    await asyncio.sleep(wait)
            self._last_call[key] = time.monotonic()

        self._used[provider] = self.used(provider) + 1
        return True

    def penalise(self, provider: str, model: str, retry_after: float | None = None) -> None:
        """
        Called after a 429. Permanently widens this model's gap for the rest of
        the session — if the published limit was optimistic, stop trusting it.
        """
        key = f"{provider}:{model}"
        current = self._penalty.get(key, 0.0)
        bump = max(retry_after or 0.0, current + 2.0, 2.0)
        self._penalty[key] = min(bump, 60.0)
        logger.warning(f"429 from {provider}/{model} — slowing to "
                       f"+{self._penalty[key]:.0f}s extra between calls")

    def summary(self, provider: str) -> str:
        return (f"{provider}: {self.used(provider)}/{self.budget_for(provider)} "
                f"requests used")


# Shared instance
limiter = RateLimiter()
