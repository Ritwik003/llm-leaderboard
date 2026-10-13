"""
Runs all 4 benchmark categories for ONE model and computes its composite score.
"""

from app.benchmarks.knowledge import runner as knowledge
from app.benchmarks.reasoning import runner as reasoning
from app.benchmarks.math import runner as math_r
from app.benchmarks.coding import runner as coding
from app.utils.settings import settings
from app.utils.logger import get_logger

logger = get_logger("orchestrator")

RUNNERS = {
    "knowledge": knowledge,
    "reasoning": reasoning,
    "math": math_r,
    "coding": coding,
}

WEIGHTS = {
    "knowledge": settings.WEIGHT_KNOWLEDGE,
    "reasoning": settings.WEIGHT_REASONING,
    "math": settings.WEIGHT_MATH,
    "coding": settings.WEIGHT_CODING,
}


async def run_model(provider: str, model: str) -> dict:
    """Returns {per_category: {cat: {score, sample_count}}, composite: float}."""

    per_category = {}

    for category, runner in RUNNERS.items():
        try:
            logger.info(f"[{provider}] {model} → {category}")
            per_category[category] = await runner.run(provider=provider, model=model)
        except Exception as e:
            logger.error(f"[{provider}] {model} {category} failed: {e}")
            per_category[category] = {"score": 0.0, "sample_count": 0}

    composite = _composite(per_category)
    logger.info(f"[{provider}] {model} composite={composite}")

    return {"per_category": per_category, "composite": composite}


def _composite(per_category: dict) -> float:
    weighted = sum(per_category[c]["score"] * w for c, w in WEIGHTS.items() if c in per_category)
    total_w = sum(w for c, w in WEIGHTS.items() if c in per_category)
    return round(weighted / total_w, 2) if total_w else 0.0
