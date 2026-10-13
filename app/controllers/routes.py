"""
The ENTIRE API surface. Nothing else. Exactly what was asked for:

  GET  /health                  — is the service (and Redis) alive
  GET  /leaderboard             — combined results across all 4 providers
  POST /sync/models             — discover main models from all providers
  GET  /models                  — list all known models

  POST /benchmark/groq          — run benchmarks on Groq's main models
  POST /benchmark/nvidia        — run benchmarks on NVIDIA's main models
  POST /benchmark/gemini        — run benchmarks on Gemini's main models
  POST /benchmark/mistral       — run benchmarks on Mistral's main models

  GET  /runs/{run_id}           — progress of a specific run (for polling)

No compare, history, categories, run-list, or per-model-detail endpoints — those
were removed as requested.
"""

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.database import get_db
from app.utils.cache import get_redis
from app.services import service

router = APIRouter()


# ── Health ──────────────────────────────────────────────────────────────────

@router.get("/health", tags=["Health"])
async def health():
    redis_ok = "ok"
    try:
        await get_redis().ping()
    except Exception as e:
        redis_ok = f"error: {e}"
    return {"status": "ok", "service": "LLM Leaderboard", "redis": redis_ok}


# ── Leaderboard ─────────────────────────────────────────────────────────────

@router.get("/leaderboard", tags=["Leaderboard"])
async def leaderboard(db: AsyncSession = Depends(get_db)):
    """Combined leaderboard — best result per model across all providers."""
    return await service.get_leaderboard(db)


# ── Models ──────────────────────────────────────────────────────────────────

@router.post("/sync/models", tags=["Models"])
async def sync_models(db: AsyncSession = Depends(get_db)):
    """Discover and store the main chat models from all 4 providers."""
    return await service.sync_all_models(db)


@router.get("/models", tags=["Models"])
async def all_models(db: AsyncSession = Depends(get_db)):
    """List every known model."""
    return await service.list_all_models(db)


# ── One benchmark trigger per provider ──────────────────────────────────────

@router.post("/benchmark/groq", tags=["Benchmark"])
async def benchmark_groq():
    """Run all benchmarks on Groq's main models."""
    return await service.start_provider_run("groq")


@router.post("/benchmark/nvidia", tags=["Benchmark"])
async def benchmark_nvidia():
    """Run all benchmarks on NVIDIA's main models."""
    return await service.start_provider_run("nvidia")


@router.post("/benchmark/gemini", tags=["Benchmark"])
async def benchmark_gemini():
    """Run all benchmarks on Gemini's main models."""
    return await service.start_provider_run("gemini")


@router.post("/benchmark/mistral", tags=["Benchmark"])
async def benchmark_mistral():
    """Run all benchmarks on Mistral's main models."""
    return await service.start_provider_run("mistral")


@router.post("/benchmark/all", tags=["Benchmark"])
async def benchmark_all():
    """Run every provider, sequentially. One click, full leaderboard."""
    return await service.start_all_providers()


# ── Run progress (for the frontend to poll) ─────────────────────────────────

@router.get("/runs/{run_id}", tags=["Benchmark"])
async def run_status(run_id: int = Path(...), db: AsyncSession = Depends(get_db)):
    status = await service.get_run_status(db, run_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return status
