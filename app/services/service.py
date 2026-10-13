"""
Business logic between controllers and repositories.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.database import AsyncLocalSession
from app.repositories import repository as repo
from app.utils.provider_clients import get_models_for_provider, get_all_models
from app.utils.cache import cache_get, cache_set
from app.utils.settings import settings, PROVIDERS
from app.utils.logger import get_logger

logger = get_logger("service")


# ── Trigger a provider benchmark ────────────────────────────────────────────

async def start_provider_run(provider: str) -> dict:
    """Create a run row and dispatch the single Celery task for this provider."""

    if provider not in PROVIDERS:
        return {"error": f"Unknown provider '{provider}'. Valid: {', '.join(PROVIDERS)}"}

    # local import so the web app doesn't need Celery imported at module load
    from app.celery.tasks import run_provider_benchmark

    async with AsyncLocalSession() as db:
        # Guard against duplicate concurrent runs. Clicking "Execute" twice used
        # to launch two runs for the same provider that then fought over the
        # (tiny) free-tier rate limit and deadlocked each other. Refuse a second
        # run while one is still active.
        active = await repo.get_recent_active_run(
            db, provider, settings.DUPLICATE_RUN_WINDOW_SECONDS)
        if active:
            return {
                "message": f"A {provider} benchmark is already running "
                           f"(run_id={active.id}). Wait for it to finish before "
                           f"starting another.",
                "provider": provider,
                "run_id": active.id,
                "status": "running",
                "already_running": True,
                "tip": f"Check progress: GET /runs/{active.id}",
            }

        # total_models is set for real inside the task once models are discovered
        run = await repo.create_run(db, provider=provider, total_models=0)
        run_id = run.id

    run_provider_benchmark.delay(run_id=run_id, provider=provider)

    logger.info(f"dispatched run_id={run_id} for provider={provider}")
    return {
        "message": f"Benchmark started for {provider}",
        "provider": provider,
        "run_id": run_id,
        "status": "running",
        "tip": f"Check progress: GET /runs/{run_id}",
    }


async def get_run_status(db: AsyncSession, run_id: int) -> dict | None:
    run = await repo.get_run(db, run_id)
    if not run:
        return None
    pct = round((run.completed_models / run.total_models) * 100, 1) if run.total_models else 0.0
    return {
        "run_id": run.id,
        "provider": run.provider,
        "status": run.status,
        "total_models": run.total_models,
        "completed_models": run.completed_models,
        "progress_percent": pct,
        "started_at": str(run.started_at),
        "completed_at": str(run.completed_at) if run.completed_at else None,
        "error_message": run.error_message,
    }


# ── Leaderboard (combined, all providers) ───────────────────────────────────

async def get_leaderboard(db: AsyncSession) -> dict:
    cached = await cache_get("leaderboard:all")
    if cached:
        return cached

    rows = await repo.get_latest_snapshot_per_model(db)

    entries = [
        {
            "rank": i,
            "model_id": model.model_id,
            "model_name": model.name,
            "provider": model.provider,
            "composite_score": snap.composite_score,
            "scores": snap.scores_json or {},
        }
        for i, (snap, model) in enumerate(rows, start=1)
    ]

    top_n = settings.LEADERBOARD_TOP_N
    shown = entries[:top_n] if top_n else entries
    result = {"total": len(entries), "showing": len(shown), "entries": shown}
    await cache_set("leaderboard:all", result, ttl=settings.LEADERBOARD_CACHE_TTL)
    return result


# ── Models ──────────────────────────────────────────────────────────────────

async def sync_all_models(db: AsyncSession) -> dict:
    """Discover main models from all 4 providers and store them."""
    all_models = await get_all_models()
    for m in all_models:
        await repo.upsert_model(db, m)
    return {"message": "Model sync complete", "total_models": len(all_models)}


async def list_all_models(db: AsyncSession) -> dict:
    models = await repo.get_all_active_models(db)
    return {
        "total": len(models),
        "models": [
            {"model_id": m.model_id, "name": m.name, "provider": m.provider}
            for m in models
        ],
    }


async def start_all_providers() -> dict:
    """
    Queue every provider, one after another. Each is a separate Celery task, so
    they run sequentially on a single worker — which is what you want, because
    running them in parallel would multiply the rate-limit pressure.
    """
    from app.celery.tasks import run_provider_benchmark

    runs = []
    async with AsyncLocalSession() as db:
        for provider in PROVIDERS:
            run = await repo.create_run(db, provider=provider, total_models=0)
            runs.append({"provider": provider, "run_id": run.id})

    for r in runs:
        run_provider_benchmark.delay(run_id=r["run_id"], provider=r["provider"])

    return {
        "message": "Queued all 4 providers (they run one at a time)",
        "runs": runs,
        "tip": "Poll GET /runs/{run_id}, or just watch GET /leaderboard fill up.",
    }
