"""
All database queries in one place. Small enough that splitting it across
several files just added indirection.
"""

from datetime import datetime, timedelta
from sqlalchemy import select, desc, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Model, BenchmarkRun, Score, LeaderboardSnapshot


# ── Models ──────────────────────────────────────────────────────────────────

async def upsert_model(db: AsyncSession, data: dict) -> Model:
    """Insert the model if new, else refresh last_seen. Returns the row."""
    existing = (await db.execute(
        select(Model).where(Model.model_id == data["model_id"])
    )).scalar_one_or_none()

    if existing:
        existing.name = data["name"]
        existing.provider = data["provider"]
        existing.is_active = True
        existing.last_seen = datetime.utcnow()
        await db.commit()
        return existing

    model = Model(**data, is_active=True)
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


async def get_models_by_provider(db: AsyncSession, provider: str) -> list[Model]:
    rows = await db.execute(
        select(Model).where(Model.provider == provider, Model.is_active == True)  # noqa: E712
    )
    return list(rows.scalars().all())


async def get_all_active_models(db: AsyncSession) -> list[Model]:
    rows = await db.execute(select(Model).where(Model.is_active == True))  # noqa: E712
    return list(rows.scalars().all())


# ── Runs ────────────────────────────────────────────────────────────────────

async def create_run(db: AsyncSession, provider: str, total_models: int) -> BenchmarkRun:
    run = BenchmarkRun(provider=provider, status="running", total_models=total_models)
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def get_run(db: AsyncSession, run_id: int) -> BenchmarkRun | None:
    return (await db.execute(
        select(BenchmarkRun).where(BenchmarkRun.id == run_id)
    )).scalar_one_or_none()


async def get_recent_active_run(db: AsyncSession, provider: str,
                                within_seconds: int) -> BenchmarkRun | None:
    """A still-'running' run for this provider started within `within_seconds`.
    The time window means a run orphaned by a killed worker stops blocking new
    runs after it goes stale, instead of wedging the provider forever."""
    cutoff = datetime.utcnow() - timedelta(seconds=within_seconds)
    return (await db.execute(
        select(BenchmarkRun)
        .where(BenchmarkRun.provider == provider,
               BenchmarkRun.status == "running",
               BenchmarkRun.started_at >= cutoff)
        .order_by(desc(BenchmarkRun.started_at))
    )).scalars().first()


async def fail_stale_running_runs(db: AsyncSession, older_than_seconds: int) -> int:
    """Mark 'running' runs older than the cutoff as failed. Cleans up runs left
    dangling by a killed/crashed worker. Returns how many were updated."""
    cutoff = datetime.utcnow() - timedelta(seconds=older_than_seconds)
    result = await db.execute(
        update(BenchmarkRun)
        .where(BenchmarkRun.status == "running", BenchmarkRun.started_at < cutoff)
        .values(status="failed", completed_at=datetime.utcnow(),
                error_message="Marked failed: worker stopped before completion.")
    )
    await db.commit()
    return result.rowcount or 0


async def increment_completed(db: AsyncSession, run_id: int) -> None:
    """Atomic +1 so parallel model tasks don't clobber each other."""
    await db.execute(
        update(BenchmarkRun)
        .where(BenchmarkRun.id == run_id)
        .values(completed_models=BenchmarkRun.completed_models + 1)
    )
    await db.commit()


async def complete_run(db: AsyncSession, run_id: int) -> None:
    run = await get_run(db, run_id)
    if run:
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        run.completed_models = run.total_models
        await db.commit()


async def fail_run(db: AsyncSession, run_id: int, error: str) -> None:
    run = await get_run(db, run_id)
    if run:
        run.status = "failed"
        run.completed_at = datetime.utcnow()
        run.error_message = error[:500]
        await db.commit()


# ── Scores & snapshots ──────────────────────────────────────────────────────

async def save_scores(db: AsyncSession, run_id: int, model_db_id: int,
                      per_category: dict[str, dict]) -> None:
    for category, res in per_category.items():
        db.add(Score(
            run_id=run_id,
            model_id=model_db_id,
            category=category,
            score=res.get("score", 0.0),
            sample_count=res.get("sample_count", 0),
        ))
    await db.commit()


async def save_snapshot(db: AsyncSession, run_id: int, model_db_id: int,
                        provider: str, composite: float, scores_json: dict) -> None:
    db.add(LeaderboardSnapshot(
        run_id=run_id,
        model_id=model_db_id,
        provider=provider,
        composite_score=composite,
        scores_json=scores_json,
    ))
    await db.commit()


async def get_latest_snapshot_per_model(db: AsyncSession) -> list[tuple]:
    """
    For the combined leaderboard: the most recent snapshot for EACH model,
    across all providers/runs. Returns list of (snapshot, model) sorted by
    composite score descending.
    """
    rows = (await db.execute(
        select(LeaderboardSnapshot, Model)
        .join(Model, LeaderboardSnapshot.model_id == Model.id)
        .order_by(desc(LeaderboardSnapshot.created_at))
    )).all()

    seen: set[int] = set()
    latest = []
    for snap, model in rows:
        if model.id in seen:
            continue
        seen.add(model.id)
        latest.append((snap, model))

    latest.sort(key=lambda x: x[0].composite_score, reverse=True)
    return latest
