"""
ONE task: benchmark every main model of a single provider, sequentially, in one
self-contained run. No chord, no second queue, no cross-task coordination.

Flow:
  1. discover the provider's main models  → save them to DB
  2. create a run row (status=running)
  3. for each model: run 4 benchmarks, save scores + a leaderboard snapshot
  4. mark run completed  (or failed, once, on a fatal error)

Because it's one task, a run can only ever be running → completed or
running → failed. It can never get stuck half-done the way the old chord could.
"""

import asyncio
from app.celery.celery_app import celery_app
from app.utils.settings import settings as _s
from app.utils.logger import get_logger

logger = get_logger("tasks")


async def _run_provider_async(run_id: int, provider: str) -> None:
    from app.config.database import AsyncLocalSession
    from app.benchmarks.orchestrator import run_model
    from app.repositories import repository as repo
    from app.utils.provider_clients import get_models_for_provider, close_client, BudgetExhausted
    from app.utils.cache import invalidate_leaderboard_cache, close_redis
    from app.utils.rate_limits import limiter

    limiter.reset_provider(provider)

    try:
        async with AsyncLocalSession() as db:
            models = await get_models_for_provider(provider)

            if not models:
                await repo.fail_run(db, run_id, f"No usable models found for '{provider}'")
                return

            db_models = [await repo.upsert_model(db, m) for m in models]

            run = await repo.get_run(db, run_id)
            if run:
                run.total_models = len(db_models)
                await db.commit()

            logger.info(f"[run={run_id}] {provider}: benchmarking {len(db_models)} models. "
                        f"{limiter.summary(provider)}")

            done, skipped = 0, 0

            for row in db_models:
                # Stop cleanly if the provider's request budget is spent. Results
                # already saved stay saved — a partial leaderboard beats none.
                if limiter.is_exhausted(provider):
                    logger.warning(f"[run={run_id}] budget spent — "
                                   f"skipping remaining {len(db_models) - done - skipped} models")
                    break

                logger.info(f"[run={run_id}] ({done + 1}/{len(db_models)}) {row.model_id}")

                try:
                    result = await run_model(provider, row.model_id)

                    # Save immediately after each model, not at the end. If the
                    # run dies later, everything finished so far is preserved.
                    await repo.save_scores(db, run_id, row.id, result["per_category"])
                    await repo.save_snapshot(
                        db, run_id, row.id, provider, result["composite"],
                        {c: v["score"] for c, v in result["per_category"].items()},
                    )
                    await invalidate_leaderboard_cache()
                    done += 1
                    logger.info(f"[run={run_id}] {row.model_id} -> {result['composite']}")

                except BudgetExhausted:
                    logger.warning(f"[run={run_id}] budget hit during {row.model_id}")
                    break
                except Exception as e:
                    skipped += 1
                    logger.error(f"[run={run_id}] {row.model_id} failed: {e}")

                await repo.increment_completed(db, run_id)

            await repo.complete_run(db, run_id)
            await invalidate_leaderboard_cache()
            logger.info(f"[run={run_id}] {provider} DONE — {done} scored, {skipped} failed. "
                        f"{limiter.summary(provider)}")

    except Exception as e:
        logger.error(f"[run={run_id}] FATAL: {e}")
        async with AsyncLocalSession() as db:
            await repo.fail_run(db, run_id, str(e))
    finally:
        await close_client()
        await close_redis()
        await AsyncLocalSession.dispose_current()


@celery_app.task(name="run_provider_benchmark",
                 time_limit=_s.RUN_TIME_LIMIT,
                 soft_time_limit=_s.RUN_TIME_LIMIT - 60)
def run_provider_benchmark(run_id: int, provider: str):
    """Sync Celery entry point — bridges into the async run above."""
    logger.info(f"[run={run_id}] START provider={provider}")
    asyncio.run(_run_provider_async(run_id, provider))
