from celery import Celery
from app.utils.settings import settings

# ONE Celery app, ONE queue. The old design had two queues (benchmark +
# assembly) coordinated by a chord, which is what could leave a run stuck at
# "running" forever if a task died. Now each provider run is a single
# self-contained task — far simpler and impossible to half-finish.

celery_app = Celery(
    "llm_leaderboard",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.celery.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,

    # Behaviour
    task_acks_late=False,               # ack on receive — a crashed task will
                                        # NOT be redelivered and re-run forever.
                                        # (The old acks_late=True + retries is
                                        # part of what made tasks "never stop".)
    task_track_started=True,

    # NO automatic task-level retries. If a provider run fails, it fails and is
    # marked failed in the DB. You re-click to retry. This is deliberate — auto
    # retries are what filled the queue.
    task_max_retries=0,

    result_expires=3600,                # 1 hour
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=20,      # recycle workers to release memory

    # Everything on the default queue. No routing needed.
    task_default_queue="benchmark",

    beat_schedule={},                   # no cron, manual trigger only
)
