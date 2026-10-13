"""
Async SQLAlchemy setup.

The _LoopAwareSessionFactory looks fancy but exists for ONE reason: Celery on
Windows runs --pool=threads and calls asyncio.run() per task, making a fresh
event loop each time. asyncpg pools bind to the loop they were created on, so a
single global engine would crash every task after the first. We therefore cache
one engine per event loop. FastAPI (one persistent loop) only ever makes one —
this is transparent to it.
"""

import asyncio
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine,
)
from sqlalchemy.orm import declarative_base

from app.utils.settings import settings

Base = declarative_base()


def _to_async_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


class _LoopAwareSessionFactory:
    def __init__(self):
        self._engines: dict[int, AsyncEngine] = {}
        self._makers: dict[int, async_sessionmaker] = {}

    def _maker(self) -> async_sessionmaker:
        key = id(asyncio.get_event_loop())
        if key not in self._makers:
            engine = create_async_engine(
                _to_async_url(settings.DB_Configuration),
                pool_pre_ping=True, pool_size=5, max_overflow=10,
            )
            self._engines[key] = engine
            self._makers[key] = async_sessionmaker(
                bind=engine, class_=AsyncSession, expire_on_commit=False,
            )
        return self._makers[key]

    def __call__(self) -> AsyncSession:
        return self._maker()()

    def get_engine(self) -> AsyncEngine:
        self._maker()
        return self._engines[id(asyncio.get_event_loop())]

    async def dispose_current(self):
        key = id(asyncio.get_event_loop())
        engine = self._engines.pop(key, None)
        self._makers.pop(key, None)
        if engine is not None:
            await engine.dispose()


AsyncLocalSession = _LoopAwareSessionFactory()


async def get_db():
    async with AsyncLocalSession() as session:
        yield session


async def create_all_tables():
    engine = AsyncLocalSession.get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
