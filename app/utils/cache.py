"""
Tiny Redis cache wrapper. Only used to cache the leaderboard response so
repeated frontend calls are fast. Everything degrades gracefully — if Redis
is down, cache calls just return None and the app keeps working.
"""

import json
import asyncio
from typing import Any, Optional
from redis import asyncio as aioredis

from app.utils.settings import settings
from app.utils.logger import get_logger

logger = get_logger("cache")

# One Redis client per event loop (same reasoning as the HTTP client).
_clients: dict[int, aioredis.Redis] = {}


def get_redis() -> aioredis.Redis:
    key = id(asyncio.get_event_loop())
    if key not in _clients:
        _clients[key] = aioredis.from_url(
            settings.REDIS_URL, decode_responses=True,
            socket_connect_timeout=5, socket_timeout=5,
        )
    return _clients[key]


async def close_redis():
    key = id(asyncio.get_event_loop())
    client = _clients.pop(key, None)
    if client is not None:
        await client.aclose()


async def cache_get(key: str) -> Optional[Any]:
    try:
        val = await get_redis().get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


async def cache_set(key: str, value: Any, ttl: int = 120) -> None:
    try:
        await get_redis().setex(key, ttl, json.dumps(value, default=str))
    except Exception as e:
        logger.warning(f"cache set failed key={key}: {e}")


async def invalidate_leaderboard_cache() -> None:
    try:
        client = get_redis()
        keys = [k async for k in client.scan_iter(match="leaderboard:*")]
        if keys:
            await client.delete(*keys)
    except Exception as e:
        logger.warning(f"cache invalidate failed: {e}")
