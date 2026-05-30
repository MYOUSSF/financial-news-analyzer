"""
CacheClient — Redis-backed analysis cache with transparent in-memory fallback.

When Redis is unavailable (REDIS_URL not set, connection refused, redis package
absent) the client automatically falls back to a TTL-aware in-memory dict, so
the application works without any code change and without Redis running.
"""
import json
import os
import time
from typing import Any, Dict, Optional

from loguru import logger


class CacheClient:
    """
    Wraps Redis with a transparent in-memory fallback.

    The Redis path serialises values as JSON; the memory path stores Python
    dicts directly.  Both expose the same get/set interface so callers never
    need to know which backend is active.

    Usage:
        cache = CacheClient()                          # auto-detects REDIS_URL
        cache = CacheClient(redis_url="redis://...")   # explicit URL

        key = CacheClient.make_cache_key("AAPL", 7, "2024-01-15-14")
        cache.set(key, result, ttl_seconds=14400)      # 4 hours
        value = cache.get(key)                         # None on miss / expiry
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Connect to Redis if a URL is available; otherwise fall back to an
        in-memory dict.

        Args:
            redis_url: Redis connection URL (e.g. "redis://localhost:6379/0").
                       Falls back to the REDIS_URL environment variable when
                       not supplied explicitly.
        """
        self._redis = None
        # memory store: key → (value_dict, expire_at_timestamp)
        self._memory: Dict[str, tuple] = {}

        url = redis_url or os.getenv("REDIS_URL")
        if url:
            try:
                import redis as _redis_lib

                client = _redis_lib.Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                client.ping()
                self._redis = client
                logger.info(f"Cache: connected to Redis ({url})")
            except Exception as exc:
                logger.warning(
                    f"Cache: Redis unavailable ({exc}) — using in-memory fallback"
                )
        else:
            logger.info("Cache: REDIS_URL not set — using in-memory fallback")

    @property
    def backend(self) -> str:
        """Return ``'redis'`` or ``'memory'`` depending on active backend."""
        return "redis" if self._redis is not None else "memory"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached value.

        Returns the deserialized dict on a hit, or ``None`` on a miss /
        expired entry.  Redis errors are treated as misses and logged.
        """
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                if raw is None:
                    return None
                return json.loads(raw)
            except Exception as exc:
                logger.warning(f"Cache get error ({exc}), treating as miss")
                return None

        # ── in-memory path ────────────────────────────────────────────────
        entry = self._memory.get(key)
        if entry is None:
            return None
        value, expire_at = entry
        if time.time() >= expire_at:
            del self._memory[key]
            return None
        return value

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
        """
        Store a value with the given TTL.

        Silently no-ops on Redis errors so a cache failure never crashes the
        pipeline.
        """
        if self._redis is not None:
            try:
                self._redis.setex(key, ttl_seconds, json.dumps(value))
            except Exception as exc:
                logger.warning(f"Cache set error: {exc}")
            return

        # ── in-memory path ────────────────────────────────────────────────
        self._memory[key] = (value, time.time() + ttl_seconds)

    @staticmethod
    def make_cache_key(symbol: str, days_back: int, date_str: str) -> str:
        """
        Build a deterministic cache key for a stock analysis result.

        Args:
            symbol:    Ticker symbol (should already be uppercased).
            days_back: Look-back window used for the analysis.
            date_str:  Hourly timestamp string, e.g. ``"2024-01-15-14"``.
                       Hour granularity means the same analysis is served from
                       cache for the full hour it was computed in.

        Returns:
            Key string, e.g. ``"analysis:AAPL:7:2024-01-15-14"``.
        """
        return f"analysis:{symbol}:{days_back}:{date_str}"
