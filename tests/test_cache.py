"""
Unit tests for CacheClient.

Run with:
    pytest tests/test_cache.py -v
"""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.cache import CacheClient


# ===========================================================================
# CacheClient tests
# ===========================================================================

class TestCacheClientBackendSelection:
    """Tests that verify which backend is chosen based on environment / Redis availability."""

    def test_no_redis_url_uses_memory_backend(self):
        """Without REDIS_URL the backend is always in-memory."""
        c = CacheClient(redis_url=None)
        assert c.backend == "memory"

    def test_redis_import_error_falls_back_to_memory(self):
        """If the redis package is absent the client falls back to memory."""
        with patch.dict("sys.modules", {"redis": None}):
            c = CacheClient(redis_url="redis://localhost:6379")
        assert c.backend == "memory"

    def test_redis_connection_error_falls_back_to_memory(self):
        """A connection-refused error from Redis falls back to memory."""
        pytest.importorskip("redis")
        import redis as _redis_lib

        with patch.object(_redis_lib, "Redis") as mock_cls:
            mock_client = MagicMock()
            mock_client.ping.side_effect = ConnectionRefusedError("refused")
            mock_cls.from_url.return_value = mock_client
            c = CacheClient(redis_url="redis://localhost:6379")

        assert c.backend == "memory"

    def test_successful_redis_connection_uses_redis_backend(self):
        """A successful ping means the Redis backend is used."""
        pytest.importorskip("redis")
        import redis as _redis_lib

        with patch.object(_redis_lib, "Redis") as mock_cls:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_cls.from_url.return_value = mock_client
            c = CacheClient(redis_url="redis://localhost:6379")

        assert c.backend == "redis"


class TestCacheClientMemoryFallback:
    """Tests for the in-memory dict fallback (no Redis required)."""

    def test_miss_returns_none(self):
        c = CacheClient()
        assert c.get("nonexistent") is None

    def test_set_and_get_roundtrip(self):
        c = CacheClient()
        payload = {"recommendation": "BUY", "score": 0.8, "symbol": "AAPL"}
        c.set("k1", payload, ttl_seconds=60)
        assert c.get("k1") == payload

    def test_overwrite_replaces_value(self):
        c = CacheClient()
        c.set("k2", {"v": 1}, ttl_seconds=60)
        c.set("k2", {"v": 2}, ttl_seconds=60)
        assert c.get("k2") == {"v": 2}

    def test_multiple_keys_independent(self):
        c = CacheClient()
        c.set("a", {"x": 1}, ttl_seconds=60)
        c.set("b", {"x": 2}, ttl_seconds=60)
        assert c.get("a") == {"x": 1}
        assert c.get("b") == {"x": 2}


class TestCacheClientTTLExpiry:
    """TTL expiry tests — require freezegun."""

    def test_expired_entry_returns_none(self):
        """After TTL elapses, get() returns None."""
        from datetime import datetime, timedelta
        freezegun = pytest.importorskip("freezegun")
        freeze_time = freezegun.freeze_time

        c = CacheClient()
        base = datetime(2024, 1, 15, 12, 0, 0)

        with freeze_time(base):
            c.set("ttl_key", {"data": "value"}, ttl_seconds=30)

        # 31 seconds later — TTL has elapsed
        with freeze_time(base + timedelta(seconds=31)):
            result = c.get("ttl_key")

        assert result is None

    def test_expired_entry_is_evicted_from_memory(self):
        """Expired entry is removed from the internal dict on access."""
        from datetime import datetime, timedelta
        freezegun = pytest.importorskip("freezegun")
        freeze_time = freezegun.freeze_time

        c = CacheClient()
        base = datetime(2024, 1, 15, 12, 0, 0)

        with freeze_time(base):
            c.set("evict_key", {"data": "value"}, ttl_seconds=10)

        with freeze_time(base + timedelta(seconds=11)):
            c.get("evict_key")  # triggers eviction

        assert "evict_key" not in c._memory

    def test_entry_valid_before_ttl_elapses(self):
        """Before TTL elapses, get() returns the stored value."""
        from datetime import datetime, timedelta
        freezegun = pytest.importorskip("freezegun")
        freeze_time = freezegun.freeze_time

        c = CacheClient()
        base = datetime(2024, 1, 15, 12, 0, 0)

        with freeze_time(base):
            c.set("fresh_key", {"data": "fresh"}, ttl_seconds=60)

        # Only 10 seconds have passed
        with freeze_time(base + timedelta(seconds=10)):
            result = c.get("fresh_key")

        assert result == {"data": "fresh"}

    def test_entry_valid_at_exact_ttl_boundary(self):
        """An entry retrieved at exactly its expiry time is treated as expired."""
        from datetime import datetime, timedelta
        freezegun = pytest.importorskip("freezegun")
        freeze_time = freezegun.freeze_time

        c = CacheClient()
        base = datetime(2024, 1, 15, 12, 0, 0)

        with freeze_time(base):
            c.set("boundary_key", {"v": 1}, ttl_seconds=30)

        # Exactly at expire_at: time.time() >= expire_at → expired
        with freeze_time(base + timedelta(seconds=30)):
            result = c.get("boundary_key")

        assert result is None


class TestCacheClientMakeCacheKey:
    """Tests for the static key-builder."""

    def test_key_format(self):
        key = CacheClient.make_cache_key("AAPL", 7, "2024-01-15-14")
        assert key == "analysis:AAPL:7:2024-01-15-14"

    def test_different_symbols_produce_different_keys(self):
        k1 = CacheClient.make_cache_key("AAPL", 7, "2024-01-15-14")
        k2 = CacheClient.make_cache_key("GOOGL", 7, "2024-01-15-14")
        assert k1 != k2

    def test_different_days_back_produce_different_keys(self):
        k1 = CacheClient.make_cache_key("AAPL", 7, "2024-01-15-14")
        k2 = CacheClient.make_cache_key("AAPL", 30, "2024-01-15-14")
        assert k1 != k2

    def test_different_hours_produce_different_keys(self):
        k1 = CacheClient.make_cache_key("AAPL", 7, "2024-01-15-14")
        k2 = CacheClient.make_cache_key("AAPL", 7, "2024-01-15-15")
        assert k1 != k2

    def test_callable_on_instance(self):
        """make_cache_key is a static method callable on an instance."""
        c = CacheClient()
        key = c.make_cache_key("TSLA", 14, "2024-06-01-09")
        assert key == "analysis:TSLA:14:2024-06-01-09"


class TestCacheClientRedisPath:
    """Tests for the Redis backend path (mocked)."""

    @pytest.fixture
    def redis_cache(self):
        """CacheClient backed by a fully-mocked Redis client."""
        pytest.importorskip("redis")
        import redis as _redis_lib

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        with patch.object(_redis_lib, "Redis") as mock_cls:
            mock_cls.from_url.return_value = mock_redis
            c = CacheClient(redis_url="redis://localhost:6379")
        # Stash the mock so tests can configure it
        c._mock_redis = mock_redis
        return c

    def test_get_miss_returns_none(self, redis_cache):
        redis_cache._mock_redis.get.return_value = None
        assert redis_cache.get("missing") is None

    def test_get_deserialises_json(self, redis_cache):
        import json
        payload = {"recommendation": "HOLD", "score": 0.5}
        redis_cache._mock_redis.get.return_value = json.dumps(payload)
        assert redis_cache.get("k") == payload

    def test_set_calls_setex_with_correct_ttl(self, redis_cache):
        import json
        payload = {"recommendation": "BUY"}
        redis_cache.set("k", payload, ttl_seconds=120)
        redis_cache._mock_redis.setex.assert_called_once_with(
            "k", 120, json.dumps(payload)
        )

    def test_get_error_returns_none(self, redis_cache):
        """A Redis error during get is silently treated as a miss."""
        redis_cache._mock_redis.get.side_effect = Exception("Redis down")
        assert redis_cache.get("k") is None

    def test_set_error_is_silent(self, redis_cache):
        """A Redis error during set does not propagate."""
        redis_cache._mock_redis.setex.side_effect = Exception("Redis down")
        # Should not raise
        redis_cache.set("k", {"v": 1}, ttl_seconds=60)


# ===========================================================================
# Session-scoped setup
# ===========================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    # Ensure no REDIS_URL leaks from the real environment into cache tests
    os.environ.pop("REDIS_URL", None)
    yield
