"""Unit tests for BoundedLRUCache."""

import time
from pathlib import Path

import pytest

from td_linter.cache import (
    BoundedLRUCache,
    CacheEntry,
    CacheStats,
    cached,
)


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_default_values(self) -> None:
        """Should have zero values by default."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0

    def test_hit_rate_empty(self) -> None:
        """Hit rate should be 0 when no accesses."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Hit rate should be calculated correctly."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 75.0

    def test_reset(self) -> None:
        """Reset should clear all statistics."""
        stats = CacheStats(hits=10, misses=5, evictions=2, expirations=1)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_creates_with_timestamp(self) -> None:
        """Should record creation timestamp."""
        before = time.time()
        entry = CacheEntry(value="test")
        after = time.time()
        assert before <= entry.created_at <= after

    def test_not_expired_without_ttl(self) -> None:
        """Should never expire when TTL is None."""
        entry = CacheEntry(value="test")
        assert not entry.is_expired(None)

    def test_not_expired_within_ttl(self) -> None:
        """Should not expire within TTL."""
        entry = CacheEntry(value="test")
        assert not entry.is_expired(60.0)

    def test_expired_after_ttl(self) -> None:
        """Should expire after TTL."""
        entry = CacheEntry(value="test", created_at=time.time() - 10)
        assert entry.is_expired(5.0)


class TestBoundedLRUCache:
    """Tests for BoundedLRUCache."""

    def test_invalid_max_size_raises(self) -> None:
        """Should raise error for invalid max_size."""
        with pytest.raises(ValueError) as exc_info:
            BoundedLRUCache(max_size=0)
        assert "max_size must be positive" in str(exc_info.value)

    def test_set_and_get(self) -> None:
        """Should store and retrieve values."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache.set("key", 42)
        assert cache.get("key") == 42

    def test_get_missing_returns_default(self) -> None:
        """Should return default for missing keys."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        assert cache.get("missing") is None
        assert cache.get("missing", 0) == 0

    def test_contains(self) -> None:
        """Should support 'in' operator."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache.set("key", 42)
        assert "key" in cache
        assert "missing" not in cache

    def test_len(self) -> None:
        """Should return correct length."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        assert len(cache) == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2

    def test_delete(self) -> None:
        """Should remove entries."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache.set("key", 42)
        assert cache.delete("key") is True
        assert cache.get("key") is None
        assert cache.delete("missing") is False

    def test_clear(self) -> None:
        """Should remove all entries."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert len(cache) == 0

    def test_keys(self) -> None:
        """Should return all keys."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache.set("a", 1)
        cache.set("b", 2)
        keys = cache.keys()
        assert set(keys) == {"a", "b"}

    def test_items(self) -> None:
        """Should return all key-value pairs."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache.set("a", 1)
        cache.set("b", 2)
        items = cache.items()
        assert set(items) == {("a", 1), ("b", 2)}


class TestBoundedLRUCacheLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_evicts_oldest_when_full(self) -> None:
        """Should evict LRU entry when capacity exceeded."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_get_updates_lru_order(self) -> None:
        """Accessing entry should move it to end (most recent)."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # Access "a" to make it most recent
        cache.set("c", 3)  # Should evict "b" (now LRU)

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_update_existing_key(self) -> None:
        """Updating existing key should not increase size."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("a", 10)  # Update "a"

        assert len(cache) == 2
        assert cache.get("a") == 10


class TestBoundedLRUCacheTTL:
    """Tests for TTL expiration behavior."""

    def test_entry_expires_after_ttl(self) -> None:
        """Entry should expire after TTL."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(ttl_seconds=0.1)
        cache.set("key", 42)

        # Should be available immediately
        assert cache.get("key") == 42

        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("key") is None

    def test_no_expiration_without_ttl(self) -> None:
        """Entries should not expire when TTL is None."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(ttl_seconds=None)
        cache.set("key", 42)
        # Manipulate internal entry to have old timestamp
        cache._cache["key"].created_at = 0
        assert cache.get("key") == 42


class TestBoundedLRUCacheStats:
    """Tests for cache statistics."""

    def test_tracks_hits(self) -> None:
        """Should track cache hits."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache.set("key", 42)
        cache.get("key")
        cache.get("key")
        assert cache.stats.hits == 2

    def test_tracks_misses(self) -> None:
        """Should track cache misses."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        cache.get("missing")
        cache.get("missing")
        assert cache.stats.misses == 2

    def test_tracks_evictions(self) -> None:
        """Should track evictions."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(max_size=1)
        cache.set("a", 1)
        cache.set("b", 2)  # Evicts "a"
        assert cache.stats.evictions == 1

    def test_tracks_expirations(self) -> None:
        """Should track expirations."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(ttl_seconds=0.05)
        cache.set("key", 42)
        time.sleep(0.1)
        cache.get("key")  # Triggers expiration
        assert cache.stats.expirations == 1


class TestBoundedLRUCacheEvictionCallback:
    """Tests for eviction callback."""

    def test_calls_on_evict_for_lru_eviction(self) -> None:
        """Should call callback when entry evicted due to size."""
        evicted: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            max_size=1,
            on_evict=lambda k, v: evicted.append((k, v)),
        )
        cache.set("a", 1)
        cache.set("b", 2)

        assert evicted == [("a", 1)]

    def test_calls_on_evict_for_ttl_expiration(self) -> None:
        """Should call callback when entry expires."""
        evicted: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            ttl_seconds=0.05,
            on_evict=lambda k, v: evicted.append((k, v)),
        )
        cache.set("key", 42)
        time.sleep(0.1)
        cache.get("key")  # Triggers expiration

        assert evicted == [("key", 42)]

    def test_calls_on_evict_for_delete(self) -> None:
        """Should call callback when entry explicitly deleted."""
        evicted: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            on_evict=lambda k, v: evicted.append((k, v)),
        )
        cache.set("key", 42)
        cache.delete("key")

        assert evicted == [("key", 42)]

    def test_calls_on_evict_for_clear(self) -> None:
        """Should call callback for all entries when cleared."""
        evicted: list[tuple[str, int]] = []
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(
            on_evict=lambda k, v: evicted.append((k, v)),
        )
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()

        assert set(evicted) == {("a", 1), ("b", 2)}


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_caches_function_results(self) -> None:
        """Should cache function results."""
        call_count = 0

        @cached(max_size=10)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count == 1  # Only called once

    def test_different_args_different_cache(self) -> None:
        """Different args should have separate cache entries."""
        call_count = 0

        @cached(max_size=10)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(6) == 12
        assert call_count == 2

    def test_cache_clear(self) -> None:
        """Should support clearing the cache."""
        call_count = 0

        @cached(max_size=10)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive(5)
        expensive.cache_clear()
        expensive(5)
        assert call_count == 2

    def test_cache_stats_accessible(self) -> None:
        """Should expose cache stats."""

        @cached(max_size=10)
        def expensive(x: int) -> int:
            return x * 2

        expensive(5)
        expensive(5)
        assert expensive.cache_stats.hits == 1


class TestBoundedLRUCacheWithPathKeys:
    """Tests for using Path objects as cache keys (common pattern in td-linter)."""

    def test_path_as_key(self, tmp_path: Path) -> None:
        """Should support Path objects as keys."""
        cache: BoundedLRUCache[Path, list[str]] = BoundedLRUCache()
        path = tmp_path / "test.txt"
        cache.set(path, ["line1", "line2"])
        assert cache.get(path) == ["line1", "line2"]

    def test_different_paths_different_entries(self, tmp_path: Path) -> None:
        """Different paths should be different cache entries."""
        cache: BoundedLRUCache[Path, list[str]] = BoundedLRUCache()
        path1 = tmp_path / "a.txt"
        path2 = tmp_path / "b.txt"

        cache.set(path1, ["a"])
        cache.set(path2, ["b"])

        assert cache.get(path1) == ["a"]
        assert cache.get(path2) == ["b"]
