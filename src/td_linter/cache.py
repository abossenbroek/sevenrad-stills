"""Bounded LRU cache with TTL support.

Provides a cache implementation with:
- Maximum entry limit (LRU eviction)
- Time-to-live for entries (automatic expiration)
- Thread-safe operations
- Cache statistics for debugging
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class CacheStats:
    """Statistics about cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0


@dataclass
class CacheEntry(Generic[V]):
    """A single cache entry with value and timestamp."""

    value: V
    created_at: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: float | None) -> bool:
        """Check if this entry has expired."""
        if ttl_seconds is None:
            return False
        return time.time() - self.created_at > ttl_seconds


class BoundedLRUCache(Generic[K, V]):
    """Thread-safe LRU cache with maximum size and TTL support.

    Features:
    - Bounded size with LRU eviction policy
    - Optional TTL for automatic entry expiration
    - Thread-safe for concurrent access
    - Cache statistics for debugging and monitoring

    Example:
        cache: BoundedLRUCache[str, dict] = BoundedLRUCache(
            max_size=1000,
            ttl_seconds=300,  # 5 minute TTL
        )
        cache.set("key", {"data": "value"})
        result = cache.get("key")  # Returns {"data": "value"}
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float | None = 300.0,
        on_evict: Callable[[K, V], None] | None = None,
    ) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries. Must be positive.
            ttl_seconds: Time-to-live in seconds. None means no expiration.
            on_evict: Optional callback when entries are evicted or expired.
                      Called with (key, value) arguments.
        """
        if max_size < 1:
            msg = f"max_size must be positive, got {max_size}"
            raise ValueError(msg)

        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._on_evict = on_evict
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

    @property
    def max_size(self) -> int:
        """Maximum number of entries in the cache."""
        return self._max_size

    @property
    def ttl_seconds(self) -> float | None:
        """Time-to-live in seconds, or None if no TTL."""
        return self._ttl_seconds

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def __len__(self) -> int:
        """Return number of entries in the cache."""
        with self._lock:
            self._cleanup_expired()
            return len(self._cache)

    def __contains__(self, key: K) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def get(self, key: K, default: V | None = None) -> V | None:
        """Get a value from the cache.

        Args:
            key: The cache key.
            default: Value to return if key not found.

        Returns:
            The cached value, or default if not found or expired.
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            if entry.is_expired(self._ttl_seconds):
                # Entry has expired
                self._stats.misses += 1
                self._stats.expirations += 1
                self._remove_entry(key, entry)
                return default

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return entry.value

    def set(self, key: K, value: V) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        with self._lock:
            # If key exists, remove old entry first
            if key in self._cache:
                del self._cache[key]

            # Add new entry
            self._cache[key] = CacheEntry(value=value)

            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._evict_oldest()

    def delete(self, key: K) -> bool:
        """Remove a key from the cache.

        Args:
            key: The cache key to remove.

        Returns:
            True if the key was found and removed.
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                if self._on_evict:
                    self._on_evict(key, entry.value)
                return True
            return False

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            if self._on_evict:
                for key, entry in self._cache.items():
                    self._on_evict(key, entry.value)
            self._cache.clear()

    def keys(self) -> list[K]:
        """Return list of non-expired keys."""
        with self._lock:
            self._cleanup_expired()
            return list(self._cache.keys())

    def items(self) -> list[tuple[K, V]]:
        """Return list of (key, value) tuples for non-expired entries."""
        with self._lock:
            self._cleanup_expired()
            return [(k, e.value) for k, e in self._cache.items()]

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if not self._cache:
            return

        key, entry = self._cache.popitem(last=False)
        self._stats.evictions += 1
        if self._on_evict:
            self._on_evict(key, entry.value)

    def _remove_entry(self, key: K, entry: CacheEntry[V]) -> None:
        """Remove a specific entry and call eviction callback."""
        del self._cache[key]
        if self._on_evict:
            self._on_evict(key, entry.value)

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        if self._ttl_seconds is None:
            return

        expired_keys: list[K] = []
        for key, entry in self._cache.items():
            if entry.is_expired(self._ttl_seconds):
                expired_keys.append(key)

        for key in expired_keys:
            entry = self._cache.pop(key)
            self._stats.expirations += 1
            if self._on_evict:
                self._on_evict(key, entry.value)


def cached(
    max_size: int = 128,
    ttl_seconds: float | None = None,
) -> Callable[[Callable[..., V]], Callable[..., V]]:
    """Decorator to cache function results.

    Creates a function-specific LRU cache that caches results based
    on all arguments (which must be hashable).

    Args:
        max_size: Maximum number of cached results.
        ttl_seconds: Time-to-live for cached results.

    Returns:
        Decorated function with caching.

    Example:
        @cached(max_size=100, ttl_seconds=60)
        def expensive_lookup(key: str) -> dict:
            return fetch_from_database(key)
    """

    def decorator(func: Callable[..., V]) -> Callable[..., V]:
        cache: BoundedLRUCache[tuple, V] = BoundedLRUCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )

        def wrapper(*args: Hashable, **kwargs: Hashable) -> V:
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))

            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        # Expose cache for testing/debugging
        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = cache.stats  # type: ignore[attr-defined]

        return wrapper

    return decorator
