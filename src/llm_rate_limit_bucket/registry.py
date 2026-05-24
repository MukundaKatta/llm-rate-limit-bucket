"""Per-key bucket registries.

Most API quotas are scoped per key (or per provider, or per model
endpoint). `BucketRegistry` and `AsyncBucketRegistry` let callers ask
for a bucket by string key; one is created on first access using the
registry's default `rate_per_sec` / `burst`.

Override the bucket for a key via `.set(key, bucket)` (useful when one
key has a different quota than the rest).
"""

from __future__ import annotations

import threading

from llm_rate_limit_bucket.bucket import AsyncTokenBucket, TokenBucket


class BucketRegistry:
    """Thread-safe map of key -> `TokenBucket`.

    New keys default to `TokenBucket(rate_per_sec, burst)`. Override a
    single key with `.set(key, custom_bucket)` if its quota differs.
    """

    __slots__ = ("_rate", "_burst", "_buckets", "_lock")

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")
        self._rate: float = float(rate_per_sec)
        self._burst: int = int(burst)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    @property
    def default_rate_per_sec(self) -> float:
        return self._rate

    @property
    def default_burst(self) -> int:
        return self._burst

    def get(self, key: str) -> TokenBucket:
        """Return the bucket for `key`, creating one with the defaults on first access."""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(rate_per_sec=self._rate, burst=self._burst)
                self._buckets[key] = bucket
            return bucket

    def set(self, key: str, bucket: TokenBucket) -> None:
        """Replace (or insert) the bucket for `key`. The previous bucket is dropped."""
        if not isinstance(bucket, TokenBucket):
            raise TypeError("bucket must be a TokenBucket instance")
        with self._lock:
            self._buckets[key] = bucket

    def keys(self) -> list[str]:
        """Snapshot of registered keys at the moment of the call."""
        with self._lock:
            return list(self._buckets.keys())

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._buckets

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)


class AsyncBucketRegistry:
    """Map of key -> `AsyncTokenBucket`.

    Same shape as `BucketRegistry`, but stores async buckets. The
    registry itself uses a `threading.Lock` (not `asyncio.Lock`)
    because `.get` / `.set` themselves are tiny and synchronous; only
    the bucket's `acquire` awaits.
    """

    __slots__ = ("_rate", "_burst", "_buckets", "_lock")

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")
        self._rate: float = float(rate_per_sec)
        self._burst: int = int(burst)
        self._buckets: dict[str, AsyncTokenBucket] = {}
        self._lock = threading.Lock()

    @property
    def default_rate_per_sec(self) -> float:
        return self._rate

    @property
    def default_burst(self) -> int:
        return self._burst

    def get(self, key: str) -> AsyncTokenBucket:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = AsyncTokenBucket(rate_per_sec=self._rate, burst=self._burst)
                self._buckets[key] = bucket
            return bucket

    def set(self, key: str, bucket: AsyncTokenBucket) -> None:
        if not isinstance(bucket, AsyncTokenBucket):
            raise TypeError("bucket must be an AsyncTokenBucket instance")
        with self._lock:
            self._buckets[key] = bucket

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._buckets.keys())

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._buckets

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)
