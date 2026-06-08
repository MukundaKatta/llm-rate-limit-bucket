"""
llm-rate-limit-bucket: Token-bucket rate limiter for LLM API calls.
"""

from __future__ import annotations

import time
from typing import Any, Callable


class RateLimitExceeded(Exception):
    """Raised when the rate limit is exceeded and blocking is disabled."""

    def __init__(self, bucket_name: str, retry_after: float) -> None:
        self.bucket_name = bucket_name
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for '{bucket_name}'. Retry after {retry_after:.2f}s."
        )


class TokenBucket:
    """
    Token-bucket rate limiter.

    Tokens refill at ``rate_per_second`` up to ``capacity``.
    Each call consumes ``cost`` tokens (default 1.0).

    Usage::

        bucket = TokenBucket(capacity=10, rate_per_second=2.0, name="anthropic")
        bucket.acquire()           # blocks if needed
        bucket.try_acquire()       # returns False if not available
        bucket.wait_time()         # seconds until next token available
    """

    def __init__(
        self,
        capacity: float,
        rate_per_second: float,
        name: str = "default",
    ) -> None:
        if capacity <= 0 or rate_per_second <= 0:
            raise ValueError("capacity and rate_per_second must be positive")
        self._capacity = capacity
        self._rate = rate_per_second
        self._name = name
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def try_acquire(self, cost: float = 1.0) -> bool:
        """Consume tokens if available. Returns True on success, False on failure."""
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False

    def acquire(self, cost: float = 1.0, block: bool = True) -> None:
        """
        Consume tokens, blocking until available if ``block=True``.
        Raises RateLimitExceeded if block=False and tokens unavailable.
        """
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return
        wait = (cost - self._tokens) / self._rate
        if not block:
            raise RateLimitExceeded(self._name, wait)
        time.sleep(wait)
        self._refill()
        self._tokens -= cost

    def wait_time(self, cost: float = 1.0) -> float:
        """Seconds until cost tokens are available (0 if already available)."""
        self._refill()
        if self._tokens >= cost:
            return 0.0
        return (cost - self._tokens) / self._rate

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def name(self) -> str:
        return self._name

    def wrap(self, cost: float = 1.0, block: bool = True) -> Callable[..., Any]:
        """Decorator: acquire before each call."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.acquire(cost=cost, block=block)
                return fn(*args, **kwargs)

            return wrapper

        return decorator


class MultiKeyBucket:
    """
    Maintain a separate TokenBucket per key (e.g. per API key or per model).
    """

    def __init__(self, capacity: float, rate_per_second: float) -> None:
        self._capacity = capacity
        self._rate = rate_per_second
        self._buckets: dict[str, TokenBucket] = {}

    def _get(self, key: str) -> TokenBucket:
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(self._capacity, self._rate, name=key)
        return self._buckets[key]

    def acquire(self, key: str, cost: float = 1.0, block: bool = True) -> None:
        self._get(key).acquire(cost=cost, block=block)

    def try_acquire(self, key: str, cost: float = 1.0) -> bool:
        return self._get(key).try_acquire(cost=cost)

    def wait_time(self, key: str, cost: float = 1.0) -> float:
        return self._get(key).wait_time(cost=cost)

    def keys(self) -> list[str]:
        return list(self._buckets.keys())


__all__ = ["TokenBucket", "MultiKeyBucket", "RateLimitExceeded"]
