"""llm-rate-limit-bucket: a token-bucket rate limiter for LLM API calls.

Classic token-bucket algorithm sized for LLM API quotas, with zero runtime
dependencies (standard library only):

  * A bucket holds up to ``capacity`` tokens and refills continuously at
    ``rate_per_second`` tokens per second, measured against
    ``time.monotonic()`` so it is immune to wall-clock changes.
  * ``try_acquire(cost)`` is non-blocking: it consumes ``cost`` tokens and
    returns ``True``, or leaves the bucket untouched and returns ``False``.
  * ``acquire(cost, block=True)`` blocks until enough tokens have refilled.
    With ``block=False`` it raises :class:`RateLimitExceeded` instead.
  * ``wait_time(cost)`` reports how long until ``cost`` tokens are available
    without consuming anything, so callers can budget their own sleep.

All operations are thread-safe. :class:`MultiKeyBucket` keeps one bucket per
key (e.g. per API key, per model, or per provider endpoint) so that one
caller's traffic does not exhaust another's quota.

Example::

    from llm_rate_limit_bucket import TokenBucket

    bucket = TokenBucket(capacity=10, rate_per_second=2.0, name="anthropic")
    bucket.acquire()            # blocks until a token is available
    if bucket.try_acquire():    # non-blocking
        ...

Use this proactively to stay under a known provider rate limit; for reactive
backoff after a 429 response, pair it with a retry helper.
"""
from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable, TypeVar

__version__ = "0.1.0"

F = TypeVar("F", bound=Callable[..., Any])


class RateLimitExceeded(Exception):
    """Raised when the rate limit is exceeded and blocking is disabled.

    Attributes:
        bucket_name: Name of the bucket that rejected the request.
        retry_after: Seconds the caller should wait before retrying.
    """

    def __init__(self, bucket_name: str, retry_after: float) -> None:
        self.bucket_name = bucket_name
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for '{bucket_name}'. "
            f"Retry after {retry_after:.2f}s."
        )


class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Tokens refill continuously at ``rate_per_second`` up to ``capacity``.
    Each call consumes ``cost`` tokens (default ``1.0``). The bucket starts
    full, so the first ``capacity`` tokens are available without waiting.

    Args:
        capacity: Maximum number of tokens the bucket can hold. This is also
            the largest single ``cost`` that can ever be satisfied.
        rate_per_second: Tokens added per second (continuous refill).
        name: Label used in error messages and :func:`repr`.

    Raises:
        ValueError: If ``capacity`` or ``rate_per_second`` is not positive.

    Example::

        bucket = TokenBucket(capacity=10, rate_per_second=2.0, name="anthropic")
        bucket.acquire()           # blocks if needed
        bucket.try_acquire()       # returns False if not available
        bucket.wait_time()         # seconds until next token available
    """

    __slots__ = ("_capacity", "_rate", "_name", "_tokens", "_last_refill", "_lock")

    def __init__(
        self,
        capacity: float,
        rate_per_second: float,
        name: str = "default",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._capacity = float(capacity)
        self._rate = float(rate_per_second)
        self._name = name
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        """Add tokens accrued since the last refill. Caller must hold the lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now

    def _validate_cost(self, cost: float) -> None:
        if cost <= 0:
            raise ValueError("cost must be positive")
        if cost > self._capacity:
            raise ValueError(
                f"cost ({cost}) > capacity ({self._capacity}); "
                "request can never succeed"
            )

    def try_acquire(self, cost: float = 1.0) -> bool:
        """Consume ``cost`` tokens if available.

        Returns:
            ``True`` and decrements the bucket if ``cost`` tokens are
            available right now; ``False`` (without decrementing) otherwise.

        Raises:
            ValueError: If ``cost`` is not positive or exceeds ``capacity``.
        """
        self._validate_cost(cost)
        with self._lock:
            self._refill_locked()
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    def acquire(self, cost: float = 1.0, block: bool = True) -> None:
        """Consume ``cost`` tokens, optionally blocking until available.

        Args:
            cost: Number of tokens to consume.
            block: If ``True`` (default), sleep until enough tokens have
                refilled. If ``False``, raise :class:`RateLimitExceeded`
                immediately when the tokens are not available.

        Raises:
            ValueError: If ``cost`` is not positive or exceeds ``capacity``.
            RateLimitExceeded: If ``block`` is ``False`` and tokens are
                unavailable.
        """
        self._validate_cost(cost)
        with self._lock:
            self._refill_locked()
            if self._tokens >= cost:
                self._tokens -= cost
                return
            wait = (cost - self._tokens) / self._rate
            if not block:
                raise RateLimitExceeded(self._name, wait)
        # Sleep outside the lock so other threads can still refill / probe,
        # then re-check in a loop to guard against lost races and any
        # floating-point shortfall in the elapsed-time refill.
        time.sleep(wait)
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                wait = (cost - self._tokens) / self._rate
            time.sleep(wait)

    def wait_time(self, cost: float = 1.0) -> float:
        """Return seconds until ``cost`` tokens are available.

        Returns ``0.0`` if ``cost`` tokens are available right now. Takes a
        snapshot only; it does not reserve or consume any tokens.

        Raises:
            ValueError: If ``cost`` is not positive or exceeds ``capacity``.
        """
        self._validate_cost(cost)
        with self._lock:
            self._refill_locked()
            if self._tokens >= cost:
                return 0.0
            return (cost - self._tokens) / self._rate

    def reset(self) -> None:
        """Refill the bucket to full capacity immediately."""
        with self._lock:
            self._tokens = self._capacity
            self._last_refill = time.monotonic()

    @property
    def available_tokens(self) -> float:
        """Current token count, including fractional refill since last access."""
        with self._lock:
            self._refill_locked()
            return self._tokens

    @property
    def capacity(self) -> float:
        """Maximum number of tokens the bucket can hold."""
        return self._capacity

    @property
    def rate_per_second(self) -> float:
        """Tokens added per second."""
        return self._rate

    @property
    def name(self) -> str:
        """Bucket label used in error messages and :func:`repr`."""
        return self._name

    def wrap(self, cost: float = 1.0, block: bool = True) -> Callable[[F], F]:
        """Return a decorator that acquires ``cost`` tokens before each call.

        Args:
            cost: Tokens to consume per call.
            block: Passed through to :meth:`acquire`.

        Example::

            bucket = TokenBucket(capacity=5, rate_per_second=1.0)

            @bucket.wrap()
            def call_llm(messages):
                ...
        """

        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.acquire(cost=cost, block=block)
                return fn(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def __repr__(self) -> str:
        return (
            f"TokenBucket(name={self._name!r}, capacity={self._capacity}, "
            f"rate_per_second={self._rate})"
        )


class MultiKeyBucket:
    """Maintain a separate :class:`TokenBucket` per key.

    Useful for limiting each API key, model, or provider endpoint
    independently so that one key's traffic cannot exhaust another's quota.
    Buckets are created lazily on first use with the shared ``capacity`` and
    ``rate_per_second``. Access is thread-safe.

    Args:
        capacity: Capacity for each per-key bucket.
        rate_per_second: Refill rate for each per-key bucket.

    Raises:
        ValueError: If ``capacity`` or ``rate_per_second`` is not positive.
    """

    __slots__ = ("_capacity", "_rate", "_buckets", "_lock")

    def __init__(self, capacity: float, rate_per_second: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._capacity = float(capacity)
        self._rate = float(rate_per_second)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _get(self, key: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self._capacity, self._rate, name=key)
                self._buckets[key] = bucket
            return bucket

    def acquire(self, key: str, cost: float = 1.0, block: bool = True) -> None:
        """Acquire ``cost`` tokens from ``key``'s bucket (see :meth:`TokenBucket.acquire`)."""
        self._get(key).acquire(cost=cost, block=block)

    def try_acquire(self, key: str, cost: float = 1.0) -> bool:
        """Non-blocking acquire from ``key``'s bucket (see :meth:`TokenBucket.try_acquire`)."""
        return self._get(key).try_acquire(cost=cost)

    def wait_time(self, key: str, cost: float = 1.0) -> float:
        """Seconds until ``cost`` tokens are available for ``key``."""
        return self._get(key).wait_time(cost=cost)

    def get(self, key: str) -> TokenBucket:
        """Return ``key``'s bucket, creating it with the defaults on first access."""
        return self._get(key)

    def keys(self) -> list[str]:
        """Snapshot of keys that currently have a bucket."""
        with self._lock:
            return list(self._buckets.keys())

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._buckets

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)


__all__ = ["TokenBucket", "MultiKeyBucket", "RateLimitExceeded", "__version__"]
