"""Core TokenBucket and AsyncTokenBucket implementations.

Classic token-bucket algorithm sized for LLM API quotas:

  * Capacity is `burst`. Tokens replenish continuously at `rate_per_sec`
    tokens per second using `time.monotonic()` for the clock.
  * `try_acquire(n)` is non-blocking and either decrements the bucket
    by `n` (returns True) or leaves it untouched (returns False).
  * `acquire(n, timeout)` blocks (or awaits) up to `timeout` seconds,
    sleeping until enough tokens have refilled.
  * `wait_seconds(n)` returns the projected delay until `n` tokens are
    available. Useful for callers that want to budget their own sleep.

Both the sync (`TokenBucket`, using `threading.Lock`) and async
(`AsyncTokenBucket`, using `asyncio.Lock`) variants share the same
math; only the locking primitive and the sleep primitive differ.
"""

from __future__ import annotations

import asyncio
import threading
import time


class TokenBucket:
    """Thread-safe classic token-bucket rate limiter.

    Args:
      rate_per_sec: tokens added per second (continuous refill).
      burst: maximum tokens that can sit in the bucket; this is also
        the largest single `acquire` that can ever succeed.

    The bucket starts FULL (`burst` tokens available), so the first
    `burst` calls succeed without waiting. After that, each call costs
    one token (or `n`) and waits / fails until enough have refilled.
    """

    __slots__ = ("_rate", "_burst", "_tokens", "_last", "_lock")

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")
        self._rate: float = float(rate_per_sec)
        self._burst: int = int(burst)
        self._tokens: float = float(burst)
        self._last: float = time.monotonic()
        self._lock = threading.Lock()

    # ---- introspection ----

    @property
    def rate_per_sec(self) -> float:
        return self._rate

    @property
    def burst(self) -> int:
        return self._burst

    def available(self) -> float:
        """Return current token count (continuous, may include fractional refill)."""
        with self._lock:
            self._refill_locked()
            return self._tokens

    # ---- acquisition ----

    def try_acquire(self, n: int = 1) -> bool:
        """Non-blocking acquire of `n` tokens.

        Returns True and decrements if `n` tokens are available right now.
        Returns False without decrementing otherwise.

        Raises ValueError if `n` is < 1 or > burst (the latter could
        never succeed).
        """
        self._validate_n(n)
        with self._lock:
            self._refill_locked()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def acquire(self, n: int = 1, timeout: float | None = None) -> bool:
        """Block the calling thread until `n` tokens are available.

        timeout=None waits indefinitely. timeout=0 is equivalent to
        try_acquire. Returns True if acquired, False on timeout.
        """
        self._validate_n(n)
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be >= 0 or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        # Loop because, in theory, another waiter could win the race
        # between our sleep and our retry. In practice this is rare.
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= n:
                    self._tokens -= n
                    return True
                wait = (n - self._tokens) / self._rate
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            # Sleep outside the lock so other callers can still refill / probe.
            time.sleep(wait)

    def wait_seconds(self, n: int = 1) -> float:
        """How long until `n` tokens will be available.

        Returns 0.0 if `n` tokens are available right now. Does not
        reserve or hold the tokens; the answer is a snapshot.
        """
        self._validate_n(n)
        with self._lock:
            self._refill_locked()
            if self._tokens >= n:
                return 0.0
            return (n - self._tokens) / self._rate

    # ---- internal ----

    def _validate_n(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self._burst:
            raise ValueError(f"n ({n}) > burst ({self._burst}); request can never succeed")

    def _refill_locked(self) -> None:
        """Refill tokens based on elapsed monotonic time. Caller holds the lock."""
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(float(self._burst), self._tokens + elapsed * self._rate)
            self._last = now


class AsyncTokenBucket:
    """asyncio-friendly mirror of `TokenBucket`.

    Same algorithm; uses `asyncio.Lock` and `asyncio.sleep` so that
    waiting acquires do not block the event loop. Construct and use
    inside a running event loop.
    """

    __slots__ = ("_rate", "_burst", "_tokens", "_last", "_lock")

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")
        self._rate: float = float(rate_per_sec)
        self._burst: int = int(burst)
        self._tokens: float = float(burst)
        self._last: float = time.monotonic()
        self._lock = asyncio.Lock()

    # ---- introspection ----

    @property
    def rate_per_sec(self) -> float:
        return self._rate

    @property
    def burst(self) -> int:
        return self._burst

    async def available(self) -> float:
        """Return current token count after refilling against the monotonic clock."""
        async with self._lock:
            self._refill_locked()
            return self._tokens

    # ---- acquisition ----

    async def try_acquire(self, n: int = 1) -> bool:
        """Non-blocking async acquire. Returns True / False like the sync version."""
        self._validate_n(n)
        async with self._lock:
            self._refill_locked()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    async def acquire(self, n: int = 1, timeout: float | None = None) -> bool:
        """Await until `n` tokens are available, up to `timeout` seconds.

        timeout=None waits indefinitely. Returns True if acquired,
        False on timeout.
        """
        self._validate_n(n)
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be >= 0 or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            async with self._lock:
                self._refill_locked()
                if self._tokens >= n:
                    self._tokens -= n
                    return True
                wait = (n - self._tokens) / self._rate
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            await asyncio.sleep(wait)

    async def wait_seconds(self, n: int = 1) -> float:
        """Async snapshot of how long until `n` tokens are available."""
        self._validate_n(n)
        async with self._lock:
            self._refill_locked()
            if self._tokens >= n:
                return 0.0
            return (n - self._tokens) / self._rate

    # ---- internal ----

    def _validate_n(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self._burst:
            raise ValueError(f"n ({n}) > burst ({self._burst}); request can never succeed")

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(float(self._burst), self._tokens + elapsed * self._rate)
            self._last = now
