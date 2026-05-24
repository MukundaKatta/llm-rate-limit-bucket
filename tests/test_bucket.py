import threading
import time

import pytest

from llm_rate_limit_bucket import AsyncTokenBucket, TokenBucket

# ---------- construction validation ----------


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=0, burst=10)
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=-1.0, burst=10)


def test_invalid_burst_raises():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=10.0, burst=0)
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=10.0, burst=-5)


# ---------- try_acquire ----------


def test_full_bucket_try_acquire_burst_times():
    b = TokenBucket(rate_per_sec=1.0, burst=5)
    # Bucket starts FULL, so the first `burst` calls all succeed.
    for _ in range(5):
        assert b.try_acquire() is True
    # Sixth call fails - no tokens left, and we haven't waited.
    assert b.try_acquire() is False


def test_empty_bucket_try_acquire_returns_false():
    b = TokenBucket(rate_per_sec=0.001, burst=1)
    assert b.try_acquire() is True  # drains the one token
    assert b.try_acquire() is False  # bucket is empty, refill is glacial


def test_try_acquire_n_consumes_n_tokens():
    b = TokenBucket(rate_per_sec=1.0, burst=10)
    assert b.try_acquire(3) is True
    assert b.try_acquire(3) is True
    assert b.try_acquire(3) is True
    # 9 of 10 consumed; 4 more is too many.
    assert b.try_acquire(4) is False
    # but 1 still fits.
    assert b.try_acquire(1) is True


def test_try_acquire_no_decrement_on_failure():
    b = TokenBucket(rate_per_sec=0.001, burst=2)
    assert b.try_acquire(2) is True
    # Failed acquire must not touch the count.
    assert b.try_acquire(1) is False
    # available() should still be ~0, not negative.
    assert b.available() >= 0.0


def test_try_acquire_invalid_n_raises():
    b = TokenBucket(rate_per_sec=1.0, burst=5)
    with pytest.raises(ValueError):
        b.try_acquire(0)
    with pytest.raises(ValueError):
        b.try_acquire(-1)
    with pytest.raises(ValueError):
        b.try_acquire(6)  # > burst, can never succeed


# ---------- refill over time ----------


def test_refill_over_time():
    # 100 tokens/sec, burst 1. After draining, ~0.05s sleep gives ~5 tokens
    # but burst caps at 1, so we see 1 token back.
    b = TokenBucket(rate_per_sec=100.0, burst=1)
    assert b.try_acquire() is True
    assert b.try_acquire() is False
    time.sleep(0.05)
    assert b.try_acquire() is True


def test_refill_capped_at_burst():
    b = TokenBucket(rate_per_sec=1000.0, burst=3)
    # Drain it.
    assert b.try_acquire(3) is True
    # Wait long enough that 1000 tokens "should" have refilled.
    time.sleep(0.1)
    # But burst caps at 3.
    avail = b.available()
    assert avail <= 3.0
    assert avail >= 2.5  # roughly full again


def test_burst_respected_from_start():
    b = TokenBucket(rate_per_sec=1.0, burst=4)
    # Starts FULL at burst.
    assert b.available() == pytest.approx(4.0, rel=0.05)


# ---------- acquire (blocking) ----------


def test_acquire_returns_true_immediately_when_available():
    b = TokenBucket(rate_per_sec=1.0, burst=2)
    t0 = time.monotonic()
    assert b.acquire() is True
    assert time.monotonic() - t0 < 0.05  # essentially no wait


def test_acquire_waits_then_returns_true():
    # 20 tokens/sec, burst 1. Drain it, then acquire(1) should wait ~0.05s.
    b = TokenBucket(rate_per_sec=20.0, burst=1)
    assert b.try_acquire() is True
    t0 = time.monotonic()
    assert b.acquire(timeout=1.0) is True
    elapsed = time.monotonic() - t0
    assert 0.03 < elapsed < 0.25


def test_acquire_timeout_returns_false():
    # 1 token/sec, burst 1. Drain it. acquire with 0.05s timeout cannot wait
    # the full 1s needed for refill, so returns False.
    b = TokenBucket(rate_per_sec=1.0, burst=1)
    assert b.try_acquire() is True
    t0 = time.monotonic()
    assert b.acquire(timeout=0.05) is False
    elapsed = time.monotonic() - t0
    assert 0.03 < elapsed < 0.5


def test_acquire_zero_timeout_is_try_acquire():
    b = TokenBucket(rate_per_sec=0.001, burst=1)
    assert b.acquire(timeout=0) is True  # full
    assert b.acquire(timeout=0) is False  # empty now


def test_acquire_negative_timeout_raises():
    b = TokenBucket(rate_per_sec=1.0, burst=1)
    with pytest.raises(ValueError):
        b.acquire(timeout=-0.1)


# ---------- wait_seconds ----------


def test_wait_seconds_zero_when_available():
    b = TokenBucket(rate_per_sec=10.0, burst=5)
    assert b.wait_seconds(1) == 0.0
    assert b.wait_seconds(5) == 0.0


def test_wait_seconds_estimates_correctly():
    # 10 tokens/sec, burst 2. Drain. Then waiting for 1 token = ~0.1s.
    b = TokenBucket(rate_per_sec=10.0, burst=2)
    assert b.try_acquire(2) is True
    w = b.wait_seconds(1)
    # Allow generous slop for elapsed time during the call itself.
    assert 0.05 < w <= 0.11


def test_wait_seconds_does_not_reserve():
    # wait_seconds is read-only; calling it shouldn't consume tokens.
    b = TokenBucket(rate_per_sec=1.0, burst=5)
    _ = b.wait_seconds(1)
    _ = b.wait_seconds(1)
    assert b.available() == pytest.approx(5.0, rel=0.05)


# ---------- available ----------


def test_available_decreases_after_acquire():
    b = TokenBucket(rate_per_sec=0.0001, burst=10)
    a0 = b.available()
    assert b.try_acquire(3) is True
    a1 = b.available()
    assert a0 - a1 == pytest.approx(3.0, abs=0.01)


# ---------- concurrency ----------


def test_concurrent_try_acquire_does_not_over_issue():
    """10 threads race for 5 tokens. Exactly 5 should succeed."""
    b = TokenBucket(rate_per_sec=0.0001, burst=5)
    results = []
    barrier = threading.Barrier(10)

    def worker():
        barrier.wait()  # release them all at once
        results.append(b.try_acquire())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = sum(1 for r in results if r)
    assert successes == 5


def test_concurrent_acquire_with_timeout_does_not_over_issue():
    """20 threads race for 5 tokens with short timeout. At most 5 succeed."""
    b = TokenBucket(rate_per_sec=0.0001, burst=5)
    results = []
    barrier = threading.Barrier(20)
    lock = threading.Lock()

    def worker():
        barrier.wait()
        ok = b.acquire(timeout=0.05)
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = sum(1 for r in results if r)
    assert successes == 5


# ---------- async ----------


async def test_async_try_acquire_basic():
    b = AsyncTokenBucket(rate_per_sec=1.0, burst=3)
    assert await b.try_acquire() is True
    assert await b.try_acquire() is True
    assert await b.try_acquire() is True
    assert await b.try_acquire() is False


async def test_async_acquire_waits_then_returns_true():
    # 20 tokens/sec, burst 1. Drain it, then async-acquire ~0.05s.
    b = AsyncTokenBucket(rate_per_sec=20.0, burst=1)
    assert await b.try_acquire() is True
    t0 = time.monotonic()
    assert await b.acquire(timeout=1.0) is True
    elapsed = time.monotonic() - t0
    assert 0.03 < elapsed < 0.3


async def test_async_acquire_timeout_returns_false():
    b = AsyncTokenBucket(rate_per_sec=1.0, burst=1)
    assert await b.try_acquire() is True
    t0 = time.monotonic()
    assert await b.acquire(timeout=0.05) is False
    assert time.monotonic() - t0 < 0.5


async def test_async_wait_seconds_estimates():
    b = AsyncTokenBucket(rate_per_sec=10.0, burst=2)
    assert await b.try_acquire(2) is True
    w = await b.wait_seconds(1)
    assert 0.05 < w <= 0.11


async def test_async_available_decreases_after_acquire():
    b = AsyncTokenBucket(rate_per_sec=0.0001, burst=10)
    a0 = await b.available()
    assert await b.try_acquire(3) is True
    a1 = await b.available()
    assert a0 - a1 == pytest.approx(3.0, abs=0.01)


async def test_async_invalid_n_raises():
    b = AsyncTokenBucket(rate_per_sec=1.0, burst=3)
    with pytest.raises(ValueError):
        await b.try_acquire(0)
    with pytest.raises(ValueError):
        await b.try_acquire(4)


async def test_async_construction_validates_inputs():
    with pytest.raises(ValueError):
        AsyncTokenBucket(rate_per_sec=0, burst=10)
    with pytest.raises(ValueError):
        AsyncTokenBucket(rate_per_sec=1.0, burst=0)


# ---------- properties ----------


def test_rate_and_burst_properties():
    b = TokenBucket(rate_per_sec=50.0, burst=100)
    assert b.rate_per_sec == 50.0
    assert b.burst == 100


def test_async_rate_and_burst_properties():
    b = AsyncTokenBucket(rate_per_sec=7.5, burst=15)
    assert b.rate_per_sec == 7.5
    assert b.burst == 15
