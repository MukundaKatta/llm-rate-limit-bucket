"""Tests for llm-rate-limit-bucket."""

import time
import pytest
from llm_rate_limit_bucket import TokenBucket, MultiKeyBucket, RateLimitExceeded


def test_try_acquire_success():
    b = TokenBucket(capacity=5, rate_per_second=1.0)
    assert b.try_acquire() is True


def test_try_acquire_exhausted():
    b = TokenBucket(capacity=2, rate_per_second=0.1)
    b.try_acquire()
    b.try_acquire()
    assert b.try_acquire() is False


def test_available_tokens():
    b = TokenBucket(capacity=10, rate_per_second=1.0)
    assert b.available_tokens == pytest.approx(10.0, abs=0.1)


def test_tokens_decrease_after_acquire():
    b = TokenBucket(capacity=10, rate_per_second=1.0)
    b.try_acquire(cost=3.0)
    assert b.available_tokens == pytest.approx(7.0, abs=0.1)


def test_acquire_non_blocking_raises():
    b = TokenBucket(capacity=1, rate_per_second=0.1)
    b.try_acquire()  # drain
    with pytest.raises(RateLimitExceeded) as exc_info:
        b.acquire(block=False)
    assert exc_info.value.retry_after > 0


def test_rate_limit_exceeded_has_bucket_name():
    b = TokenBucket(capacity=1, rate_per_second=0.1, name="myapi")
    b.try_acquire()
    with pytest.raises(RateLimitExceeded) as exc_info:
        b.acquire(block=False)
    assert exc_info.value.bucket_name == "myapi"


def test_wait_time_zero_when_available():
    b = TokenBucket(capacity=5, rate_per_second=1.0)
    assert b.wait_time() == 0.0


def test_wait_time_positive_when_drained():
    b = TokenBucket(capacity=1, rate_per_second=0.5)
    b.try_acquire()
    assert b.wait_time() > 0.0


def test_capacity_property():
    b = TokenBucket(capacity=20, rate_per_second=1.0)
    assert b.capacity == 20


def test_name_property():
    b = TokenBucket(capacity=5, rate_per_second=1.0, name="test")
    assert b.name == "test"


def test_invalid_capacity():
    with pytest.raises(ValueError):
        TokenBucket(capacity=0, rate_per_second=1.0)


def test_invalid_rate():
    with pytest.raises(ValueError):
        TokenBucket(capacity=5, rate_per_second=0)


def test_wrap_decorator():
    b = TokenBucket(capacity=5, rate_per_second=1.0)
    calls = []

    @b.wrap()
    def do_thing():
        calls.append(1)
        return "ok"

    result = do_thing()
    assert result == "ok"
    assert len(calls) == 1
    assert b.available_tokens < 5.0


def test_multi_key_bucket_separate_buckets():
    mb = MultiKeyBucket(capacity=2, rate_per_second=0.1)
    mb.try_acquire("key1")
    mb.try_acquire("key1")
    # key1 should be drained, key2 should be full
    assert mb.try_acquire("key1") is False
    assert mb.try_acquire("key2") is True


def test_multi_key_bucket_wait_time():
    mb = MultiKeyBucket(capacity=1, rate_per_second=0.5)
    mb.try_acquire("k")
    assert mb.wait_time("k") > 0.0


def test_multi_key_bucket_keys():
    mb = MultiKeyBucket(capacity=5, rate_per_second=1.0)
    mb.try_acquire("a")
    mb.try_acquire("b")
    assert "a" in mb.keys()
    assert "b" in mb.keys()


def test_refill_over_time():
    b = TokenBucket(capacity=2, rate_per_second=100.0)  # very fast refill
    b.try_acquire()
    b.try_acquire()
    time.sleep(0.02)  # wait for refill
    assert b.try_acquire() is True


def test_acquire_succeeds_when_available():
    b = TokenBucket(capacity=5, rate_per_second=1.0)
    b.acquire()  # should consume a token without blocking
    assert b.available_tokens == pytest.approx(4.0, abs=0.1)


def test_acquire_blocks_then_succeeds():
    b = TokenBucket(capacity=1, rate_per_second=100.0)  # fast refill
    b.try_acquire()  # drain
    start = time.monotonic()
    b.acquire()  # must wait for a token to refill, then succeed
    elapsed = time.monotonic() - start
    assert elapsed > 0.0


def test_multi_key_bucket_acquire_drains():
    mb = MultiKeyBucket(capacity=1, rate_per_second=0.1)
    mb.acquire("k")  # consumes the only token
    assert mb.try_acquire("k") is False


def test_multi_key_bucket_acquire_non_blocking_raises():
    mb = MultiKeyBucket(capacity=1, rate_per_second=0.1)
    mb.acquire("k")  # drain
    with pytest.raises(RateLimitExceeded):
        mb.acquire("k", block=False)
