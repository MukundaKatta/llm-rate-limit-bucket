"""Tests for :class:`TokenBucket`, using the standard-library ``unittest``."""
import threading
import time
import unittest

from llm_rate_limit_bucket import RateLimitExceeded, TokenBucket, __version__


class TestConstruction(unittest.TestCase):
    def test_starts_full(self):
        b = TokenBucket(capacity=10, rate_per_second=1.0)
        self.assertAlmostEqual(b.available_tokens, 10.0, delta=0.1)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            TokenBucket(capacity=0, rate_per_second=1.0)
        with self.assertRaises(ValueError):
            TokenBucket(capacity=-1, rate_per_second=1.0)

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            TokenBucket(capacity=5, rate_per_second=0)
        with self.assertRaises(ValueError):
            TokenBucket(capacity=5, rate_per_second=-2.0)

    def test_properties(self):
        b = TokenBucket(capacity=20, rate_per_second=3.0, name="api")
        self.assertEqual(b.capacity, 20)
        self.assertEqual(b.rate_per_second, 3.0)
        self.assertEqual(b.name, "api")

    def test_repr_contains_name(self):
        b = TokenBucket(capacity=5, rate_per_second=1.0, name="myapi")
        self.assertIn("myapi", repr(b))

    def test_version_is_string(self):
        self.assertIsInstance(__version__, str)


class TestTryAcquire(unittest.TestCase):
    def test_success(self):
        b = TokenBucket(capacity=5, rate_per_second=1.0)
        self.assertTrue(b.try_acquire())

    def test_exhausted_returns_false(self):
        b = TokenBucket(capacity=2, rate_per_second=0.001)
        self.assertTrue(b.try_acquire())
        self.assertTrue(b.try_acquire())
        self.assertFalse(b.try_acquire())

    def test_tokens_decrease(self):
        b = TokenBucket(capacity=10, rate_per_second=0.001)
        b.try_acquire(cost=3.0)
        self.assertAlmostEqual(b.available_tokens, 7.0, delta=0.1)

    def test_failed_acquire_does_not_decrement(self):
        b = TokenBucket(capacity=1, rate_per_second=0.001)
        b.try_acquire()
        before = b.available_tokens
        self.assertFalse(b.try_acquire())
        self.assertAlmostEqual(b.available_tokens, before, delta=0.05)

    def test_cost_must_be_positive(self):
        b = TokenBucket(capacity=5, rate_per_second=1.0)
        with self.assertRaises(ValueError):
            b.try_acquire(cost=0)
        with self.assertRaises(ValueError):
            b.try_acquire(cost=-1.0)

    def test_cost_above_capacity_raises(self):
        b = TokenBucket(capacity=5, rate_per_second=1.0)
        with self.assertRaises(ValueError):
            b.try_acquire(cost=6.0)


class TestAcquire(unittest.TestCase):
    def test_non_blocking_raises_when_drained(self):
        b = TokenBucket(capacity=1, rate_per_second=0.001)
        b.try_acquire()
        with self.assertRaises(RateLimitExceeded) as ctx:
            b.acquire(block=False)
        self.assertGreater(ctx.exception.retry_after, 0)

    def test_rate_limit_exceeded_has_bucket_name(self):
        b = TokenBucket(capacity=1, rate_per_second=0.001, name="myapi")
        b.try_acquire()
        with self.assertRaises(RateLimitExceeded) as ctx:
            b.acquire(block=False)
        self.assertEqual(ctx.exception.bucket_name, "myapi")

    def test_blocking_succeeds_after_refill(self):
        b = TokenBucket(capacity=1, rate_per_second=100.0)
        b.try_acquire()  # drain
        start = time.monotonic()
        b.acquire()  # must wait ~0.01s then succeed
        elapsed = time.monotonic() - start
        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 1.0)

    def test_acquire_when_available_does_not_block(self):
        b = TokenBucket(capacity=5, rate_per_second=1.0)
        start = time.monotonic()
        b.acquire()
        self.assertLess(time.monotonic() - start, 0.1)

    def test_cost_above_capacity_raises(self):
        b = TokenBucket(capacity=2, rate_per_second=1.0)
        with self.assertRaises(ValueError):
            b.acquire(cost=3.0)


class TestWaitTime(unittest.TestCase):
    def test_zero_when_available(self):
        b = TokenBucket(capacity=5, rate_per_second=1.0)
        self.assertEqual(b.wait_time(), 0.0)

    def test_positive_when_drained(self):
        b = TokenBucket(capacity=1, rate_per_second=0.5)
        b.try_acquire()
        self.assertGreater(b.wait_time(), 0.0)

    def test_does_not_consume(self):
        b = TokenBucket(capacity=5, rate_per_second=0.001)
        before = b.available_tokens
        b.wait_time()
        self.assertAlmostEqual(b.available_tokens, before, delta=0.05)


class TestRefillAndReset(unittest.TestCase):
    def test_refill_over_time(self):
        b = TokenBucket(capacity=2, rate_per_second=100.0)
        b.try_acquire()
        b.try_acquire()
        time.sleep(0.05)
        self.assertTrue(b.try_acquire())

    def test_refill_caps_at_capacity(self):
        b = TokenBucket(capacity=3, rate_per_second=1000.0)
        b.try_acquire()
        time.sleep(0.05)
        self.assertLessEqual(b.available_tokens, 3.0)

    def test_reset_refills_to_full(self):
        b = TokenBucket(capacity=4, rate_per_second=0.001)
        b.try_acquire(cost=4.0)
        self.assertFalse(b.try_acquire())
        b.reset()
        self.assertAlmostEqual(b.available_tokens, 4.0, delta=0.05)


class TestWrap(unittest.TestCase):
    def test_decorator_calls_through(self):
        b = TokenBucket(capacity=5, rate_per_second=0.001)
        calls = []

        @b.wrap()
        def do_thing():
            calls.append(1)
            return "ok"

        self.assertEqual(do_thing(), "ok")
        self.assertEqual(len(calls), 1)
        self.assertLess(b.available_tokens, 5.0)

    def test_decorator_preserves_metadata(self):
        b = TokenBucket(capacity=5, rate_per_second=1.0)

        @b.wrap()
        def documented():
            """A docstring."""

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "A docstring.")

    def test_decorator_forwards_args(self):
        b = TokenBucket(capacity=5, rate_per_second=1.0)

        @b.wrap()
        def add(a, b_=0):
            return a + b_

        self.assertEqual(add(2, b_=3), 5)


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_try_acquire_does_not_oversubscribe(self):
        # Many threads racing on a slow-refilling bucket must not consume
        # more tokens than exist. Without locking this would over-subscribe.
        capacity = 50
        b = TokenBucket(capacity=capacity, rate_per_second=0.001)
        successes = []
        lock = threading.Lock()

        def worker():
            if b.try_acquire():
                with lock:
                    successes.append(1)

        threads = [threading.Thread(target=worker) for _ in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(successes), capacity)


if __name__ == "__main__":
    unittest.main()
