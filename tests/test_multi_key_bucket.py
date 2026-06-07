"""Tests for :class:`MultiKeyBucket`, using the standard-library ``unittest``."""
import unittest

from llm_rate_limit_bucket import MultiKeyBucket, RateLimitExceeded, TokenBucket


class TestMultiKeyBucket(unittest.TestCase):
    def test_separate_buckets_per_key(self):
        mb = MultiKeyBucket(capacity=2, rate_per_second=0.001)
        mb.try_acquire("key1")
        mb.try_acquire("key1")
        # key1 is drained; key2 is untouched (full).
        self.assertFalse(mb.try_acquire("key1"))
        self.assertTrue(mb.try_acquire("key2"))

    def test_wait_time_per_key(self):
        mb = MultiKeyBucket(capacity=1, rate_per_second=0.5)
        mb.try_acquire("k")
        self.assertGreater(mb.wait_time("k"), 0.0)

    def test_keys_lists_created_buckets(self):
        mb = MultiKeyBucket(capacity=5, rate_per_second=1.0)
        mb.try_acquire("a")
        mb.try_acquire("b")
        self.assertIn("a", mb.keys())
        self.assertIn("b", mb.keys())

    def test_contains_and_len(self):
        mb = MultiKeyBucket(capacity=5, rate_per_second=1.0)
        self.assertEqual(len(mb), 0)
        self.assertNotIn("x", mb)
        mb.try_acquire("x")
        self.assertIn("x", mb)
        self.assertEqual(len(mb), 1)

    def test_get_returns_named_token_bucket(self):
        mb = MultiKeyBucket(capacity=5, rate_per_second=1.0)
        bucket = mb.get("provider-a")
        self.assertIsInstance(bucket, TokenBucket)
        self.assertEqual(bucket.name, "provider-a")
        self.assertEqual(bucket.capacity, 5)

    def test_get_is_idempotent(self):
        mb = MultiKeyBucket(capacity=5, rate_per_second=1.0)
        first = mb.get("same")
        second = mb.get("same")
        self.assertIs(first, second)

    def test_acquire_blocking_path(self):
        # block=False on a drained key raises RateLimitExceeded.
        mb = MultiKeyBucket(capacity=1, rate_per_second=0.001)
        mb.acquire("k")  # consumes the only token
        with self.assertRaises(RateLimitExceeded):
            mb.acquire("k", block=False)

    def test_invalid_construction(self):
        with self.assertRaises(ValueError):
            MultiKeyBucket(capacity=0, rate_per_second=1.0)
        with self.assertRaises(ValueError):
            MultiKeyBucket(capacity=5, rate_per_second=0)


if __name__ == "__main__":
    unittest.main()
