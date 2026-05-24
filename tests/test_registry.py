import threading

import pytest

from llm_rate_limit_bucket import (
    AsyncBucketRegistry,
    AsyncTokenBucket,
    BucketRegistry,
    TokenBucket,
)

# ---------- BucketRegistry: validation ----------


def test_registry_invalid_construction():
    with pytest.raises(ValueError):
        BucketRegistry(rate_per_sec=0, burst=5)
    with pytest.raises(ValueError):
        BucketRegistry(rate_per_sec=-1.0, burst=5)
    with pytest.raises(ValueError):
        BucketRegistry(rate_per_sec=1.0, burst=0)


# ---------- BucketRegistry: get / set ----------


def test_registry_same_key_returns_same_bucket():
    reg = BucketRegistry(rate_per_sec=10.0, burst=20)
    b1 = reg.get("anthropic-prod")
    b2 = reg.get("anthropic-prod")
    assert b1 is b2


def test_registry_different_keys_get_different_buckets():
    reg = BucketRegistry(rate_per_sec=10.0, burst=20)
    a = reg.get("anthropic-prod")
    o = reg.get("openai-prod")
    assert a is not o
    # Draining one does not drain the other.
    assert a.try_acquire(20) is True
    assert a.try_acquire() is False
    assert o.try_acquire() is True  # untouched


def test_registry_get_creates_bucket_with_defaults():
    reg = BucketRegistry(rate_per_sec=42.0, burst=7)
    b = reg.get("k")
    assert isinstance(b, TokenBucket)
    assert b.rate_per_sec == 42.0
    assert b.burst == 7


def test_registry_set_overrides_existing():
    reg = BucketRegistry(rate_per_sec=10.0, burst=20)
    original = reg.get("custom")
    override = TokenBucket(rate_per_sec=1.0, burst=2)
    reg.set("custom", override)
    assert reg.get("custom") is override
    assert reg.get("custom") is not original


def test_registry_set_inserts_new_key():
    reg = BucketRegistry(rate_per_sec=10.0, burst=20)
    custom = TokenBucket(rate_per_sec=1.0, burst=2)
    reg.set("brand-new", custom)
    assert reg.get("brand-new") is custom


def test_registry_set_rejects_wrong_type():
    reg = BucketRegistry(rate_per_sec=10.0, burst=20)
    with pytest.raises(TypeError):
        reg.set("k", "not a bucket")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        # AsyncTokenBucket is the wrong type for a sync registry.
        reg.set("k", AsyncTokenBucket(rate_per_sec=1.0, burst=1))  # type: ignore[arg-type]


# ---------- BucketRegistry: introspection ----------


def test_registry_contains_and_len():
    reg = BucketRegistry(rate_per_sec=1.0, burst=1)
    assert len(reg) == 0
    assert "x" not in reg
    reg.get("x")
    assert "x" in reg
    assert len(reg) == 1
    reg.get("y")
    assert len(reg) == 2


def test_registry_keys_snapshot():
    reg = BucketRegistry(rate_per_sec=1.0, burst=1)
    reg.get("a")
    reg.get("b")
    keys = reg.keys()
    assert set(keys) == {"a", "b"}


def test_registry_default_properties():
    reg = BucketRegistry(rate_per_sec=3.5, burst=9)
    assert reg.default_rate_per_sec == 3.5
    assert reg.default_burst == 9


# ---------- BucketRegistry: concurrency ----------


def test_registry_concurrent_get_same_key_returns_same_bucket():
    """100 threads racing on the same key all get the SAME bucket instance."""
    reg = BucketRegistry(rate_per_sec=1.0, burst=1)
    results: list[TokenBucket] = []
    lock = threading.Lock()
    barrier = threading.Barrier(100)

    def worker():
        barrier.wait()
        b = reg.get("hot-key")
        with lock:
            results.append(b)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 100
    first = results[0]
    assert all(r is first for r in results)
    assert len(reg) == 1


# ---------- AsyncBucketRegistry ----------


def test_async_registry_invalid_construction():
    with pytest.raises(ValueError):
        AsyncBucketRegistry(rate_per_sec=0, burst=5)
    with pytest.raises(ValueError):
        AsyncBucketRegistry(rate_per_sec=1.0, burst=0)


def test_async_registry_same_key_returns_same_bucket():
    reg = AsyncBucketRegistry(rate_per_sec=10.0, burst=20)
    b1 = reg.get("k")
    b2 = reg.get("k")
    assert b1 is b2
    assert isinstance(b1, AsyncTokenBucket)


def test_async_registry_different_keys_get_different_buckets():
    reg = AsyncBucketRegistry(rate_per_sec=10.0, burst=20)
    a = reg.get("a")
    b = reg.get("b")
    assert a is not b


def test_async_registry_set_overrides():
    reg = AsyncBucketRegistry(rate_per_sec=10.0, burst=20)
    override = AsyncTokenBucket(rate_per_sec=1.0, burst=1)
    reg.set("k", override)
    assert reg.get("k") is override


def test_async_registry_set_rejects_wrong_type():
    reg = AsyncBucketRegistry(rate_per_sec=10.0, burst=20)
    with pytest.raises(TypeError):
        reg.set("k", TokenBucket(rate_per_sec=1.0, burst=1))  # type: ignore[arg-type]


def test_async_registry_keys_and_len():
    reg = AsyncBucketRegistry(rate_per_sec=1.0, burst=1)
    assert len(reg) == 0
    reg.get("x")
    reg.get("y")
    assert len(reg) == 2
    assert set(reg.keys()) == {"x", "y"}
    assert "x" in reg
    assert "z" not in reg


async def test_async_registry_buckets_actually_work():
    """End-to-end: get a bucket from the async registry and acquire from it."""
    reg = AsyncBucketRegistry(rate_per_sec=10.0, burst=2)
    b = reg.get("anthropic")
    assert await b.try_acquire() is True
    assert await b.try_acquire() is True
    assert await b.try_acquire() is False
