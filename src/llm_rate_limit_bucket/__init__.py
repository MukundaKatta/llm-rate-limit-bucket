"""llm-rate-limit-bucket - token-bucket rate limiter for LLM API calls.

Classic token bucket sized for LLM API quotas. One bucket per API key
or per model endpoint. Refills continuously at a fixed rate, with a
configurable burst capacity. Sync and asyncio variants share the same
algorithm and zero runtime dependencies.

    from llm_rate_limit_bucket import TokenBucket, BucketRegistry

    bucket = TokenBucket(rate_per_sec=50.0, burst=100)
    if bucket.try_acquire():
        do_call()
    else:
        bucket.acquire(timeout=1.0)
        do_call()

    registry = BucketRegistry(rate_per_sec=10.0, burst=20)
    registry.get("anthropic-prod").acquire()
    registry.get("openai-prod").acquire()  # separate bucket per key

For async callers use `AsyncTokenBucket` / `AsyncBucketRegistry`; they
mirror the sync API but use `asyncio.sleep` so waiting acquires do not
block the event loop.

Use this PROACTIVELY to stay under a known provider rate limit. For
reactive backoff after a 429, pair with `llm-retry-py`. For total
token / USD spend caps, pair with `llm-budget-window`.
"""

from llm_rate_limit_bucket.bucket import AsyncTokenBucket, TokenBucket
from llm_rate_limit_bucket.registry import AsyncBucketRegistry, BucketRegistry

__version__ = "0.1.0"

__all__ = [
    "AsyncBucketRegistry",
    "AsyncTokenBucket",
    "BucketRegistry",
    "TokenBucket",
    "__version__",
]
