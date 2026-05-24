# llm-rate-limit-bucket

[![PyPI](https://img.shields.io/pypi/v/llm-rate-limit-bucket.svg)](https://pypi.org/project/llm-rate-limit-bucket/)
[![Python](https://img.shields.io/pypi/pyversions/llm-rate-limit-bucket.svg)](https://pypi.org/project/llm-rate-limit-bucket/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Token-bucket rate limiter for LLM API calls.**

Classic token bucket sized for LLM API quotas. One bucket per API key
or per model endpoint. Refills continuously at a fixed rate, with a
configurable burst capacity. Sync and asyncio variants share the same
algorithm and zero runtime dependencies.

Use this when you want to stay UNDER a provider rate limit instead of
catching 429s after the fact.

## Install

```bash
pip install llm-rate-limit-bucket
```

## Sync use

```python
from llm_rate_limit_bucket import TokenBucket

# 50 requests per second, burst of 100
bucket = TokenBucket(rate_per_sec=50.0, burst=100)

if bucket.try_acquire():
    do_call()
else:
    # Wait up to 1 second for a slot
    if bucket.acquire(timeout=1.0):
        do_call()
    else:
        # Still no slot, skip or queue
        ...
```

`try_acquire(n)` returns `True` only if `n` tokens are available right
now (and decrements). `acquire(n, timeout)` blocks the calling thread
until tokens are available, up to the timeout. `wait_seconds(n)` tells
you how long until `n` tokens would be ready (`0.0` if available now).

## Async use

```python
import asyncio
from llm_rate_limit_bucket import AsyncTokenBucket

async def main():
    bucket = AsyncTokenBucket(rate_per_sec=50.0, burst=100)
    await bucket.acquire()           # waits with asyncio.sleep
    await do_async_call()

asyncio.run(main())
```

`AsyncTokenBucket` mirrors the sync API: `try_acquire`, `acquire`,
`wait_seconds`, `available`. The async `acquire` uses `asyncio.sleep`,
so it does not block the event loop.

## Per-key registry

Different keys (or providers, or model endpoints) usually have separate
quotas. Wrap them with `BucketRegistry`:

```python
from llm_rate_limit_bucket import BucketRegistry

registry = BucketRegistry(rate_per_sec=10.0, burst=20)
registry.get("anthropic-prod").acquire()
registry.get("openai-prod").acquire()   # separate bucket, separate quota
```

`.get(key)` returns the bucket for that key, creating one with the
default `rate_per_sec` and `burst` on first access. Override per key:

```python
registry.set("anthropic-prod", TokenBucket(rate_per_sec=5.0, burst=10))
```

There is also an `AsyncBucketRegistry` with the same API that creates
`AsyncTokenBucket` instances.

## When to use this vs `llm-budget-window`

`llm-rate-limit-bucket` is PROACTIVE. It blocks (or returns False) when
you are about to exceed a `requests per second` quota, so you never send
the request that would trigger a 429. Useful for known per-key rate
limits.

[`llm-budget-window`](https://crates.io/crates/llm-budget-window) is a
sliding-window TOKEN / USD cap. It tells you whether you have spent too
much in the last N minutes / hours / days. Use it for cost or quota
ceilings measured in tokens or dollars, not request rate.

You usually want both layered: rate-limit-bucket gates the request
cadence, budget-window gates total spend, and
[`llm-retry-py`](https://pypi.org/project/llm-retry-py/) handles the
backoff path when a 429 slips through anyway.

## What it does NOT do

- No HTTP. It does not call any LLM provider.
- No persistence. State lives in process. For multi-process or
  multi-host rate limiting, back it with Redis (`INCR` + `EXPIRE`) or
  a managed limiter.
- No 429 parsing. If the provider sends a `Retry-After`, pair this with
  `llm-retry-py` to honor it.
- No dynamic rate. You set `rate_per_sec` at construction. If your
  quota changes, build a new bucket.

## License

MIT
