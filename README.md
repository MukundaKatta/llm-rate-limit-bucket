# llm-rate-limit-bucket

Token-bucket rate limiter for LLM API calls. Supports blocking acquire, non-blocking try, and per-key buckets.

```python
from llm_rate_limit_bucket import TokenBucket, MultiKeyBucket

bucket = TokenBucket(capacity=10, rate_per_second=2.0, name="anthropic")
bucket.acquire()            # blocks until token available
bucket.try_acquire()        # returns False if not available
bucket.acquire(block=False) # raises RateLimitExceeded

@bucket.wrap()
def call_llm(messages): ...

# per-API-key limiting
mb = MultiKeyBucket(capacity=10, rate_per_second=2.0)
mb.acquire("user-123")
```
