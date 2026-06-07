# llm-rate-limit-bucket

A small, dependency-free **token-bucket rate limiter** for LLM API calls (and any
other rate-limited resource). Use it to stay *proactively* under a provider's
requests-per-second quota instead of reacting to `429` errors after the fact.

- **Token bucket algorithm** — a bucket holds up to `capacity` tokens and refills
  continuously at `rate_per_second`. Each call consumes one token (or `cost`).
- **Blocking and non-blocking** acquire, plus a `wait_time()` you can use to budget
  your own sleep.
- **Per-key limiting** via `MultiKeyBucket` — one independent bucket per API key,
  model, or provider endpoint.
- **Thread-safe** — every operation is guarded by a lock, so buckets can be shared
  across worker threads without over-subscribing the quota.
- **Zero runtime dependencies** — standard library only. Ships with type hints and a
  `py.typed` marker.

## Installation

```bash
pip install llm-rate-limit-bucket
```

Or install from source:

```bash
git clone https://github.com/MukundaKatta/llm-rate-limit-bucket
cd llm-rate-limit-bucket
pip install -e .
```

Requires Python 3.10+.

## Quickstart

```python
from llm_rate_limit_bucket import TokenBucket

# Allow bursts of up to 10 calls, refilling at 2 calls/second.
bucket = TokenBucket(capacity=10, rate_per_second=2.0, name="anthropic")

bucket.acquire()             # blocks until a token is available
ok = bucket.try_acquire()    # non-blocking: True if a token was consumed
secs = bucket.wait_time()    # seconds until the next token (0.0 if ready now)
```

### Fail fast instead of waiting

```python
from llm_rate_limit_bucket import TokenBucket, RateLimitExceeded

bucket = TokenBucket(capacity=5, rate_per_second=1.0, name="openai")

try:
    bucket.acquire(block=False)
    response = call_llm(...)
except RateLimitExceeded as exc:
    print(f"Slow down — retry after {exc.retry_after:.2f}s ({exc.bucket_name})")
```

### Decorate a function

```python
bucket = TokenBucket(capacity=10, rate_per_second=2.0)

@bucket.wrap()                      # acquires a token before each call
def call_llm(messages):
    ...
```

`wrap()` preserves the wrapped function's name and docstring via
`functools.wraps`.

### Weighted calls

Charge more than one token for an expensive call by passing `cost`:

```python
bucket.acquire(cost=3.0)       # consume 3 tokens
bucket.try_acquire(cost=2.0)   # non-blocking, consume 2
```

`cost` must be positive and no larger than `capacity` (a `cost` greater than
`capacity` could never succeed and raises `ValueError`).

### Per-key limiting

```python
from llm_rate_limit_bucket import MultiKeyBucket

# Each key gets its own bucket with the same capacity / rate.
mb = MultiKeyBucket(capacity=10, rate_per_second=2.0)

mb.acquire("user-123")             # bucket for "user-123"
mb.try_acquire("model:gpt-4o")     # independent bucket for the model
mb.wait_time("anthropic-prod")     # seconds until that key has a token

mb.get("user-123")                 # the underlying TokenBucket
"user-123" in mb                   # membership test
len(mb)                            # number of keys seen so far
mb.keys()                          # ["user-123", "model:gpt-4o", ...]
```

## API

### `TokenBucket(capacity, rate_per_second, name="default")`

| Member | Description |
| --- | --- |
| `try_acquire(cost=1.0) -> bool` | Consume `cost` tokens if available; never blocks. Returns whether tokens were consumed. |
| `acquire(cost=1.0, block=True) -> None` | Consume `cost` tokens. Blocks until available when `block=True`; otherwise raises `RateLimitExceeded`. |
| `wait_time(cost=1.0) -> float` | Seconds until `cost` tokens are available (0.0 if ready). Does not consume tokens. |
| `reset() -> None` | Refill the bucket to full capacity immediately. |
| `available_tokens -> float` | Current token count, including fractional refill. |
| `capacity -> float` | Maximum tokens the bucket holds. |
| `rate_per_second -> float` | Refill rate. |
| `name -> str` | Bucket label used in errors and `repr`. |
| `wrap(cost=1.0, block=True)` | Decorator that calls `acquire` before each invocation. |

Construction raises `ValueError` if `capacity` or `rate_per_second` is not positive.

### `MultiKeyBucket(capacity, rate_per_second)`

Keeps one `TokenBucket` per string key (created lazily on first use). Exposes
`acquire(key, ...)`, `try_acquire(key, ...)`, `wait_time(key, ...)`, `get(key)`,
`keys()`, plus `in` / `len()` support.

### `RateLimitExceeded(bucket_name, retry_after)`

Raised by `acquire(..., block=False)` when no tokens are available. Attributes:

- `bucket_name: str` — the bucket that rejected the request.
- `retry_after: float` — seconds to wait before retrying.

## How refill works

The bucket starts full. Tokens accrue continuously based on elapsed
`time.monotonic()` time (so it is unaffected by wall-clock adjustments) and are
capped at `capacity`. A drained bucket of `rate_per_second = r` recovers one token
every `1 / r` seconds.

## Development

Run the test suite with the standard library — no third-party dependencies needed:

```bash
python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
