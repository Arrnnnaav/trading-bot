# LLM Client + News Sentiment Redesign

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove Anthropic API key dependency by routing LLM calls through the local `claude` CLI subprocess, and replace the broken CryptoPanic news API with CryptoCompare (free, no key).

**Architecture:** A new `ClaudeCodeClient` in `core/llm_client.py` wraps async subprocess calls to the `claude` CLI with a semaphore cap of 3 concurrent processes. The debate engine adopts this client and runs its two haiku researcher calls in parallel via `asyncio.gather`. The news sentiment agent swaps its HTTP endpoint from CryptoPanic to CryptoCompare, mapping the pre-computed `SENTIMENT` field directly to scores.

**Tech Stack:** Python asyncio subprocesses, `asyncio.Semaphore`, `asyncio.create_subprocess_exec`, CryptoCompare Data API v1 (no auth).

---

## Files Changed

| File | Change |
|------|--------|
| `core/llm_client.py` | **CREATE** — ClaudeCodeClient with semaphore |
| `core/debate_engine.py` | **MODIFY** — use ClaudeCodeClient, parallel haiku calls |
| `harnesses/base_harness.py` | **MODIFY** — use ClaudeCodeClient in update_learnings |
| `agents/news_sentiment.py` | **MODIFY** — CryptoCompare endpoint, no API key |
| `config.py` | **MODIFY** — remove cryptopanic_api_key field |
| `harnesses/crypto_harness.py` | **MODIFY** — remove cryptopanic_api_key from agent init |
| `requirements.txt` | **MODIFY** — remove anthropic |
| `.env` | **MODIFY** — remove CRYPTOPANIC_API_KEY line |
| `tests/test_llm_client.py` | **CREATE** — unit tests |
| `tests/test_news_sentiment.py` | **CREATE** — unit tests |

---

## Design

### core/llm_client.py

```python
class ClaudeCodeClient:
    def __init__(self, max_concurrent: int = 3):
        self._sem = asyncio.Semaphore(max_concurrent)

    async def call(self, prompt: str, system: str = "", model: str = "claude-haiku-4-5-20251001") -> str:
        async with self._sem:
            args = ["claude", "-p", prompt, "--output-format", "text", "--model", model, "--bare"]
            if system:
                args += ["--system-prompt", system]
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip()
```

- Single public method `call(prompt, system, model) -> str`
- Semaphore ensures ≤3 concurrent claude processes system-wide
- Returns raw text — callers parse as needed (regex for adjudicator, raw text for researchers)
- If subprocess returns non-zero exit, returns empty string (callers fall back to initial values)

### core/debate_engine.py changes

Replace `Anthropic` client construction with `ClaudeCodeClient`. Make `_adversarial_debate` async and run the two researcher calls in parallel:

```python
# Before (sequential):
bull_arg = self._run_researcher("bull", ...)   # 2s
bear_arg = self._run_researcher("bear", ...)   # 2s

# After (parallel):
bull_arg, bear_arg = await asyncio.gather(
    self._run_researcher("bull", ...),
    self._run_researcher("bear", ...),
)  # 2s total
```

`reach_consensus`, `_adversarial_debate`, `_run_researcher`, `_adjudicate`, `generate_reasoning` all become `async def`. Callers (`base_harness.run_session`) already `await` the result.

### agents/news_sentiment.py changes

**Endpoint:** `https://data-api.cryptocompare.com/news/v1/article/list?limit=20&lang=EN&categories={ticker}`

**Sentiment mapping:**
- `POSITIVE` → +1
- `NEGATIVE` → -1
- `NEUTRAL` → 0

**Score calculation:** `avg_score = sum(scores) / len(articles)`

**Signal thresholds (unchanged):**
- `avg_score > 0.2` → LONG
- `avg_score < -0.2` → SHORT
- else → HOLD

**Ticker → category mapping:**
- `BTC/USDT` → `BTC`
- `ETH/USDT` → `ETH`
- `SOL/USDT` → `SOL`
- `BNB/USDT` → `BNB`
- `XRP/USDT` → `XRP`

No API key needed. Free tier: 100k requests/month. Bot makes ≤480 calls/day (5 tickers × 96 intervals — far under limit).

### config.py

Remove `cryptopanic_api_key` field entirely. No replacement field needed (CryptoCompare requires no key).

### requirements.txt

Remove `anthropic` line. No replacement needed (`claude` CLI already installed system-wide).

---

## Error Handling

- `ClaudeCodeClient.call()` timeout: wrap `proc.communicate()` in `asyncio.wait_for(..., timeout=30)` — if timeout, return `""` and caller uses fallback values
- CryptoCompare HTTP error (non-200 or exception): return `Direction.HOLD, confidence=0.0` with reasoning "News API unavailable" — same as current CryptoPanic error path
- Empty claude output: `_parse_adjudication` already has `fallback_dir/fallback_conf` — no change needed

---

## Testing

`tests/test_llm_client.py`:
- Semaphore limits concurrent calls (mock subprocesses, verify ≤3 at once)
- Returns stdout text on success
- Returns empty string on non-zero exit
- Timeout returns empty string

`tests/test_news_sentiment.py`:
- POSITIVE articles → LONG signal
- NEGATIVE articles → SHORT signal
- Mixed articles → HOLD
- API failure → HOLD with 0.0 confidence
- Ticker → category mapping correct
