# LLM Client + News Sentiment Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Anthropic SDK dependency by routing all LLM calls through the local `claude` CLI subprocess, and replace the broken CryptoPanic news API with CryptoCompare (free, no key required).

**Architecture:** A new `ClaudeCodeClient` wraps `asyncio.create_subprocess_exec("claude", ...)` behind an `asyncio.Semaphore(3)` to cap concurrent processes. The debate engine adopts it and runs its two haiku researcher calls in parallel via `asyncio.gather`. The news sentiment agent swaps its HTTP endpoint to CryptoCompare and maps the pre-computed `SENTIMENT` field to scores. The `anthropic` package and `CRYPTOPANIC_API_KEY` config field are removed entirely.

**Tech Stack:** Python `asyncio`, `asyncio.create_subprocess_exec`, `asyncio.Semaphore`, `httpx`, CryptoCompare Data API v1 (no auth required).

---

## File Map

| File | Action |
|------|--------|
| `core/llm_client.py` | CREATE — `ClaudeCodeClient` with semaphore |
| `core/debate_engine.py` | MODIFY — swap Anthropic client, make methods async, parallel haiku |
| `harnesses/base_harness.py` | MODIFY — async `update_learnings`, use `ClaudeCodeClient` |
| `agents/news_sentiment.py` | MODIFY — CryptoCompare endpoint, no API key |
| `harnesses/crypto_harness.py` | MODIFY — remove `cryptopanic_api_key` from `NewsSentimentAgent()` |
| `config.py` | MODIFY — remove `cryptopanic_api_key` field |
| `requirements.txt` | MODIFY — remove `anthropic` |
| `.env` | MODIFY — remove `CRYPTOPANIC_API_KEY` line |
| `tests/test_llm_client.py` | CREATE — unit tests for `ClaudeCodeClient` |
| `tests/test_news_sentiment.py` | CREATE — unit tests for `NewsSentimentAgent` |

---

### Task 1: Create ClaudeCodeClient

**Files:**
- Create: `core/llm_client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_client.py
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_call_returns_stdout():
    from core.llm_client import ClaudeCodeClient
    client = ClaudeCodeClient()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"LONG signal confirmed.\n", b""))
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await client.call("test prompt", model="claude-haiku-4-5-20251001")

    assert result == "LONG signal confirmed."


@pytest.mark.asyncio
async def test_call_returns_empty_on_nonzero_exit():
    from core.llm_client import ClaudeCodeClient
    client = ClaudeCodeClient()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await client.call("test prompt")

    assert result == ""


@pytest.mark.asyncio
async def test_call_returns_empty_on_timeout():
    from core.llm_client import ClaudeCodeClient
    client = ClaudeCodeClient()

    async def slow_communicate():
        await asyncio.sleep(999)
        return (b"", b"")

    mock_proc = MagicMock()
    mock_proc.communicate = slow_communicate
    mock_proc.kill = MagicMock()

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await client.call("test prompt", timeout=0.01)

    assert result == ""


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    from core.llm_client import ClaudeCodeClient
    client = ClaudeCodeClient(max_concurrent=2)

    active = []
    peak = []

    async def fake_communicate():
        active.append(1)
        peak.append(len(active))
        await asyncio.sleep(0.05)
        active.pop()
        return (b"ok", b"")

    mock_proc = MagicMock()
    mock_proc.communicate = fake_communicate
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        await asyncio.gather(*[client.call(f"prompt {i}") for i in range(5)])

    assert max(peak) <= 2


@pytest.mark.asyncio
async def test_system_prompt_passed_as_flag():
    from core.llm_client import ClaudeCodeClient
    client = ClaudeCodeClient()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"result", b""))
    mock_proc.returncode = 0

    captured_args = []

    async def capture(*args, **kwargs):
        captured_args.extend(args)
        return mock_proc

    with patch("asyncio.create_subprocess_exec", capture):
        await client.call("my prompt", system="you are a trader")

    assert "--system-prompt" in captured_args
    assert "you are a trader" in captured_args
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_llm_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.llm_client'`

- [ ] **Step 3: Implement ClaudeCodeClient**

```python
# core/llm_client.py
import asyncio


class ClaudeCodeClient:
    def __init__(self, max_concurrent: int = 3):
        self._sem = asyncio.Semaphore(max_concurrent)

    async def call(
        self,
        prompt: str,
        system: str = "",
        model: str = "claude-haiku-4-5-20251001",
        timeout: float = 30.0,
    ) -> str:
        args = [
            "claude", "-p", prompt,
            "--output-format", "text",
            "--model", model,
            "--bare",
        ]
        if system:
            args += ["--system-prompt", system]

        async with self._sem:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                if proc.returncode != 0:
                    return ""
                return stdout.decode().strip()
            except (asyncio.TimeoutError, Exception):
                try:
                    proc.kill()
                except Exception:
                    pass
                return ""
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_llm_client.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add core/llm_client.py tests/test_llm_client.py
git commit -m "feat: add ClaudeCodeClient subprocess wrapper with semaphore"
```

---

### Task 2: Rewrite NewsSentimentAgent (CryptoCompare)

**Files:**
- Modify: `agents/news_sentiment.py`
- Create: `tests/test_news_sentiment.py`

CryptoCompare endpoint: `https://data-api.cryptocompare.com/news/v1/article/list?limit=20&lang=EN&categories={currency}`

Response shape per article:
```json
{"SENTIMENT": "POSITIVE"|"NEGATIVE"|"NEUTRAL", "TITLE": "...", ...}
```

Ticker → currency: strip `/USDT` suffix. `BTC/USDT` → `BTC`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_news_sentiment.py
import pytest
from unittest.mock import patch, MagicMock
from core.models import Direction, Market


def make_articles(sentiments: list[str]) -> dict:
    return {
        "Data": [
            {"SENTIMENT": s, "TITLE": f"Article about {s}"}
            for s in sentiments
        ]
    }


def test_positive_articles_return_long():
    from agents.news_sentiment import NewsSentimentAgent
    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = make_articles(["POSITIVE"] * 8 + ["NEUTRAL"] * 2)
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.LONG
    assert vote.confidence > 0.5


def test_negative_articles_return_short():
    from agents.news_sentiment import NewsSentimentAgent
    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = make_articles(["NEGATIVE"] * 8 + ["NEUTRAL"] * 2)
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.SHORT
    assert vote.confidence > 0.5


def test_mixed_articles_return_hold():
    from agents.news_sentiment import NewsSentimentAgent
    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = make_articles(["POSITIVE"] * 3 + ["NEGATIVE"] * 3 + ["NEUTRAL"] * 4)
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0


def test_api_failure_returns_hold():
    from agents.news_sentiment import NewsSentimentAgent
    agent = NewsSentimentAgent()

    with patch("httpx.get", side_effect=Exception("connection error")):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0
    assert "failed" in vote.reasoning.lower()


def test_empty_articles_return_hold():
    from agents.news_sentiment import NewsSentimentAgent
    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"Data": []}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.HOLD


def test_ticker_to_currency_mapping():
    from agents.news_sentiment import NewsSentimentAgent
    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"Data": []}
    mock_resp.raise_for_status = MagicMock()

    captured_params = {}

    def capture_get(url, params=None, timeout=None):
        captured_params.update(params or {})
        return mock_resp

    with patch("httpx.get", side_effect=capture_get):
        agent.analyze("ETH/USDT", [], Market.CRYPTO)

    assert captured_params.get("categories") == "ETH"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_news_sentiment.py -v
```

Expected: failures — old code uses `api_key` constructor arg and CryptoPanic URL

- [ ] **Step 3: Rewrite NewsSentimentAgent**

```python
# agents/news_sentiment.py
import json
import httpx
from pathlib import Path
from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

_SENTIMENT_MAP = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}


class NewsSentimentAgent(BaseAgent):
    name = "NewsSentiment"
    BASE_URL = "https://data-api.cryptocompare.com/news/v1/article/list"

    def __init__(self):
        thresholds = self._load_thresholds()
        self._sentiment_threshold = thresholds.get("news_sentiment_threshold", 0.2)

    def _load_thresholds(self) -> dict:
        try:
            return json.loads(Path("data/calibrated_thresholds.json").read_text())
        except Exception:
            return {}

    def _fetch_news(self, ticker: str) -> list[dict]:
        currency = ticker.split("/")[0].upper()
        resp = httpx.get(
            self.BASE_URL,
            params={"limit": 20, "lang": "EN", "categories": currency},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("Data", [])

    def _score_articles(self, articles: list[dict]) -> float:
        if not articles:
            return 0.0
        scores = [_SENTIMENT_MAP.get(a.get("SENTIMENT", "NEUTRAL"), 0.0) for a in articles[:10]]
        return sum(scores) / len(scores)

    def analyze(self, ticker: str, klines: list[dict], market: Market, **kwargs) -> AgentVote:
        try:
            articles = self._fetch_news(ticker)
            score = self._score_articles(articles)
        except Exception:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="News fetch failed",
            )

        if score > self._sentiment_threshold:
            direction, confidence = Direction.LONG, min(0.5 + score, 0.85)
        elif score < -self._sentiment_threshold:
            direction, confidence = Direction.SHORT, min(0.5 + abs(score), 0.85)
        else:
            direction, confidence = Direction.HOLD, 0.0

        headlines = [a.get("TITLE", "")[:60] for a in articles[:3]]
        reasoning = f"Sentiment score: {score:.2f}. Top: {' | '.join(headlines)}"
        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_news_sentiment.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add agents/news_sentiment.py tests/test_news_sentiment.py
git commit -m "feat: replace CryptoPanic with CryptoCompare in NewsSentimentAgent"
```

---

### Task 3: Remove cryptopanic_api_key from config and harness

**Files:**
- Modify: `config.py` (line 23 — remove `cryptopanic_api_key`)
- Modify: `harnesses/crypto_harness.py` (line 23 — `NewsSentimentAgent()` call)

- [ ] **Step 1: Remove `cryptopanic_api_key` from config.py**

Open `config.py`. Remove this line:
```python
cryptopanic_api_key: str = os.environ.get("CRYPTOPANIC_API_KEY", "")
```

- [ ] **Step 2: Remove api_key arg from NewsSentimentAgent in crypto_harness.py**

In `harnesses/crypto_harness.py`, change:
```python
NewsSentimentAgent(config.cryptopanic_api_key),
```
to:
```python
NewsSentimentAgent(),
```

- [ ] **Step 3: Remove CRYPTOPANIC_API_KEY from .env**

Open `.env`. Remove the line:
```
CRYPTOPANIC_API_KEY=FILL_THIS
```

- [ ] **Step 4: Run tests to verify nothing broke**

```
pytest tests/ -q
```

Expected: all previously passing tests still pass

- [ ] **Step 5: Commit**

```bash
git add config.py harnesses/crypto_harness.py .env
git commit -m "chore: remove cryptopanic_api_key — no longer needed"
```

---

### Task 4: Rewrite DebateEngine to use ClaudeCodeClient

**Files:**
- Modify: `core/debate_engine.py`

The key changes:
1. Replace `Anthropic` client with `ClaudeCodeClient`
2. Make `_adversarial_debate`, `_run_researcher`, `_adjudicate`, `generate_reasoning` all `async def`
3. Run bull+bear haiku calls in parallel with `asyncio.gather`
4. `reach_consensus` calls `_adversarial_debate` — must become `async` too

Note: `reach_consensus` is called from `base_harness.run_session` which already `await`s it.

- [ ] **Step 1: Write failing test for async debate engine**

Add to `tests/test_debate_engine.py` (file already exists — append these tests):

```python
@pytest.mark.asyncio
async def test_reach_consensus_calls_adversarial_debate():
    """reach_consensus should return a dict with direction/confidence/transcript."""
    from core.debate_engine import DebateEngine
    from core.models import AgentVote, Direction

    engine = DebateEngine.__new__(DebateEngine)
    engine.learnings = ""

    async def fake_debate(votes, ticker, direction, confidence):
        return {"direction": direction, "confidence": confidence, "transcript": "fake"}

    engine._adversarial_debate = fake_debate

    votes = [
        AgentVote(agent_name="A", direction=Direction.LONG, confidence=0.80, reasoning="r1"),
        AgentVote(agent_name="B", direction=Direction.LONG, confidence=0.75, reasoning="r2"),
    ]
    result = await engine.reach_consensus(votes, ticker="BTC/USDT")
    assert result["direction"] == Direction.LONG


@pytest.mark.asyncio
async def test_adversarial_debate_runs_researchers_in_parallel():
    """Bull and bear calls should be issued concurrently."""
    import asyncio
    from core.debate_engine import DebateEngine
    from core.models import Direction

    engine = DebateEngine.__new__(DebateEngine)
    engine.learnings = ""

    call_times = []

    async def fake_researcher(role, transcript, ticker, direction):
        call_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.05)
        return f"{role} argument"

    async def fake_adjudicate(bull, bear, transcript, ticker, dir_, conf):
        return {"direction": dir_, "confidence": conf, "transcript": "t"}

    engine._run_researcher = fake_researcher
    engine._adjudicate = fake_adjudicate

    from core.models import AgentVote
    votes = [AgentVote(agent_name="X", direction=Direction.LONG, confidence=0.8, reasoning="r")]
    await engine._adversarial_debate(votes, "BTC/USDT", Direction.LONG, 0.75)

    assert len(call_times) == 2
    # Both calls started within 10ms of each other (parallel, not sequential)
    assert abs(call_times[1] - call_times[0]) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_debate_engine.py::test_reach_consensus_calls_adversarial_debate tests/test_debate_engine.py::test_adversarial_debate_runs_researchers_in_parallel -v
```

Expected: FAIL — methods are not yet async

- [ ] **Step 3: Rewrite debate_engine.py**

```python
# core/debate_engine.py
"""
DebateEngine — multi-agent vote consensus + adversarial bull/bear debate.

Flow:
  1. reach_consensus(votes, ticker) — vote counting + threshold check
  2. If direction != HOLD: _adversarial_debate() runs
     a. Bull researcher (haiku)  — argues FOR the trade  } parallel
     b. Bear researcher (haiku)  — argues AGAINST the trade }
     c. Opus adjudicator         — makes final LONG/SHORT/HOLD + confidence
  3. generate_reasoning()        — Telegram-facing 2-sentence rationale (haiku)
"""

import asyncio
import re
from core.llm_client import ClaudeCodeClient
from core.models import AgentVote, Direction
from config import config

_CLIENT = ClaudeCodeClient(max_concurrent=3)


class DebateEngine:
    def __init__(self, learnings: str = ""):
        self.learnings = learnings

    def update_learnings(self, learnings: str):
        self.learnings = learnings

    # ── Public API ────────────────────────────────────────────────────────────

    async def reach_consensus(
        self,
        votes: list[AgentVote],
        ticker: str = "",
        agent_weights: dict[str, float] | None = None,
    ) -> dict:
        if not votes:
            return {"direction": Direction.HOLD, "confidence": 0.0, "transcript": "No votes"}

        active = [v for v in votes if v.direction != Direction.HOLD and v.confidence >= 0.50]
        long_v = [v for v in active if v.direction == Direction.LONG]
        short_v = [v for v in active if v.direction == Direction.SHORT]

        if len(long_v) >= 2 and len(long_v) > len(short_v):
            direction, agreeing = Direction.LONG, long_v
        elif len(short_v) >= 2 and len(short_v) > len(long_v):
            direction, agreeing = Direction.SHORT, short_v
        else:
            return {"direction": Direction.HOLD, "confidence": 0.0, "transcript": self._build_transcript(votes)}

        if agent_weights:
            weights = [agent_weights.get(v.agent_name, 1.0) for v in agreeing]
            confidence = sum(w * v.confidence for w, v in zip(weights, agreeing)) / sum(weights)
        else:
            confidence = sum(v.confidence for v in agreeing) / len(agreeing)

        if confidence < config.min_confidence:
            return {"direction": Direction.HOLD, "confidence": confidence, "transcript": self._build_transcript(votes)}

        return await self._adversarial_debate(votes, ticker, direction, round(confidence, 3))

    async def generate_reasoning(self, votes: list[AgentVote], direction: Direction, ticker: str) -> str:
        transcript = self._build_transcript(votes)
        prompt = (
            f"Agent debate for {ticker}:\n{transcript}\n\n"
            f"Consensus: {direction.value}\n"
            "Write a 2-sentence trading rationale. Be specific about key signals."
        )
        result = await _CLIENT.call(prompt, system=self._build_system_prompt())
        return result or f"{direction.value} signal on {ticker} based on agent consensus."

    # ── Adversarial debate ────────────────────────────────────────────────────

    async def _adversarial_debate(
        self,
        votes: list[AgentVote],
        ticker: str,
        initial_direction: Direction,
        initial_confidence: float,
    ) -> dict:
        transcript = self._build_transcript(votes)
        bull_arg, bear_arg = await asyncio.gather(
            self._run_researcher("bull", transcript, ticker, initial_direction),
            self._run_researcher("bear", transcript, ticker, initial_direction),
        )
        return await self._adjudicate(bull_arg, bear_arg, transcript, ticker, initial_direction, initial_confidence)

    async def _run_researcher(self, role: str, transcript: str, ticker: str, direction: Direction) -> str:
        stance = "FOR" if role == "bull" else "AGAINST"
        prompt = (
            f"You are a {role}ish researcher. Make the strongest 2-sentence case "
            f"{stance} a {direction.value} trade on {ticker}.\n\n"
            f"Agent signals:\n{transcript}\n\nBe specific, not generic."
        )
        result = await _CLIENT.call(prompt, system=self._build_system_prompt())
        return result or f"{role} argument unavailable"

    async def _adjudicate(
        self,
        bull_arg: str,
        bear_arg: str,
        transcript: str,
        ticker: str,
        initial_direction: Direction,
        initial_confidence: float,
    ) -> dict:
        prompt = (
            f"You are the senior portfolio manager making the final trading decision for {ticker}.\n\n"
            f"Initial consensus: {initial_direction.value} (confidence: {initial_confidence:.2f})\n\n"
            f"Bull argument:\n{bull_arg}\n\n"
            f"Bear argument:\n{bear_arg}\n\n"
            f"Agent votes:\n{transcript}\n\n"
            "Respond in EXACTLY this format (no extra text):\n"
            "DECISION: LONG|SHORT|HOLD\n"
            "CONFIDENCE: 0.50-0.95\n"
            "REASON: one sentence"
        )
        text = await _CLIENT.call(prompt, system=self._build_system_prompt(), model="claude-opus-4-7")
        direction, confidence = self._parse_adjudication(text, initial_direction, initial_confidence)
        full_transcript = (
            f"{transcript}\n\n[Bull] {bull_arg}\n[Bear] {bear_arg}\n[Adjudicator] {text}"
        )
        return {"direction": direction, "confidence": confidence, "transcript": full_transcript}

    @staticmethod
    def _parse_adjudication(text: str, fallback_dir: Direction, fallback_conf: float) -> tuple[Direction, float]:
        dir_match = re.search(r"DECISION:\s*(LONG|SHORT|HOLD)", text, re.IGNORECASE)
        conf_match = re.search(r"CONFIDENCE:\s*(0\.\d+)", text)
        direction = Direction(dir_match.group(1).upper()) if dir_match else fallback_dir
        confidence = float(conf_match.group(1)) if conf_match else fallback_conf
        return direction, round(min(max(confidence, 0.50), 0.95), 3)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_transcript(self, votes: list[AgentVote]) -> str:
        return "\n".join(
            f"{v.agent_name}: {v.direction.value} (conf={v.confidence:.2f}) — {v.reasoning}"
            for v in votes
        )

    def _build_system_prompt(self) -> str:
        base = "You are a crypto trading analyst."
        if self.learnings:
            return f"{base}\n\nRecent learnings from signal outcomes:\n{self.learnings}"
        return base
```

- [ ] **Step 4: Update `base_harness.py` — await `reach_consensus` and `generate_reasoning`**

In `harnesses/base_harness.py`, `run_session` already awaits `reach_consensus`. Verify line 53 reads:
```python
consensus = await self.debate_engine.reach_consensus(
    votes, ticker=ticker, agent_weights=agent_weights
)
```

Also update line 67 to await `generate_reasoning`:
```python
reasoning = await self.debate_engine.generate_reasoning(votes, direction, ticker)
```

- [ ] **Step 5: Run new debate engine tests to verify they pass**

```
pytest tests/test_debate_engine.py -v
```

Expected: all passing (including the 2 new async tests)

- [ ] **Step 6: Commit**

```bash
git add core/debate_engine.py harnesses/base_harness.py
git commit -m "feat: rewrite DebateEngine to use ClaudeCodeClient with parallel haiku calls"
```

---

### Task 5: Rewrite update_learnings in base_harness.py

**Files:**
- Modify: `harnesses/base_harness.py`

`update_learnings` currently uses `Anthropic` client directly. Replace with `ClaudeCodeClient`.
Also make it `async def` since it now awaits a subprocess call. The caller in `crypto_harness.py` calls `self.update_learnings()` — update to `await self.update_learnings()`.

- [ ] **Step 1: Rewrite `update_learnings` in `base_harness.py`**

Remove the `from anthropic import Anthropic` import at the top of `base_harness.py`.

Replace `update_learnings`:
```python
async def update_learnings(self):
    if self.state.session_count % 10 != 0:
        return
    recent_ids = self.state.signal_history[-50:]
    all_signals = {s["id"]: s for s in self.aggregator.get_all_signals()}
    outcomes = [all_signals[id] for id in recent_ids if id in all_signals]
    if not outcomes:
        return

    from core.llm_client import ClaudeCodeClient
    client = ClaudeCodeClient(max_concurrent=1)
    summary_prompt = (
        f"Analyze these {len(outcomes)} trading signal outcomes:\n"
        f"{json.dumps([{'ticker': o['ticker'], 'direction': o['direction'], 'outcome': o['outcome'], 'pnl_pct': o.get('hypothetical_pnl_pct')} for o in outcomes], indent=2)}\n\n"
        "Write 3 concise sentences about: (1) which setups worked, (2) which failed, (3) one rule to apply next session."
    )
    learnings = await client.call(summary_prompt)
    if not learnings:
        return

    self.state.agent_learnings = learnings
    self._save_state()
    self.debate_engine.update_learnings(learnings)

    from pathlib import Path
    log_path = Path("data") / f"{self.MARKET.value}_learnings.md"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"\n## {ts}\n{learnings}\n")
```

- [ ] **Step 2: Update callers of `update_learnings`**

In `harnesses/crypto_harness.py` (`_run_all_tickers`), change:
```python
self.update_learnings()
```
to:
```python
await self.update_learnings()
```

In `harnesses/india_harness.py` (`_run_all_tickers`), same change.

- [ ] **Step 3: Remove Anthropic import from base_harness.py**

Remove from top of `harnesses/base_harness.py`:
```python
from anthropic import Anthropic
```

- [ ] **Step 4: Run full test suite**

```
pytest tests/ -q
```

Expected: all passing

- [ ] **Step 5: Commit**

```bash
git add harnesses/base_harness.py harnesses/crypto_harness.py harnesses/india_harness.py
git commit -m "feat: replace Anthropic client in update_learnings with ClaudeCodeClient"
```

---

### Task 6: Remove anthropic from requirements.txt and verify clean

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Remove `anthropic` from requirements.txt**

Open `requirements.txt`. Remove the line:
```
anthropic==0.28.0
```

- [ ] **Step 2: Verify no remaining Anthropic imports**

```
grep -r "from anthropic\|import anthropic" . --include="*.py" --exclude-dir=".venv"
```

Expected: no output (zero matches)

- [ ] **Step 3: Run full test suite one final time**

```
pytest tests/ -q
```

Expected: all passing

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: remove anthropic SDK dependency — fully replaced by ClaudeCodeClient"
```

---

## Done

After all 6 tasks:
- No `ANTHROPIC_API_KEY` needed
- No `CRYPTOPANIC_API_KEY` needed
- `anthropic` package removed from requirements
- All LLM calls go through `claude` CLI subprocess (uses Claude Pro subscription)
- Bull+bear haiku calls run in parallel (saves ~2s per signal)
- News sentiment uses CryptoCompare free API
