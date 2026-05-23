# Crypto Intelligence Overhaul — Design Spec

**Date:** 2026-05-24  
**Scope:** Crypto market only (BTC/ETH/SOL/BNB/XRP)  
**Status:** Approved, pending implementation

---

## Problem Statement

The current `KronosTechnicalAgent` is non-functional:
- `kronos` package does not exist on PyPI — import always fails silently
- Fallback uses normalized close prices as cosine similarity vectors — signal is noise
- Pattern matching history is RAM-only — resets on every restart, always returns HOLD for first 5 sessions
- Agent learnings loop calls claude-haiku every 10 sessions but output goes nowhere (no agent uses an LLM prompt)
- OnChain/News agent thresholds are hardcoded guesses with no empirical basis

---

## Approach: Focused Crypto Overhaul (Approach B)

Six components, delivered in order:

1. Data pipeline — download + label 5yr Binance OHLCV
2. Chronos-2 fine-tuning — classification head on T5 encoder
3. ChronosTechnicalAgent — replaces KronosTechnicalAgent
4. Agent learnings loop fix — route haiku output to DebateEngine
5. Backtesting harness — validate on test split before going live
6. Threshold auto-calibration — OnChain/News learn from signal history

---

## Section 1: Data Pipeline

### Download
- Source: Binance public REST API (`/api/v3/klines`) — no auth required
- Tickers: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT
- Interval: 15min candles
- History: 5 years (~175,000 candles per ticker, ~875,000 total)
- Format: parquet, one file per ticker
- Output: `data/training/{ticker}_15m.parquet`
- Script: `training/download_data.py`

### Labeling
- Script: `training/label_data.py`
- Forward-look window: 4 candles (1 hour ahead)

```
For each candle at index i, look at candles i+1 through i+4:
  LONG  — price hits +1.0% before hitting -0.5% stop
  SHORT — price hits -1.0% before hitting +0.5% stop
  HOLD  — neither threshold reached in window
```

- Expected class distribution: ~20% LONG / ~20% SHORT / ~60% HOLD
- HOLD dominance is correct — most candles are not actionable
- Output column: `label` added to parquet

### Train/Val/Test Split
- 80% train / 10% val / 10% test
- **Time-ordered only — never shuffled**
- Walk-forward split prevents future data leakage
- Test set is reserved for backtesting only, never seen during training

### PyTorch Dataset
- Script: `training/dataset.py`
- Input window: last 96 candles (24 hours) per sample
- Features per candle: [open, high, low, close, volume] — 5 channels
- Normalization: z-score per channel per sample (not global, prevents look-ahead bias)
- Returns: `(tensor[96, 5], label_int)`

---

## Section 2: Chronos-2 Fine-tuning Architecture

### Base Model
- `amazon/chronos-t5-small` from HuggingFace (46M params)
- Upgrade path: `chronos-t5-base` (200M) if small underperforms on val F1
- Multivariate input supported natively in Chronos-2

### Architecture
```
Input: OHLCV [96 × 5]
  ↓
Chronos-2 T5 Encoder
  ↓
Last hidden state [512-dim]
  ↓
LayerNorm
Dropout(0.2)
Linear(512 → 128)
GELU
Linear(128 → 3)
  ↓
Softmax → [P(LONG), P(SHORT), P(HOLD)]
```

### Training Protocol
- **Phase 1** — freeze encoder, train head only
  - Epochs: 2
  - LR: 1e-3, AdamW
  - Batch size: 64
- **Phase 2** — unfreeze all, full fine-tune
  - Epochs: 3
  - LR: 1e-5, AdamW with cosine decay
  - Batch size: 32
- Loss: CrossEntropy with class weights (upweight LONG/SHORT to counter HOLD dominance)
- Class weights: `{HOLD: 1.0, LONG: 3.0, SHORT: 3.0}`
- Checkpoint: save best by val macro-F1 (not accuracy — HOLD dominates accuracy)
- Early stopping: patience 2 epochs on val F1

### Output Path
- `models/chronos_crypto/best.pt` — best checkpoint
- `models/chronos_crypto/config.json` — input normalization stats, label map, threshold

### Scripts
- `training/model.py` — `ChronosClassifier` class (encoder + head)
- `training/train.py` — training loop, checkpointing, logging
- `training/retrain.py` — incremental re-fine-tune from `chronos_outcomes.jsonl`

---

## Section 3: ChronosTechnicalAgent

**File:** `agents/chronos_technical.py`  
**Replaces:** `agents/kronos_technical.py` (deleted)

### Startup
```python
def __init__(self, model_path: str = "models/chronos_crypto/best.pt"):
    # Fails loudly if model not found — no silent fallback
    self.model = ChronosClassifier.load(model_path)
    self.model.eval()
```

### analyze() Flow
```python
def analyze(self, ticker, klines, market, **kwargs) -> AgentVote:
    window = klines[-96:]              # last 96 candles
    x = normalize(window)              # z-score per channel
    probs = self.model(x)              # [P(LONG), P(SHORT), P(HOLD)]
    direction = argmax(probs)
    confidence = max(probs)
    if confidence < 0.55:
        direction = HOLD
    return AgentVote(direction, confidence, reasoning)
```

### Reasoning String
```
"Chronos-2: LONG (p=0.71) | LONG=0.71 SHORT=0.15 HOLD=0.14 | 96 candles, BTC/USDT"
```

### Outcome Persistence
- `SignalTracker` appends to `data/chronos_outcomes.jsonl` when signal resolves:
```json
{"ts": "2026-05-24T14:00Z", "ticker": "BTC/USDT", "predicted": "LONG", "confidence": 0.71, "actual": "TARGET_HIT"}
```
- Survives restarts — no RAM-only history
- Used by `training/retrain.py` for periodic re-fine-tuning (trigger: 500+ new outcomes)

### main.py Change
`CryptoHarness` instantiates `ChronosTechnicalAgent` instead of `KronosTechnicalAgent`.

---

## Section 4: Agent Learnings Loop Fix

### Current Bug
`BaseHarness.update_learnings()` → claude-haiku summarizes outcomes → saved to `progress.json["agent_learnings"]` → prepended to "agent prompts" → **no agent uses an LLM prompt** → learnings go nowhere.

### Fix
Route learnings to `DebateEngine`, which already calls claude-haiku.

**`DebateEngine.__init__`** — accepts `learnings: str = ""`

**`DebateEngine.generate_reasoning()`** — system prompt:
```
You are a crypto trading analyst.

Recent learnings from signal outcomes:
{learnings}

Analyze the following agent debate and write a 2-sentence trading rationale.
```

**`BaseHarness.update_learnings()`** — unchanged logic (still calls haiku every 10 sessions, still saves to `progress.json`). Additionally writes to `data/crypto_learnings.md` for human review.

**`CryptoHarness.run_session()`** — passes current learnings string into DebateEngine at each session start.

### Files Changed
- `core/debate_engine.py` — accept + inject learnings
- `harnesses/base_harness.py` — write learnings to `.md` file
- `harnesses/crypto_harness.py` — pass learnings to DebateEngine

---

## Section 5: Backtesting Harness

### Purpose
Validate signal quality on the reserved test split (last 10% of 5yr data) before live trading.

### `backtesting/crypto_backtest.py`

Walk-forward simulation over test set:
1. Feed `klines[-96:]` to `ChronosTechnicalAgent.analyze()`
2. **Bypass DebateEngine** — historical News/OnChain API snapshots unavailable, so DebateEngine's ≥2 agent consensus rule would produce zero signals. Backtest runs Chronos-2 standalone.
3. Signal fires if: `direction != HOLD` AND `confidence >= 0.55`
4. Simulate trade using forward candles — check TARGET_HIT / STOP_HIT / EXPIRED using same ATR-based stop/target as live bot
5. Log result

Note: Full multi-agent backtest (with replayed News/OnChain data) is a v2 enhancement requiring historical API snapshots.

### Metrics
```
Signal rate:         X% of candles produce a signal
Win rate:            TARGET_HIT / (TARGET_HIT + STOP_HIT)
Avg R:R achieved:    mean(reward) / mean(risk) on closed trades
Sharpe ratio:        annualized
Max drawdown:        worst peak-to-trough on equity curve
Chronos-2 F1:        per-class precision/recall/F1
```

### Output
- `backtesting/results/YYYY-MM-DD-crypto-backtest.json` — full signal log
- `backtesting/results/YYYY-MM-DD-crypto-backtest-report.md` — human-readable summary
- `backtesting/plot_results.py` — equity curve + drawdown chart (matplotlib)

---

## Section 6: Threshold Auto-calibration

### Problem
`OnChainAgent`: `funding < -0.001`, `ls_ratio < 0.9` — guesses.  
`NewsSentimentAgent`: `score > 0.2` — guess.

### Fix
`training/calibrate_thresholds.py` — reads all resolved signals from `signal_log.json`, finds threshold values that maximise precision on TARGET_HIT signals. Requires 50+ resolved signals to run; falls back to hardcoded defaults if insufficient data.

### Output
`data/calibrated_thresholds.json`:
```json
{
  "onchain_funding_threshold": -0.0008,
  "onchain_ls_ratio_threshold": 0.85,
  "news_sentiment_threshold": 0.18,
  "sample_size": 143,
  "updated_at": "2026-05-24T09:00Z"
}
```

### Agent Changes
- `OnChainAgent.__init__` — load thresholds from file, fall back to hardcoded if missing
- `NewsSentimentAgent.__init__` — same
- `CryptoHarness.__init__` — calls `calibrate_thresholds()` on startup

---

## File Change Summary

| Action | Files |
|--------|-------|
| New | `training/download_data.py` |
| New | `training/label_data.py` |
| New | `training/dataset.py` |
| New | `training/model.py` |
| New | `training/train.py` |
| New | `training/retrain.py` |
| New | `training/calibrate_thresholds.py` |
| New | `agents/chronos_technical.py` |
| New | `backtesting/crypto_backtest.py` |
| New | `backtesting/plot_results.py` |
| Changed | `agents/onchain.py` |
| Changed | `agents/news_sentiment.py` |
| Changed | `core/debate_engine.py` |
| Changed | `harnesses/base_harness.py` |
| Changed | `harnesses/crypto_harness.py` |
| Changed | `core/signal_tracker.py` |
| Changed | `main.py` |
| Deleted | `agents/kronos_technical.py` |

---

## Execution Order

1. `training/download_data.py` — download 5yr Binance data
2. `training/label_data.py` — label candles
3. `training/train.py` — fine-tune Chronos-2
4. `backtesting/crypto_backtest.py` — validate on test split
5. Review backtest report — if metrics acceptable, proceed to live
6. `main.py` — run bot with `ChronosTechnicalAgent`

---

## Out of Scope (this spec)

- India market changes
- Upstox OAuth2 token refresh
- US market screener
- Mobile app
- Trailing stop-loss
