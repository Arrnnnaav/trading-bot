# India Trading Bot v2 — Design Spec
**Date:** 2026-06-18  
**Status:** Approved for implementation planning  
**Scope:** Full pivot from crypto+India to India-only index options trading

---

## Context & Motivation

Backtests (2026-06-11) proved crypto directional signals have no deployable edge (Chronos v2: 0 signals / 87,585 candles). The bot is being fully pivoted to Indian index options trading — the proven, liquid, well-understood market. Crypto components are removed entirely. The existing architecture (BaseHarness, DebateEngine, RiskManager, SignalAggregator, StopMonitor, TelegramBot, Dashboard) is preserved and extended.

---

## Instruments

| Index | Exchange | Lot Size | Strike Step | Spot Token |
|-------|----------|----------|-------------|------------|
| Nifty 50 | NSE | **65** | 50 | `NSE_INDEX\|Nifty 50` |
| Bank Nifty | NSE | **30** | 100 | `NSE_INDEX\|Nifty Bank` |
| Sensex | BSE | **20** | 100 | `BSE_INDEX\|SENSEX` |
| Nifty IT | NSE | **35** | 50 | `NSE_INDEX\|Nifty IT` |

> Lot sizes per SEBI/NSE revisions as of 2025-2026. Verify at NSE before live trading.

---

## Sub-project Build Order

1. **Foundation** — crypto removal + data pipeline
2. **Agents** — 6 new/upgraded India agents
3. **Training** — Kronos fine-tune + XGBoost for India
4. **News intelligence** — ChromaDB + historical analogue matching
5. **Backtesting** — walk-forward on 25yr NSE data

---

## Sub-project 1: Foundation (Crypto Removal + Data Pipeline)

### Delete entirely
- `brokers/coindcx.py`
- `agents/chronos_technical.py`, `macro_crypto.py`, `news_sentiment.py`, `onchain.py`
- `harnesses/crypto_harness.py`
- `core/funding_carry_monitor.py`
- `backtesting/crypto_backtest.py`, `funding_carry.py`, `momentum_4h.py`
- `training/train_xgb.py`, `xgb_features.py`, `model_v2.py`, `train_v2.py`
- `models/chronos_crypto*/` directories
- Config fields: `coindcx_*`, `coinglass_api_key`, `crypto_signal_expiry_hours`, `enable_funding_carry`
- Tests for deleted components

### Keep unchanged
- All of `core/` (debate_engine, risk_manager, signal_aggregator, stop_monitor, models, agent_tracker, pattern_reputation, json_store, llm_client)
- `brokers/upstox.py`, `paper_broker.py`, `base_broker.py`
- `harnesses/base_harness.py`
- `tg_bot/`, `dashboard/`, `main.py` structure
- All shared tests

### Data pipeline additions

**Historical OHLCV (training + backtesting):**
- `scripts/fetch_historical.py` — yfinance download: `^NSEI`, `^NSEBANK`, `^BSESN`, `^CNXIT` from 2000-01-01
- Stored as `data/historical/{index}_daily.parquet`
- Incremental daily append after initial fetch

**Live intraday data:**
- Extend `brokers/upstox.py` `get_ohlcv()` to accept `5minute`/`15minute` intervals
- New `get_ohlcv_intraday(instrument, interval, days_back=5)` method

**FII/DII daily data:**
- `nsefin.get_fii_dii_activity()` — fetched at 19:00 IST after NSE publishes
- Cached at `data/fii_dii/YYYY-MM-DD.json`

### New harness structure
```
harnesses/india_intraday_harness.py   — 9:20/10:15/11:30/13:00/14:00 IST, 15m candles
harnesses/india_positional_harness.py — 9:20 IST daily, daily candles
```
Both extend `BaseHarness`. `Market.INDIA` used for both — shared position cap, shared progress file.

---

## Sub-project 2: Agents

### Agent roster

| Agent | File | Harness | Primary signal |
|-------|------|---------|---------------|
| TechnicalIndiaAgent | `agents/technical_india.py` | Both | RSI/MACD/BB/VWAP + S/R zones + Kronos-base |
| OptionsChainAgent | `agents/options_chain.py` | Both | PCR, max pain, OI skew, IV percentile |
| NewsAnalogueAgent | `agents/news_analogue.py` | Both | ChromaDB historical match → regime vote |
| FIIDIIAgent | `agents/fii_dii.py` (upgrade) | Positional | 5-day rolling net flow + DII/FII divergence |
| MacroIndiaAgent | `agents/macro_india.py` | Both | India VIX, SGX Nifty gap, USD/INR, RBI calendar |
| KronosAgent | `agents/kronos_india.py` | Positional | Fine-tuned Kronos-base: 3/5/10-day direction prob |

### Agent details

**TechnicalIndiaAgent:**
- Indicators: RSI-14/7, MACD, Bollinger Bands, VWAP, EMA 9/21/50/200, ATR-14
- Support/resistance: fractal pivot method + volume profile (highest-volume price clusters)
- LONG if price breaks above resistance with volume surge; SHORT if breaks below support
- Also runs Kronos-base inference for 5-candle forward forecast (shared with KronosAgent via cached result)

**OptionsChainAgent:**
- PCR >1.3 → contrarian LONG; PCR <0.7 → contrarian SHORT
- Max pain strike: price gravitates toward max pain near expiry (within 2 DTE)
- OI buildup: large call OI wall = resistance; large put OI wall = support
- IV percentile: >80th → sell premium bias; <20th → buy premium bias
- Unusual OI surge >3× avg = smart money directional signal

**NewsAnalogueAgent:**
- Embeds today's top-5 headlines (Marketaux + NSE announcements + RSS)
- ChromaDB cosine similarity → top-3 historical analogues
- Only fires if similarity score >0.75
- Historical outcome (nifty_pct_change_5d) → weighted vote direction

**FIIDIIAgent (upgraded):**
- `nsefin` library for daily net flows
- 5-day rolling net: FII >₹2000Cr AND accelerating → LONG
- Sustained FII selling >3 days → SHORT
- DII buying while FII selling = market floor signal (HOLD or weak LONG)

**MacroIndiaAgent:**
- India VIX >25 → all signals blocked (added to RiskManager gate)
- India VIX >20 → confidence penalty -0.10
- SGX Nifty gap >+1% → intraday LONG bias; gap <-1% → SHORT bias
- USD/INR rising fast (>0.5% week) → FII outflow pressure → SHORT bias
- RBI policy day: widen stops, reduce confidence

**KronosAgent (positional only):**
- `NeoQuasar/Kronos-base` via HuggingFace
- Fine-tuned on NSE daily data 2000–present
- Predicts direction probability for 3/5/10 trading days ahead
- Confidence = max(long_prob, short_prob); direction = argmax

### Consensus rules (unchanged)
- ≥2 agents agreeing on same direction
- Average confidence ≥0.65
- RiskManager gates: max 3 open positions, daily loss ≤2%, weekly ≤5%, consecutive losses ≤3
- NEW gate: India VIX >25 → reject all signals
- NEW gate: intraday signals rejected if <2hr to weekly expiry

---

## Sub-project 3: Training Pipeline

### Kronos fine-tuning
- Base: `NeoQuasar/Kronos-base` (HuggingFace, pre-trained 12B+ K-lines, 45 exchanges)
- Data: `data/historical/*.parquet` — 25yr daily OHLCV, 4 indices
- Labels: LONG if 10-day fwd return >+1.5% with clean path; SHORT if <-1.5%; HOLD otherwise
- Temporal split: 2000–2018 train / 2019–2022 val / 2023–present test
- Phase 1: frozen encoder, head only, 20 epochs, lr=1e-4
- Phase 2: full fine-tune, lr=1e-5, grad accum=4, focal loss (existing `training/focal_loss.py`)
- Save: `models/kronos_india/best.pt` on val macro-F1 improvement
- Script: `training/train_kronos_india.py`

### XGBoost / LightGBM
- Features: RSI, MACD, BB, EMA ratios, VWAP dev, ATR, volume ratio, OBV slope, India VIX, SGX gap, USD/INR change, FII 5d net, PCR, distance from S/R, days since 52w high
- Temporal CV: 5-fold expanding window
- Hyperparameter search: Optuna, 50 trials
- Script: `training/train_xgb_india.py`; features in `training/india_features.py`
- Save: `models/xgb_india/model.pkl`

### Walk-forward validation (`training/walk_forward.py`)
- Trains 2000–2018, tests 2019 OOS → repeats year by year to present
- Gate: Sharpe >1.0 across all OOS years before model graduates to paper trading
- Auto-rollback: if new model OOS F1 drops >10% vs current → keep old, alert Telegram

### Retraining schedule
- XGBoost: weekly (fast, <5min)
- Kronos: quarterly, expanding window (30–60min)
- Triggered via `scripts/retrain.py`

---

## Sub-project 4: News Intelligence

### Storage architecture
- **ChromaDB** (local, persistent at `data/chroma/`) — news embeddings + metadata, semantic search
- **SQLite** (`data/news.db`) — structured market events table
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, free, 384-dim)

### Schema
```sql
-- SQLite
CREATE TABLE market_events (
    event_id TEXT PRIMARY KEY,
    name TEXT,                        -- "dot-com crash phase 2"
    category TEXT,                    -- bubble/crash/correction/rally/policy/geopolitical
    start_date DATE, end_date DATE,
    nifty_peak_to_trough_pct REAL,
    recovery_days INTEGER,
    key_characteristics JSON          -- ["euphoria", "high PE", "retail frenzy"]
);

-- ChromaDB collection: news_articles
-- Fields: id, embedding, headline, body_snippet, date, source, event_id, tags
```

### Pre-loaded historical events
dot-com rise/crash (1999–2002), 9/11, 2008 GFC, 2013 Taper Tantrum, India demonetization (Nov 2016), 2020 COVID crash + V-recovery, 2022 rate hike cycle, India budget surprises (yearly), RBI surprise policy days, 2024 AI bubble concerns

### Live news ingestion
- Sources: Marketaux API (free, 100 req/day), NSE corporate announcements (no key), RSS (Economic Times, Moneycontrol, Mint)
- Poll interval: every 30min
- Raw storage: `data/news/YYYY-MM-DD.jsonl` (append-only)
- `scripts/backfill_news.py` — one-time historical article scrape and DB population

### Signal-time usage
1. Embed today's top-5 headlines → ChromaDB similarity search
2. Filter: similarity >0.75 only
3. Pull linked market_event → get historical Nifty reaction stats
4. NewsAnalogueAgent votes based on historical outcome distribution

---

## Sub-project 5: Backtesting

### Framework (`backtesting/india_backtest.py`)
- Data: `data/historical/*.parquet` (2000–present)
- India-specific costs: STT 0.05% on sell side, brokerage ₹20/order, SEBI charges, stamp duty
- Options premium decay modeled (Black-Scholes approximation)
- Walk-forward: same as training validation
- Metrics: Sharpe, Sortino, max drawdown, profit factor, win rate, avg hold days, monthly P&L distribution
- Gate for live: Sharpe >1.5, max drawdown <15%, win rate >45%

---

## Execution Model

**Now:** Semi-auto (Telegram EXECUTE/SKIP/DEBATE)  
**Future (when proven):** Fully auto — flip via `EXECUTION_MODE=auto` env var

**Options execution:**
- Intraday signals → weekly expiry ATM options (0–7 DTE)
- Positional signals → next monthly expiry, 1 strike OTM
- Hard intraday exit: 15:15 IST (new StopMonitor rule)
- Max 3 lots per index per direction

---

## Config Changes

### Remove
`coindcx_api_key`, `coindcx_api_secret`, `coinglass_api_key`, `crypto_signal_expiry_hours`, `enable_funding_carry`

### Add
```python
marketaux_api_key: str = ""
india_vix_gate: float = 25.0
intraday_force_exit: str = "15:15"
positional_max_days: int = 5
kronos_model_path: str = "models/kronos_india/best.pt"
xgb_model_path: str = "models/xgb_india/model.pkl"
news_db_path: str = "data/news.db"
chroma_db_path: str = "data/chroma"
enable_intraday: bool = True
enable_positional: bool = True
execution_mode: str = "semi_auto"   # or "auto"
```

---

## New Dependencies

```
yfinance              # 25yr NSE/BSE historical OHLCV
nsefin                # FII/DII daily flows (NSE-native)
nsepython             # NSE live data
chromadb              # vector store for news intelligence
sentence-transformers # local news embeddings (all-MiniLM-L6-v2)
lightgbm              # XGBoost replacement (faster)
optuna                # hyperparameter search
feedparser            # RSS news scraping
```

---

## Broker Plan

**Now:** Upstox only (already integrated)  
**Later (on user instruction):** Add Zerodha Kite as primary via `kiteconnect` Python SDK, wrapped behind existing `BrokerBase` interface. Upstox becomes secondary/fallback.

---

## What Is NOT Changing

- `BaseHarness` orchestration loop
- `DebateEngine` (haiku researchers + opus adjudicator)
- `RiskManager` (all existing gates preserved, 2 new added)
- `SignalAggregator` + dedup logic
- `StopMonitor` (+ intraday 15:15 hard exit)
- `SignalTracker` outcome resolution
- `TelegramBot` (EXECUTE/SKIP/DEBATE buttons)
- `Dashboard` (FastAPI + WebSocket)
- `AgentPerformanceTracker` + `PatternReputationTracker`
- All atomic JSON write patterns
- Paper trading mode
