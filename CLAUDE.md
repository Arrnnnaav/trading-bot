# Trading Bot v2 — India Index Options

## Before Making Any Changes

A knowledge graph of this codebase exists at `graphify-out/graph.json` with an interactive view at `graphify-out/graph.html` and summary at `graphify-out/GRAPH_REPORT.md`.

**Use it before touching code.** Run `/graphify query "<your question>"` or read `GRAPH_REPORT.md` to locate relevant nodes, dependencies, and affected components. Do NOT scan files one by one — the graph already maps every relationship.

Design spec for the v2 pivot: `docs/superpowers/specs/2026-06-18-india-trading-bot-design.md`

---

## Status: PAPER TRADING PENDING — all 5 build phases complete (2026-06-18)

Full pivot from crypto+India to **India-only index options trading**. All phases built. Waiting on real FII/DII data accumulation + improved model training before live gate passes.

---

## What This Bot Does

Semi-automated Indian index options trading bot. Generates BUY/SELL/HOLD signals → sends to Telegram with entry/target/stop/size → user taps EXECUTE → bot places options order via Upstox → stop-loss managed automatically.

**Instruments:** Nifty 50, Bank Nifty, Sensex, Nifty IT (index options only)  
**Execution:** Semi-auto (Telegram confirm) until bot is proven, then fully auto  
**Broker:** Upstox (Zerodha Kite added later on instruction)

---

## Index Options — Critical Lot Sizes

> SEBI revises these periodically. **Verify at NSE before live trading.**
> Last verified: 2026-06-18

| Index | Lot Size | Strike Step | Spot Token |
|-------|----------|-------------|------------|
| Nifty 50 | **65** | 50 | `NSE_INDEX\|Nifty 50` |
| Bank Nifty | **30** | 100 | `NSE_INDEX\|Nifty Bank` |
| Sensex | **20** | 100 | `BSE_INDEX\|SENSEX` |
| Nifty IT | **35** | 50 | `NSE_INDEX\|Nifty IT` |

---

## Architecture at a Glance

```
yfinance (25yr historical) → training pipeline
Upstox OHLCV (live 15m/daily) → IndiaIntradayHarness (9:20/10:15/11:30/13:00/14:00 IST)
                              → IndiaPositionalHarness (9:20 IST daily)
                                         ↓
                    6 agents → DebateEngine → RiskManager
                                         ↓
               SignalAggregator → TelegramBot → UpstoxBroker
                                         ↓
               signal_log.json ← SignalTracker (every 15min)
               StopMonitor runs every 5min (+ hard exit at 15:15 IST intraday)
               FastAPI dashboard at http://localhost:5000
```

---

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Fill .env — all keys required
python main.py
```

Dashboard: `http://localhost:5000`

---

## File Structure & Responsibilities

```
trading-bot/
├── main.py                      — entry point; wires all components; asyncio event loop
├── config.py                    — Config dataclass; reads env vars; config = Config() singleton
│
├── agents/
│   ├── base.py                  — abstract BaseAgent; analyze(ticker, klines, market) -> AgentVote
│   ├── technical_india.py       — TechnicalIndiaAgent; RSI/MACD/BB/VWAP/EMA + S/R zones + Kronos-base
│   ├── options_chain.py         — OptionsChainAgent; PCR, max pain, OI skew, IV percentile, unusual OI
│   ├── news_analogue.py         — NewsAnalogueAgent; ChromaDB semantic match → historical regime vote
│   ├── fii_dii.py               — FIIDIIAgent; nsefin daily flows + 5d rolling trend + DII/FII divergence
│   ├── macro_india.py           — MacroIndiaAgent; India VIX, SGX Nifty gap, USD/INR, RBI calendar
│   ├── kronos_india.py          — KronosAgent; NeoQuasar/Kronos-base fine-tuned on NSE data (positional only)
│   └── fundamentals.py          — FundamentalsAgent; NSE India API (kept for sector context)
│
├── brokers/
│   ├── base_broker.py           — abstract BrokerBase
│   ├── upstox.py                — UpstoxBroker; all 4 indices; lot sizes in _INDEX_SPEC dict
│   └── paper_broker.py          — PaperBroker wrapper; intercepts orders; logs to data/paper_trades.json
│
├── core/
│   ├── models.py                — Pydantic v2: Market/Direction/SignalOutcome, AgentVote, Signal, Position
│   ├── debate_engine.py         — DebateEngine; haiku researchers → opus adjudicator
│   ├── agent_tracker.py         — per-agent win rates → vote weights [0.5–2.0]
│   ├── pattern_reputation.py    — signal pattern fingerprinting + win rate gate (<40% blocks)
│   ├── risk_manager.py          — ATR stops; approve() gates incl. VIX gate (>25 blocks all)
│   ├── signal_aggregator.py     — atomic JSON log; 4hr dedup window (2hr intraday)
│   ├── signal_tracker.py        — resolves PENDING signals: TARGET_HIT/STOP_HIT/EXPIRED
│   ├── stop_monitor.py          — every 5min; closes positions; 15:15 IST hard exit for intraday
│   ├── json_store.py            — write_json_atomic(); read_json(); thread-safe file I/O
│   └── llm_client.py            — ClaudeCodeClient; used by DebateEngine + BaseHarness learnings
│
├── harnesses/
│   ├── base_harness.py          — BaseHarness; run_session() → vote→consensus→approve→log→telegram
│   │                              update_learnings() every 10 sessions via claude-haiku
│   ├── india_intraday_harness.py — 9:20/10:15/11:30/13:00/14:00 IST; 15m candles; 4 agents
│   └── india_positional_harness.py — 9:20 IST daily; daily candles; 6 agents incl. Kronos
│
├── tg_bot/
│   └── bot.py                   — TelegramBot; EXECUTE/SKIP/DEBATE buttons; user auth via TELEGRAM_ALLOWED_USER_IDS
│                                  Package named tg_bot/ (NOT telegram/) — avoids python-telegram-bot conflict
│
├── dashboard/
│   ├── server.py                — FastAPI; REST + WebSocket; binds 127.0.0.1 by default
│   └── static/                  — index.html, history.html, performance.html, positions.html
│
├── training/
│   ├── train_kronos_india.py    — 2-phase Kronos fine-tune on NSE data; saves models/kronos_india/best.pt
│   ├── train_xgb_india.py       — LightGBM training; Optuna HPO; saves models/xgb_india/model.pkl
│   ├── india_features.py        — feature engineering: RSI/MACD/BB/EMA/VWAP/VIX/FII/PCR/S-R distance
│   ├── walk_forward.py          — expanding-window OOS validation; gate: Sharpe >1.0 all OOS years
│   ├── label_data.py            — LONG/SHORT/HOLD labels; 10-day forward window; focal loss balanced
│   ├── focal_loss.py            — FocalLoss(gamma, weight) for class imbalance
│   └── dataset.py               — loads parquet files; temporal 80/10/10 split
│
├── backtesting/
│   └── india_backtest.py        — walk-forward backtest; STT/brokerage costs; options decay model
│                                  Gate for live: Sharpe >1.5, maxDD <15%, win rate >45%
│
├── scripts/
│   ├── fetch_historical.py      — one-time yfinance download: ^NSEI/^NSEBANK/^BSESN/^CNXIT from 2000
│   ├── backfill_news.py         — populate ChromaDB + news.db with historical events
│   └── retrain.py               — trigger retraining: XGBoost weekly, Kronos quarterly
│
├── models/
│   ├── kronos_india/            — fine-tuned Kronos checkpoint; best.pt
│   └── xgb_india/               — LightGBM model; model.pkl
│
├── data/
│   ├── historical/              — {index}_daily.parquet (25yr OHLCV, yfinance)
│   ├── fii_dii/                 — YYYY-MM-DD.json (daily FII/DII flows, nsefin)
│   ├── news/                    — YYYY-MM-DD.jsonl (raw news articles, append-only)
│   ├── chroma/                  — ChromaDB vector store (news embeddings)
│   ├── news.db                  — SQLite: market_events + linked news
│   ├── india_progress.json      — harness state: session_count, portfolio_value, open_positions
│   ├── signal_log.json          — all signals + outcomes (source of truth for dashboard)
│   └── paper_trades.json        — paper trading order log
│
├── tests/                       — pytest suite (119 tests passing as of 2026-06-18)
├── requirements.txt
├── .env.example
└── CLAUDE.md                    — this file
```

---

## Agents Per Harness

| Agent | Intraday (15m) | Positional (daily) |
|-------|---------------|-------------------|
| TechnicalIndiaAgent | ✅ primary | ✅ primary |
| OptionsChainAgent | ✅ primary | ✅ secondary |
| NewsAnalogueAgent | ✅ filter | ✅ primary |
| MacroIndiaAgent | ✅ primary | ✅ secondary |
| FIIDIIAgent | ❌ daily data only | ✅ primary |
| KronosAgent | ❌ too slow | ✅ primary |

Consensus: ≥2 agents agree + avg confidence ≥0.65

---

## Risk Gates (all active)

| Gate | Threshold |
|------|-----------|
| Max open positions | 3 per market |
| Daily loss | ≤2% portfolio |
| Weekly loss | ≤5% portfolio |
| Consecutive losses | ≤3 |
| Min R:R | 1.8 |
| Min confidence | 0.65 |
| India VIX | Reject all if >25 |
| Expiry proximity | Reject intraday if <2hr to expiry |
| Intraday hard exit | 15:15 IST (StopMonitor) |

---

## Trading Parameters

| Parameter | Value |
|-----------|-------|
| Max portfolio % per trade | 2% |
| Max risk % per trade (stop sizing) | 0.5% |
| Options stop | 35% premium drop |
| Options target | 100% premium (doubles) |
| Intraday hold | Same day, hard exit 15:15 |
| Positional hold | Max 5 trading days |
| Intraday options | Weekly expiry, ATM |
| Positional options | Monthly expiry, 1 strike OTM |

---

## LLM Models Used

| Component | Model |
|-----------|-------|
| Debate — consensus adjudicator | `claude-opus-4-7` |
| Debate — per-agent researchers | `claude-haiku-4-5-20251001` |
| Agent learnings summarizer | `claude-haiku-4-5-20251001` |

---

## Key Design Decisions & Gotchas

### 1. Package naming: `tg_bot/` not `telegram/`
Folder named `tg_bot/` because `telegram/` shadows the python-telegram-bot library.
**Always import as:** `from tg_bot.bot import TelegramBot`

### 2. Two harnesses, one Market enum
Both `IndiaIntradayHarness` and `IndiaPositionalHarness` use `Market.INDIA`. They share one position cap (max 3 total India positions across both harnesses). `main.py` tracks combined `harness_states[Market.INDIA]`.

### 3. Lot sizes are in `_INDEX_SPEC` dict — not hardcoded
`brokers/upstox.py` has a single `_INDEX_SPEC` class dict with lot size, strike step, exchange token per index. Update this dict when SEBI revises — never hardcode lot sizes elsewhere.

### 4. Intraday hard exit
`StopMonitor.run_once()` checks 15:15 IST and force-closes all intraday positions regardless of P&L. This prevents overnight options risk from day-trading positions.

### 5. VIX gate is in RiskManager
`RiskManager.approve()` checks India VIX via `MacroIndiaAgent` cached value. Above 25 = reject all new signals. This is a hard gate, not a confidence penalty.

### 6. ChromaDB is local
`data/chroma/` is a local ChromaDB instance — no server needed. Persistence is automatic. On first run, `scripts/backfill_news.py` must be run to populate historical events.

### 7. Kronos is positional-only
`NeoQuasar/Kronos-base` inference takes 2–5 seconds per prediction. Too slow for intraday. `KronosAgent` only runs in `IndiaPositionalHarness`.

### 8. Upstox token refresh
Upstox access tokens expire daily. Current code reads `UPSTOX_ACCESS_TOKEN` from env at startup. For production: implement OAuth2 refresh in `brokers/upstox.py` (v2 priority).

### 9. Zerodha Kite — not yet wired
Zerodha is the planned primary broker. Integration begins only when user instructs. Will be added as `brokers/zerodha.py` behind existing `BrokerBase` interface. `kiteconnect` Python SDK.

### 10. Sensex uses BSE exchange tokens
Sensex options trade on BSE (`BSE_FO|` prefix), not NSE. Upstox supports BSE F&O. Token format differs from NSE indices — handled in `_INDEX_SPEC`.

---

## What Needs to Happen Before Going Live

### One-time setup
1. Run `scripts/fetch_historical.py` — downloads 25yr OHLCV (takes ~5 min)
2. Run `scripts/backfill_news.py` — populates ChromaDB + news.db (~30 min)
3. Train models: `python -m training.train_kronos_india` + `python -m training.train_xgb_india`
4. Run `training/walk_forward.py` — validate both models (Sharpe >1.0 gate)
5. Run `backtesting/india_backtest.py` — validate strategy (Sharpe >1.5, maxDD <15% gate)
6. Fill all keys in `.env`
7. Set `TELEGRAM_ALLOWED_USER_IDS` to your Telegram user ID
8. Paper trade ≥4 weeks, verify signal quality, then flip to live

### Weekly (every Thursday before close)
Update `INDIA_OPTIONS_EXPIRY` in `.env` to next Thursday's date, OR leave blank to use `_next_weekly_expiry()` auto-calculation.

### Environment variables required
```
TELEGRAM_TOKEN
TELEGRAM_CRYPTO_CHAT_ID      # rename to TELEGRAM_INDIA_CHAT_ID in next cleanup
TELEGRAM_INDIA_CHAT_ID
TELEGRAM_ALLOWED_USER_IDS    # comma-separated Telegram user IDs (security)
UPSTOX_API_KEY
UPSTOX_ACCESS_TOKEN          # expires daily — refresh manually until OAuth2 implemented
MARKETAUX_API_KEY             # free tier, 100 req/day
PAPER_TRADING=true            # keep true until models validated
ENABLE_INTRADAY=true
ENABLE_POSITIONAL=true
```

---

## v2 Build Phases (in order)

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Done | Architecture hardening (atomic writes, retries, logging, tests — 119 passing) |
| 1 | ✅ Done | Crypto removal + India data pipeline + new harness structure |
| 2 | ✅ Done | 6 new/upgraded agents (Technical, OptionsChain, NewsAnalogue, FII/DII, Macro, Kronos) |
| 3 | ✅ Done | Kronos + XGBoost training on Indian data (models trained; F1=0.36 — needs more FII data) |
| 4 | ✅ Done | News intelligence (ChromaDB 66 docs, 15 events, FII/DII fetcher, live news scheduler) |
| 5 | ✅ Done | Backtesting + walk-forward validation (infrastructure complete; gate fails — model needs data) |

---

## Running Tests

```bash
python -m pytest tests/ -v --basetemp=.pytest-tmp
# Expected: 247 passed (as of 2026-06-18, post all phases)
```

---

## Related Docs

- `docs/superpowers/specs/2026-06-18-india-trading-bot-design.md` — full v2 design spec
- `docs/2026-05-23-trading-bot-design.md` — original v1 design (historical reference)
