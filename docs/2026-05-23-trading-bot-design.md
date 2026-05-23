# Trading Bot — Design Spec
**Date:** 2026-05-23  
**Status:** Approved for implementation  
**Scope:** Semi-automated trading bot (Tool 1 of 2 — separate from stock screener)

---

## 1. Goal

Build a semi-automated trading system that:
- Generates BUY/SELL/HOLD signals for **crypto** (intraday) and **Indian markets** (swing)
- Sends signals to Telegram with confidence score, entry/target/stop, position size
- User confirms with one tap → bot executes via broker API
- Auto-manages stop-losses in background without user input
- Uses Kronos + TradingAgents + harness engineering patterns throughout

---

## 2. Markets & Brokers

| Market | Instruments | Broker API | Timeframe |
|--------|------------|------------|-----------|
| Crypto | BTC, ETH, top 5 CoinDCX pairs by volume | CoinDCX SDK (Python) | 15min candles (intraday) |
| Indian Equity | Nifty 50 stocks, Sensex stocks | Upstox SDK v3 (Python) | Daily candles (swing) |
| Indian F&O | Nifty 50 options, Bank Nifty options | Upstox SDK v3 (Python) | Daily + options chain — trades are ATM call/put buying only, no futures |

---

## 3. Architecture

Two independent harnesses share common infrastructure:

```
CoinDCX OHLCV  →  CryptoHarness   →┐
Upstox OHLCV   →  IndiaHarness    →┤→ SignalAggregator → TelegramBot → Broker APIs
                                    ↓
                              progress.json (per harness)
```

### 3.1 CryptoHarness
- **Schedule:** Every 15 minutes (APScheduler)
- **Tickers:** BTC/USDT, ETH/USDT + top 3 by 24h volume on CoinDCX
- **Session state:** `data/crypto_progress.json`

### 3.2 IndiaHarness
- **Schedule:** 09:00, 11:30, 14:45 IST (weekdays only)
- **Tickers:** Nifty 50 constituents + Bank Nifty
- **Session state:** `data/india_progress.json`

---

## 4. Specialist Agents (per harness, run in parallel)

### Crypto Agents
| Agent | Input | Output |
|-------|-------|--------|
| `KronosTechnicalAgent` | Last 60 OHLCV candles → Kronos encode | Pattern match score, 5-candle forecast, similar historical outcomes |
| `NewsSentimentAgent` | CryptoPanic API last 4hrs | Sentiment score -1.0 to +1.0, top headlines |
| `OnChainAgent` | CoinGlass API | Funding rate, long/short ratio, OI delta |

### India Agents
| Agent | Input | Output |
|-------|-------|--------|
| `KronosTechnicalAgent` | Last 60 daily OHLCV → Kronos encode | Pattern match score, next-day forecast |
| `FundamentalsAgent` | NSE India API | P/E vs sector, delivery %, promoter holding delta |
| `FIIDIIAgent` | NSE FII/DII daily data | Net institutional flow (buy/sell pressure) |
| `OptionsOIAgent` | Upstox options chain | PCR, max pain level, OI buildup strikes |

---

## 5. Debate Engine

After parallel agent analysis, a `DebateEngine` runs:

1. Each agent submits: `{signal: BUY|SELL|HOLD, confidence: 0-1, reasoning: str}`
2. Agents with confidence > 0.5 get to rebut others' signals (one round)
3. `RiskManager` reads `progress.json` + debate output → final signal with position size

**Consensus rule:** Signal published only if ≥ 2 agents agree AND combined confidence ≥ 0.65. Below threshold → HOLD, no Telegram message.

---

## 6. Risk Manager

Reads `progress.json` to enforce:
- Max 2% portfolio per trade
- Max 3 open positions per market simultaneously
- No new signal if existing position in same ticker
- Stop-loss: 2× ATR below entry (crypto), 1.5× ATR (India)
- Target: minimum 1.8 R:R ratio required to publish signal

---

## 7. Harness State (`progress.json`)

```json
{
  "session_count": 0,
  "portfolio_value_inr": 100000,
  "open_positions": [],
  "signal_history": [],
  "agent_learnings": "",
  "last_run": null
}
```

`agent_learnings` field: after every 10 sessions, an LLM summarizes the last 50 signal outcomes into a 3-sentence learning note. This gets prepended to each agent's system prompt in subsequent sessions.

---

## 8. Telegram Bot

**Two channels:**
- `#crypto-signals` — crypto intraday signals
- `#india-signals` — Indian market swing signals

Telegram is for **trade execution only**. All signal tracking and history is on the web dashboard.

**Signal message format:**
```
🟢 LONG BTC/USDT
━━━━━━━━━━━━━━━━━
Entry:    ₹62,400
Target:   ₹65,100  (+4.2%)
Stop:     ₹61,000  (-2.2%)
R:R       1 : 1.9
Size:     ₹2,000  (2% portfolio)
Confidence: 74%

Kronos: 8/10 similar patterns resolved bullish
Funding: -0.02% (longs paid)
News: Neutral (0.1)

[✅ EXECUTE]  [❌ SKIP]  [📊 DEBATE]
```

**Button actions:**
- `EXECUTE` → place order via broker API → confirm fill message
- `SKIP` → log skip in progress.json (feeds agent learnings)
- `DEBATE` → send full agent debate transcript as follow-up message

**Stop-loss notifications (no confirmation needed):**
```
🔴 Stop hit: BTC/USDT
Closed at ₹61,050 | Loss: -₹182 (-0.18% portfolio)
```

---

## 9. Execution Layer

### CoinDCX
```python
# Market order if spread < 0.3%, else limit at ask+0.1%
coindcx_client.place_order(
    market="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=size_in_btc
)
```

### Upstox
```python
# Equity: market order during market hours
# F&O: limit order at LTP for options
upstox_client.place_order(
    instrument_token=token,
    transaction_type="BUY",
    order_type="MARKET",
    quantity=lot_size
)
```

Stop-loss orders placed immediately after fill via `SL-M` order type (both APIs support this).

---

## 10. Options Execution (India F&O)

When signal is LONG (bullish) → buy ATM Call option  
When signal is SHORT (bearish) → buy ATM Put option

```python
# Select ATM strike from options chain
spot_price = upstox.get_ltp("NSE_INDEX|Nifty 50")
atm_strike = round(spot_price / 50) * 50   # nearest 50 for Nifty

# Buy 1 lot ATM Call (expiry = nearest weekly)
upstox_client.place_order(
    instrument_token=atm_call_token,
    transaction_type="BUY",
    order_type="LIMIT",
    price=ltp + 0.5,   # small premium above LTP for fill
    quantity=50         # 1 lot Nifty = 50 units
)
```

Stop-loss for options: exit if option premium drops 40% from entry (e.g., buy at ₹100, exit at ₹60).  
Target: exit if premium doubles (100% gain) or at EOD, whichever comes first.

---

## 11. Background Stop Monitor

Separate async task running every 5 minutes:
- Fetches current prices for all open positions
- If price ≤ stop_loss → execute close → notify Telegram → update progress.json
- Trailing stop option (future): move stop up as price moves in favour

---

## 12. Signal Performance Tracker

**Every signal is tracked regardless of whether the user executed it.**

When a signal is generated, it is saved to `data/signal_log.json`:

```json
{
  "id": "sig_20260523_001",
  "market": "crypto",
  "ticker": "BTC/USDT",
  "direction": "LONG",
  "entry_price": 62400,
  "target_price": 65100,
  "stop_price": 61000,
  "confidence": 0.74,
  "generated_at": "2026-05-23T08:15:00Z",
  "executed": false,
  "outcome": null,
  "outcome_price": null,
  "outcome_at": null,
  "hypothetical_pnl_pct": null,
  "hypothetical_pnl_inr": null,
  "position_size_inr": 2000
}
```

**Outcome resolution** (background task, runs every 15min for crypto, daily for India):
- If price hits target → `outcome: "TARGET_HIT"`, calculate actual % gain
- If price hits stop → `outcome: "STOP_HIT"`, calculate actual % loss
- If neither within 24hrs (crypto) / 3 trading days (India) → `outcome: "EXPIRED"`

`hypothetical_pnl_inr` = what the user WOULD have made/lost if they had taken the trade at `position_size_inr`.

---

## 13. Web Dashboard

**Read-only. No trade execution from dashboard. Telegram handles execution.**

**URL:** `http://localhost:5000` (local, no auth needed for v1)

**Pages:**

### `/` — Live Feed
- Real-time signal stream (WebSocket from backend)
- Each signal card shows: ticker, direction, confidence bar, entry/target/stop, R:R
- Status badge: `PENDING` / `EXECUTED` / `SKIPPED` / `TARGET HIT` / `STOP HIT` / `EXPIRED`
- Color: green for LONG, red for SHORT, grey for expired

### `/history` — Signal History Table
- All signals ever generated, sortable by date / market / outcome / confidence
- Columns: Date | Market | Ticker | Direction | Confidence | Executed? | Outcome | Hypothetical P&L | Actual P&L
- Filter by: market (crypto/india), outcome, date range

### `/performance` — Analytics
- **Win rate:** % of signals that hit target (executed + hypothetical)
- **Hypothetical portfolio curve:** if you had taken every signal at 2% size, chart of portfolio value over time
- **Actual portfolio curve:** only executed trades
- **By market:** crypto vs India performance split
- **By agent:** which specialist agent's signals perform best (breakdown)
- **Average R:R achieved** vs planned

### `/positions` — Open Positions
- All currently open trades (executed ones)
- Live P&L per position (WebSocket price updates)
- Entry price, current price, stop level, target level

**Tech:** FastAPI backend (serves data) + vanilla JS + Chart.js frontend. No React — keep it simple for v1.

---

## 14. File Structure

```
trading-bot/
├── harnesses/
│   ├── crypto_harness.py       — CryptoHarness, APScheduler every 15min
│   └── india_harness.py        — IndiaHarness, 3x/day IST schedule
├── agents/
│   ├── kronos_technical.py     — Kronos embed + pyturboquant pattern match
│   ├── news_sentiment.py       — CryptoPanic API
│   ├── onchain.py              — CoinGlass API
│   ├── fundamentals.py         — NSE India API
│   ├── fii_dii.py              — NSE FII data
│   └── options_oi.py           — Upstox options chain (PCR, max pain)
├── core/
│   ├── debate_engine.py        — agent voting + rebuttal + consensus
│   ├── risk_manager.py         — position sizing, R:R check, ATM strike selection
│   ├── signal_aggregator.py    — dedup, rank, route to Telegram + log to signal_log.json
│   ├── signal_tracker.py       — outcome resolution, hypothetical P&L calc (background)
│   └── stop_monitor.py         — stop-loss checker every 5min (options: -40% premium)
├── brokers/
│   ├── coindcx.py              — CoinDCX SDK wrapper
│   └── upstox.py               — Upstox SDK v3 wrapper (equity + options)
├── telegram/
│   └── bot.py                  — python-telegram-bot v20, inline keyboards
├── dashboard/
│   ├── server.py               — FastAPI app, REST + WebSocket
│   └── static/
│       ├── index.html          — live signal feed
│       ├── history.html        — all signals table with filters
│       ├── performance.html    — win rate, portfolio curve, agent breakdown
│       └── positions.html      — open positions with live P&L
├── data/
│   ├── crypto_progress.json    — harness state, open positions, agent learnings
│   ├── india_progress.json     — harness state, open positions, agent learnings
│   └── signal_log.json         — every signal ever generated + outcome
├── config.py                   — API keys via env vars, all thresholds
├── main.py                     — starts harnesses + stop monitor + signal tracker + dashboard
└── requirements.txt
```

---

## 15. Tech Stack

| Component | Library |
|-----------|---------|
| Agent framework | TradingAgents (adapted) + direct Claude API |
| Market perception | Kronos (pre-trained, kronos-base) |
| Embedding compression | pyturboquant (4-bit, for pattern search index) |
| Scheduling | APScheduler |
| Telegram | python-telegram-bot v20 |
| Crypto broker | CoinDCX Python SDK |
| India broker | Upstox SDK v3 (equity + options) |
| State persistence | JSON files (progress.json per harness + signal_log.json) |
| LLM | claude-opus-4-7 (debate) + claude-haiku-4-5-20251001 (analysis agents) |
| Dashboard backend | FastAPI + WebSocket |
| Dashboard frontend | Vanilla JS + Chart.js (no framework) |
| Data: crypto news | CryptoPanic API (free) |
| Data: on-chain | CoinGlass API (free tier — 10 req/min, sufficient at 15min poll) |
| Data: India | NSE India unofficial API + Upstox historical |

---

## 16. Error Handling

- Agent failure → log + skip that agent's vote (debate proceeds with remaining agents)
- Broker API timeout → do NOT retry automatically → notify Telegram "⚠️ Order failed, place manually"
- CoinDCX/Upstox auth expiry → refresh token on startup, alert if refresh fails
- Kronos model error → fall back to raw RSI/MACD technical score
- Dashboard WebSocket disconnect → auto-reconnect with exponential backoff

---

## 17. Out of Scope (v1)

- Stock screener (separate tool, separate spec)
- Fully automated execution (v2 — after validating signal quality via signal tracker)
- Backtesting engine (use TradingAgents backtest module separately)
- US market trading (screener only, not trading)
- Portfolio rebalancing
- Mobile app (Telegram + web dashboard covers both)
