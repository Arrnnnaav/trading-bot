# Trading Bot v1 — Project Context

## Status: COMPLETE (built 2026-05-23, 25 tests passing)

Semi-automated crypto + Indian market trading bot. Generates BUY/SELL/HOLD signals → sends to Telegram with entry/target/stop/size → user taps EXECUTE → bot places order via broker API → stop-loss managed automatically in background.

---

## Architecture at a Glance

```
CoinDCX OHLCV  →  CryptoHarness (every 15min)   →┐
Upstox OHLCV   →  IndiaHarness  (9:00/11:30/14:45 IST weekdays) →┤
                                                   ↓
                         3+ parallel agents → DebateEngine → RiskManager
                                                   ↓
                         SignalAggregator → TelegramBot → Broker APIs
                                                   ↓
                         signal_log.json ← SignalTracker (every 15min)
                         StopMonitor runs every 5min in background
                         FastAPI dashboard at http://localhost:5000
```

**Two markets, two harnesses, one shared aggregator.**

---

## How to Run

```bash
# 1. Set up env (once)
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Fill in API keys
cp .env.example .env
# Edit .env — all keys required before running

# 3. Start
python main.py
```

Dashboard: `http://localhost:5000`  
Telegram: bot starts polling immediately on run.

---

## File Structure & Responsibilities

```
trading-bot/
├── main.py                  — entry point; wires all components; asyncio event loop
├── config.py                — Config dataclass; reads from env vars; config = Config() singleton
│
├── agents/
│   ├── base.py              — abstract BaseAgent; analyze(ticker, klines, market, **kwargs) -> AgentVote
│   ├── kronos_technical.py  — KronosTechnicalAgent; Kronos embed + cosine similarity pattern match
│   │                          (Kronos unavailable → fallback: normalized close embeddings)
│   ├── news_sentiment.py    — NewsSentimentAgent(api_key); CryptoPanic API; score>0.2 LONG, <-0.2 SHORT
│   ├── onchain.py           — OnChainAgent(api_key); CoinGlass funding rate + long/short ratio
│   ├── fundamentals.py      — FundamentalsAgent(); NSE India API; delivery%/P-E signals
│   ├── fii_dii.py           — FIIDIIAgent(); NSE net institutional flow; >500Cr LONG, <-500Cr SHORT
│   └── options_oi.py        — OptionsOIAgent(upstox_broker); PCR + max pain from options chain
│
├── brokers/
│   ├── base_broker.py       — abstract BrokerBase; get_price/place_order/close_position/get_ohlcv
│   ├── coindcx.py           — CoinDCXBroker; HMAC-SHA256 signed requests; market/limit orders
│   └── upstox.py            — UpstoxBroker; Bearer token; options-only (place_order raises NotImplementedError)
│                              ATM strike = round(spot/50)*50; place_options_order() handles F&O
│
├── core/
│   ├── models.py            — Pydantic v2 models: Market/Direction/SignalOutcome enums,
│   │                          AgentVote, Signal (rr_ratio @computed_field), Position, HarnessState
│   │                          Price fields have Field(gt=0.0) constraint
│   ├── debate_engine.py     — DebateEngine; votes filtered at confidence>=0.50; consensus needs
│   │                          >=2 agreeing agents AND avg_confidence>=config.min_confidence (0.65)
│   │                          Below threshold → HOLD, no Telegram message sent
│   ├── risk_manager.py      — RiskManager(market); ATR-based stops: 2×ATR crypto, 1.5×ATR India
│   │                          target = entry + 1.8 * mult * ATR; approve() returns (bool, reason)
│   ├── signal_aggregator.py — SignalAggregator; persists to signal_log.json; 4hr dedup window
│   │                          update_signal_outcome() tracks all signals even unexecuted ones
│   ├── signal_tracker.py    — SignalTracker(aggregator, crypto_broker, india_broker)
│   │                          run_forever(900); resolves PENDING signals: TARGET_HIT/STOP_HIT/EXPIRED
│   │                          crypto: 24hr expiry; india: 3 trading day expiry
│   └── stop_monitor.py      — StopMonitor(crypto_broker, india_broker, harness_states, telegram_bot)
│                              run_forever(300); checks open positions; executes close + Telegram notify
│
├── harnesses/
│   ├── base_harness.py      — BaseHarness; run_session() orchestrates vote→consensus→approve→log→telegram
│   │                          update_learnings() every 10 sessions via claude-haiku; prepends to agent prompts
│   ├── crypto_harness.py    — CryptoHarness; tickers: BTC/ETH/SOL/BNB/XRP; APScheduler interval 15min
│   └── india_harness.py     — IndiaHarness; tickers: Nifty50/NiftyBank/HDFC/Infosys/Reliance
│                              CronTrigger 09:00,11:30,14:45 IST mon-fri
│                              NEXT_WEEKLY_EXPIRY hardcoded — UPDATE EVERY FRIDAY
│
├── tg_bot/
│   └── bot.py               — TelegramBot; 3-button inline keyboard: EXECUTE / SKIP / DEBATE
│                              EXECUTE: India→place_options_order, Crypto→place_order
│                              Package named tg_bot/ (NOT telegram/) — avoids conflict with library
│
├── dashboard/
│   ├── server.py            — FastAPI; REST + WebSocket (/ws/signals); broadcast_signal() for live feed
│   │                          /api/signals  /api/performance  /api/open-positions
│   └── static/
│       ├── index.html       — live signal feed (WebSocket)
│       ├── history.html     — filterable signal history table
│       ├── performance.html — Chart.js portfolio curves; win rate; agent breakdown
│       └── positions.html   — open positions with 30s auto-refresh
│
├── data/
│   ├── crypto_progress.json — harness state: session_count, portfolio_value_inr, open_positions, agent_learnings
│   ├── india_progress.json  — same structure for India market
│   └── signal_log.json      — every signal ever generated + outcome (source of truth for dashboard)
│
├── docs/
│   ├── 2026-05-23-trading-bot-design.md  — full design spec
│   └── 2026-05-23-trading-bot.md         — implementation plan (14 tasks, all complete)
│
├── tests/                   — 25 tests, all passing
│   ├── test_models.py        (4)   — Signal/Position/HarnessState validation
│   ├── test_brokers.py       (5)   — CoinDCX/Upstox HMAC signing, ATM strike calc
│   ├── test_agents.py        (2)   — KronosTechnical pattern match; fallback mode
│   ├── test_debate_engine.py (3)   — consensus rules; HOLD fallback
│   ├── test_signal_aggregator.py (4) — dedup window; outcome update; persistence
│   ├── test_signal_tracker.py (4) — TARGET_HIT/STOP_HIT/EXPIRED resolution
│   └── test_stop_monitor.py  (3)  — long/short close conditions
│
├── requirements.txt
├── .env.example             — all required env var keys
├── .gitignore               — excludes data/*.json, .env, pycache, venv
└── CLAUDE.md                — this file
```

---

## Key Design Decisions & Gotchas

### 1. Package naming: `tg_bot/` not `telegram/`
The folder is `tg_bot/` because a folder named `telegram/` shadows the python-telegram-bot library.  
**Always import as:** `from tg_bot.bot import TelegramBot`

### 2. Harnesses own their brokers
`CryptoHarness` creates `CoinDCXBroker` internally. `IndiaHarness` creates `UpstoxBroker` internally.  
`main.py` accesses them via `crypto_harness.broker` and `india_harness.broker`.  
This is intentional — harnesses are self-contained units.

### 3. TelegramBot chicken-and-egg solved
`TelegramBot` needs `harness_states`, which only exist after harnesses are created.  
Harnesses need `telegram_bot` to send signals.  
**Solution in `build_components()`:** Create harnesses with `telegram_bot=None`, build TelegramBot with live harness_states dict, inject `telegram_bot` back into harnesses.

### 4. Shutdown: background tasks cancelled explicitly
`tracker.run_forever()` and `monitor.run_forever()` are infinite loops. On shutdown, `main.py` calls `.cancel()` on each `asyncio.Task`. The telegram polling thread (`asyncio.to_thread(run_polling)`) cannot be cancelled cross-thread — it finishes when the process exits. This is a known limitation of python-telegram-bot v20 sync polling.

### 5. India harness uses options ONLY
`UpstoxBroker.place_order()` raises `NotImplementedError`. All India trades go through `place_options_order()` which auto-selects ATM strike (nearest 50 for Nifty, 100 for BankNifty).  
LONG signal → ATM Call. SHORT signal → ATM Put.  
Options stop: exit if premium drops 40%. Target: exit if premium doubles or EOD.

### 6. Signal outcome tracking is always-on
Every signal is logged in `signal_log.json` regardless of whether the user executes it.  
`SignalTracker` resolves outcomes for ALL signals — executed and skipped alike.  
This feeds the `/performance` dashboard's "hypothetical P&L" curve.

### 7. Consensus threshold
`DebateEngine.reach_consensus()`:
- Filters out agents with confidence < 0.50
- Requires ≥ 2 agents agreeing on direction
- Requires average confidence of agreeing agents ≥ 0.65 (`config.min_confidence`)
- Below threshold → returns `Direction.HOLD`, no Telegram message sent

### 8. Agent learnings loop
Every 10 sessions, `BaseHarness.update_learnings()` calls `claude-haiku-4-5-20251001` to summarize last 50 signal outcomes into a 3-sentence learning note. Stored in `progress.json["agent_learnings"]`, prepended to each agent's system prompt next session.

### 9. Dashboard reads from file, not memory
`dashboard/server.py` creates its own `SignalAggregator` instance but calls `_load_log()` (reads JSON file) on every request. No in-memory state — always fresh from disk.

---

## What Needs to Happen Before Going Live

### Weekly (every Friday before market close)
Update `NEXT_WEEKLY_EXPIRY` in `harnesses/india_harness.py:23`:
```python
NEXT_WEEKLY_EXPIRY = "2026-05-29"  # change this to next Thursday's date
```

### Before first run
1. Fill all keys in `.env` — bot will start but all broker/API calls will fail silently without them
2. Create Telegram bot via @BotFather, get token, create two channels (`#crypto-signals`, `#india-signals`), get channel IDs
3. Upstox: generate access token (expires daily — needs refresh mechanism for production, currently manual)
4. Verify CoinGlass free tier: 10 req/min — crypto harness runs every 15min across 5 tickers, this is fine

### Upstox token refresh (v2 priority)
Upstox access tokens expire daily. Current code reads `UPSTOX_ACCESS_TOKEN` from env at startup.  
For production: implement OAuth2 refresh flow in `brokers/upstox.py`.

---

## LLM Models Used

| Component | Model | Why |
|-----------|-------|-----|
| Debate engine — consensus reasoning | `claude-opus-4-7` | Needs strongest reasoning for final signal decision |
| Analysis agents — per-agent reasoning | `claude-haiku-4-5-20251001` | High frequency (every 15min × 5 tickers × 3 agents), needs to be fast/cheap |
| Agent learnings summarizer | `claude-haiku-4-5-20251001` | Runs every 10 sessions, straightforward summarization |

---

## Trading Parameters (all in `config.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `min_confidence` | 0.65 | Minimum avg confidence across agreeing agents to publish signal |
| `max_portfolio_pct_per_trade` | 2% | Max portfolio at risk per trade |
| `max_open_positions_per_market` | 3 | Hard cap: no new signal if 3 open in same market |
| `min_rr_ratio` | 1.8 | Minimum risk:reward required to publish |
| `options_stop_pct` | 40% | Exit options if premium drops 40% from entry |
| `options_target_pct` | 100% | Exit options if premium doubles |
| Crypto stop ATR mult | 2.0 | stop = entry − 2×ATR |
| India stop ATR mult | 1.5 | stop = entry − 1.5×ATR |

---

## Running Tests

```bash
cd D:\trading-bot
python -m pytest tests/ -v
# Expected: 25 passed
```

---

## V2 Ideas (out of scope for v1)

- Upstox OAuth2 daily token refresh (currently manual)
- Dynamic expiry fetching (instead of hardcoded `NEXT_WEEKLY_EXPIRY`)
- Fully automated execution without Telegram confirmation (after validating signal quality)
- Trailing stop-loss (StopMonitor currently uses fixed stop)
- Backtesting harness using `signal_log.json` history
- US market screener (separate tool, separate repo)
- Mobile app (currently: Telegram + web dashboard)

---

## Related Docs

- `docs/2026-05-23-trading-bot-design.md` — full design spec with all decisions
- `docs/2026-05-23-trading-bot.md` — 14-task implementation plan (all tasks complete)
