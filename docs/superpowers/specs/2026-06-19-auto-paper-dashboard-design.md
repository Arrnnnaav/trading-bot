# Auto Paper Trading + Dashboard Upgrade Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Telegram entirely, make the bot auto-execute all signals in paper mode, and upgrade the web dashboard to show trades, P&L, and an equity curve.

**Architecture:** The harness generates a signal → immediately calls `paper_broker.place_options_order()` without any human gate → `StopMonitor` manages exits → `signal_log.json` is the single source of truth → dashboard reads it via REST + WebSocket.

**Tech Stack:** Python / FastAPI (existing), vanilla HTML + CSS + Chart.js CDN (no build toolchain), Upstox API for real-time price data (free with account).

---

## Global Constraints

- All changes must keep `python -m pytest tests/ -v` fully green (currently 252 tests passing).
- No new Python dependencies — Chart.js loaded from CDN only.
- `PAPER_TRADING=true` must remain the default; no real orders placed.
- `tg_bot/` folder is NOT deleted — just disconnected from `main.py`.
- Signal log path: `data/signal_log.json` (existing, source of truth for all dashboard data).
- Paper trades path: `data/paper_trades.json` (existing, written by `PaperBroker`).
- Dashboard runs at `http://localhost:5000` (existing FastAPI server).
- Dark theme throughout dashboard (background `#0f1117`, card `#1a1d2e`, accent `#00d4aa`).
- All monetary values displayed in Indian Rupees (₹).
- Upstox free API — no Zerodha Kite Connect dependency for any new code.

---

## Sub-project 1: Kill Telegram + Auto-Execute

### What changes

**`main.py`**
- Remove `TelegramBot` import and instantiation.
- Remove `asyncio.to_thread(telegram_bot.run_polling)` from tasks list.
- Remove `upstox_broker=india_broker` arg (TelegramBot gone).
- Pass `telegram_bot=None` explicitly to both harnesses (already supported by null guard in `base_harness.py`).

**`harnesses/base_harness.py`** — `run_session()` method:

Current (lines 109–111):
```python
if self.telegram_bot is not None:
    await self.telegram_bot.send_signal(signal)
return signal
```

Replace with auto-execute block:
```python
# Auto-execute: paper broker intercepts, logs to paper_trades.json
try:
    expiry = self.broker.resolve_options_expiry() if hasattr(self.broker, "resolve_options_expiry") else None
    order = self.broker.place_options_order(
        index=signal.ticker,
        direction=signal.direction.value,
        expiry=expiry,
        size_inr=signal.position_size_inr,
    )
    # Mark signal as executed in signal_log
    signal.executed = True
    self.aggregator.mark_executed(signal.id, order)
    # Add to open_positions so StopMonitor can track it
    pos = Position(
        id=signal.id,
        signal_id=signal.id,
        ticker=order.get("tradingsymbol", signal.ticker),
        direction=signal.direction,
        entry_price=order.get("fill_price", signal.entry_price),
        stop_price=signal.stop_price,
        target_price=signal.target_price,
        quantity=order.get("quantity", 0),
        broker_order_id=order.get("order_id", ""),
        opened_at=datetime.now(timezone.utc).isoformat(),
        market=self.MARKET,
    )
    self.state.open_positions.append(pos)
    self._save_state()
except Exception as exc:
    _LOG.warning("Auto-execute failed for %s: %s", signal.id, exc)
return signal
```

`Position` and `datetime`/`timezone` already imported in `base_harness.py` — verify imports before implementing.

**`core/signal_aggregator.py`** — add `mark_executed(signal_id, order_info)`:
- Reads `signal_log.json`, finds signal by id, sets `executed=True`, writes back atomically.

**`core/models.py`** — `Signal` already has `executed: bool = False`. No change needed.

### What does NOT change
- `StopMonitor` — already monitors open positions and closes them at target/stop/15:15 IST.
- `PaperBroker` — already intercepts `place_options_order()` and logs to `paper_trades.json`.
- `SignalTracker` — already resolves PENDING signals to TARGET_HIT/STOP_HIT/EXPIRED.
- All agents, harness scheduling, risk gates — unchanged.

---

## Sub-project 2: Dashboard Backend (new API endpoints)

Add to **`dashboard/server.py`**:

### `GET /api/trades`

Returns all executed signals enriched with trade metadata. Source: `signal_log.json`.

```python
@app.get("/api/trades")
async def get_trades(limit: int = 200):
    signals = aggregator.get_all_signals()
    executed = [s for s in signals if s.get("executed")]
    trades = []
    for s in executed[-limit:]:
        pnl = s.get("hypothetical_pnl_inr") or 0.0
        trades.append({
            "id": s["id"],
            "ticker": s["ticker"],
            "direction": s["direction"],
            "entry_price": s["entry_price"],
            "target_price": s["target_price"],
            "stop_price": s["stop_price"],
            "entry_time": s["generated_at"],
            "exit_time": s.get("outcome_at"),
            "outcome": s["outcome"],
            "pnl_inr": round(pnl, 2),
            "confidence": s.get("confidence", 0),
        })
    return trades
```

### `GET /api/equity-curve`

Returns `[(iso_timestamp, cumulative_pnl)]` sorted by `outcome_at`. Only resolved signals.

```python
@app.get("/api/equity-curve")
async def get_equity_curve():
    signals = aggregator.get_all_signals()
    resolved = [
        s for s in signals
        if s.get("executed") and s.get("outcome_at") and s.get("hypothetical_pnl_inr") is not None
    ]
    resolved.sort(key=lambda s: s["outcome_at"])
    cumulative = 0.0
    points = []
    for s in resolved:
        cumulative += s["hypothetical_pnl_inr"]
        points.append({"t": s["outcome_at"], "pnl": round(cumulative, 2)})
    return points
```

### `GET /api/stats`

Aggregated performance stats for the stats cards.

```python
@app.get("/api/stats")
async def get_stats():
    signals = aggregator.get_all_signals()
    executed = [s for s in signals if s.get("executed")]
    resolved = [s for s in executed if s.get("outcome") not in ("PENDING", None)]
    wins = [s for s in resolved if s["outcome"] == "TARGET_HIT"]
    losses = [s for s in resolved if s["outcome"] == "STOP_HIT"]
    win_pnls = [s["hypothetical_pnl_inr"] for s in wins if s.get("hypothetical_pnl_inr")]
    loss_pnls = [s["hypothetical_pnl_inr"] for s in losses if s.get("hypothetical_pnl_inr")]
    total_pnl = sum(s.get("hypothetical_pnl_inr") or 0 for s in resolved)

    # Max drawdown from equity curve
    cumulative, peak, max_dd = 0.0, 0.0, 0.0
    for s in sorted(resolved, key=lambda x: x.get("outcome_at", "")):
        cumulative += s.get("hypothetical_pnl_inr") or 0
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    return {
        "total_trades": len(executed),
        "resolved": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(resolved) * 100, 1) if resolved else 0.0,
        "total_pnl_inr": round(total_pnl, 2),
        "avg_win_inr": round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0,
        "avg_loss_inr": round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0,
        "max_drawdown_inr": round(max_dd, 2),
    }
```

---

## Sub-project 3: Dashboard Frontend

All pages share a common dark theme CSS block. Chart.js loaded from:
`https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`

### `dashboard/static/index.html` — Overview

Layout:
```
[Nav: Overview | Trades | Performance]   [● LIVE  TIME  PAPER]

[Portfolio ₹X]  [Today P&L ₹X (+Y%)]  [Open Positions N]

OPEN POSITIONS
[card: NIFTY LONG CE | Entry ₹120 | Est P&L: +₹480 (green)]
[card: BANKNIFTY SHORT PE | Entry ₹95 | Est P&L: -₹30 (red)]

LIVE SIGNAL FEED
[09:20  NIFTY LONG  conf=78%  ✅ AUTO-EXECUTED]
[09:20  SENSEX HOLD  —]
```

- Portfolio value: from `data/india_progress.json` → `portfolio_value_inr`
- Open positions: `GET /api/open-positions` (existing endpoint) — shows entry price only; no "Est P&L" (option premium not polled in real-time during paper mode — show status as "OPEN")
- Today P&L: sum of `pnl_inr` from `/api/trades` where `entry_time` is today
- Live feed: WebSocket `/ws/signals` — appends new row on each signal event
- Polls `/api/open-positions` every 30s

### `dashboard/static/trades.html` — Trade History

Layout:
```
[Nav]  [Filter: All | LONG | SHORT]  [Outcome: All | ✅ | ❌ | ⏳]

Time Opened     | Ticker      | Dir   | Entry | Exit  | Duration | P&L      | Outcome
2026-06-19 9:20 | NIFTY       | LONG  | ₹120  | ₹240  | 4h 20m   | +₹7,800  | TARGET_HIT ✅
2026-06-19 9:20 | BANKNIFTY   | SHORT | ₹95   | ₹61   | 2h 05m   | -₹2,850  | STOP_HIT ❌

[Showing 1–50 of N trades]  [← Prev]  [Next →]
```

- Source: `GET /api/trades`
- P&L cells: green if positive, red if negative
- Outcome badges: `TARGET_HIT` = green pill, `STOP_HIT` = red pill, `PENDING` = gray pill
- Duration: computed from `entry_time` to `exit_time` client-side
- Client-side filter (no server round-trip)
- Pagination: 50 rows per page, client-side

### `dashboard/static/performance.html` — Performance

Layout:
```
[Nav]

[Win Rate 62%]  [Total P&L +₹48,200]  [Avg Win +₹3,100]  [Avg Loss -₹1,800]  [Max DD ₹5,200]

EQUITY CURVE
[Chart.js line chart — x: date, y: cumulative P&L ₹]

WIN / LOSS SPLIT
[Chart.js donut — green=wins, red=losses]
```

- Equity curve: `GET /api/equity-curve` → Chart.js Line chart
  - x-axis: dates (formatted `DD MMM`)
  - y-axis: ₹ cumulative P&L
  - Line color: `#00d4aa` (teal accent)
  - Fill: light teal gradient below line
- Donut: `GET /api/stats` → `wins` vs `losses` count
- Stat cards: from `GET /api/stats`
- Refreshes every 60s

---

## Shared CSS (inline in each HTML `<style>` block)

```css
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f1117; color: #e0e0e0; font-family: 'Inter', monospace; }
.nav { display: flex; gap: 24px; padding: 14px 24px; background: #1a1d2e; border-bottom: 1px solid #2a2d3e; }
.nav a { color: #8888aa; text-decoration: none; font-size: 14px; }
.nav a.active { color: #00d4aa; border-bottom: 2px solid #00d4aa; padding-bottom: 2px; }
.card { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 8px; padding: 20px; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 20px 24px; }
.stat-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
.stat-value { font-size: 24px; font-weight: 700; margin-top: 4px; }
.positive { color: #00d4aa; }
.negative { color: #ff4d6d; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.badge-win { background: #00d4aa22; color: #00d4aa; }
.badge-loss { background: #ff4d6d22; color: #ff4d6d; }
.badge-pending { background: #88888822; color: #888; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; padding: 8px 12px; border-bottom: 1px solid #2a2d3e; text-align: left; }
td { padding: 10px 12px; border-bottom: 1px solid #1e2130; }
tr:hover { background: #1e2130; }
.live-dot { width: 8px; height: 8px; border-radius: 50%; background: #00d4aa; display: inline-block; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `main.py` | Remove TelegramBot; pass `telegram_bot=None` to harnesses |
| `harnesses/base_harness.py` | Replace telegram send with auto-execute block |
| `core/signal_aggregator.py` | Add `mark_executed(signal_id, order_info)` method |
| `dashboard/server.py` | Add `/api/trades`, `/api/equity-curve`, `/api/stats` endpoints |
| `dashboard/static/index.html` | Rebuild: stat cards + open positions + live feed |
| `dashboard/static/trades.html` | Rebuild: trade table with filter + pagination |
| `dashboard/static/performance.html` | Rebuild: equity curve + donut + stat cards |
| `dashboard/static/positions.html` | Delete (merged into index.html) |
| `dashboard/server.py` `/positions` route | Remove or redirect to `/` |
| `tests/test_dashboard.py` | New: test 3 new API endpoints |
| `tests/test_auto_execute.py` | New: test auto-execute flow in base_harness |

---

## Testing

```bash
python -m pytest tests/ -v --basetemp=.pytest-tmp
# Expected: all existing 252 tests pass + new tests green
```

Manual smoke test:
1. `python main.py` — confirm no Telegram errors on startup
2. Open `http://localhost:5000` — overview page loads, shows stat cards
3. Open `http://localhost:5000/trades` — table loads (empty if no signals yet)
4. Open `http://localhost:5000/performance` — equity curve renders (empty state message if no data)
5. Trigger a test signal — confirm it appears in live feed on overview without any Telegram prompt
