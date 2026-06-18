# Auto Paper Trading + Dashboard Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Telegram entirely, make the bot auto-execute all signals in paper mode without human confirmation, and rebuild the web dashboard with trade history, P&L, and an equity curve.

**Architecture:** `base_harness.run_session()` calls `paper_broker.place_options_order()` immediately after signal approval, then creates a `Position` in `state.open_positions`. `StopMonitor` manages exits as before. The FastAPI dashboard at `localhost:5000` gains three new API endpoints feeding rebuilt HTML pages with Chart.js charts.

**Tech Stack:** Python / FastAPI (existing), vanilla HTML + CSS + JavaScript, Chart.js 4.4.0 (CDN), Upstox API for prices (existing, free).

## Global Constraints

- All changes must keep `python -m pytest tests/ -v` fully green (252 tests currently passing).
- No new Python pip dependencies — Chart.js loaded from CDN only.
- `PAPER_TRADING=true` in `.env` — no real orders ever placed.
- `tg_bot/` folder is NOT deleted — just disconnected from `main.py`.
- Signal log: `data/signal_log.json` — single source of truth for all dashboard data.
- Paper trades log: `data/paper_trades.json` — written by `PaperBroker`.
- Dashboard URL: `http://localhost:5000`.
- Dark theme: background `#0f1117`, card background `#1a1d2e`, accent `#00d4aa`, danger `#ff4d6d`.
- All monetary values in Indian Rupees with `₹` prefix.
- `signal_aggregator.mark_signal_executed(signal_id: str) -> bool` already exists — use it.

---

### Task 1: Kill Telegram + add null guards to StopMonitor

**Files:**
- Modify: `main.py`
- Modify: `core/stop_monitor.py` (lines ~137 and ~182)
- Test: `tests/test_stop_monitor.py` (add 1 null-guard test)

**Interfaces:**
- Produces: `main.py` that builds without `TelegramBot`; `StopMonitor` that accepts `telegram_bot=None`

- [ ] **Step 1: Read stop_monitor.py lines 130–200 to confirm exact telegram call sites**

```bash
sed -n '130,200p' core/stop_monitor.py
```

Expected: two call sites — one unconditional `await self.telegram_bot.send_error(...)` around line 137, and one guarded with `hasattr` around lines 182–192.

- [ ] **Step 2: Write a failing test for null telegram_bot in StopMonitor**

Add to `tests/test_stop_monitor.py`:

```python
@pytest.mark.asyncio
async def test_close_position_tolerates_none_telegram_bot(tmp_path):
    """StopMonitor must not crash when telegram_bot is None."""
    from core.models import Market, Direction, HarnessState
    state = HarnessState()
    pos = make_position(entry=22400, stop=22000, target=23200)
    state.open_positions.append(pos)

    broker = MagicMock()
    broker.get_price.return_value = 21900.0
    broker.close_position.return_value = {"order_id": "ord_close", "status": "closed"}

    aggregator = MagicMock()
    state_path = str(tmp_path / "state.json")

    monitor = StopMonitor(
        india_broker=broker,
        harness_states={Market.INDIA: state},
        telegram_bot=None,           # <— key: None, not a mock
        signal_aggregator=aggregator,
        intraday_force_exit_time="23:59",
    )
    monitor._state_paths = {Market.INDIA: state_path}

    # Must not raise AttributeError
    await monitor._close_position(pos, 21900.0, Market.INDIA)
```

Run: `pytest tests/test_stop_monitor.py::test_close_position_tolerates_none_telegram_bot -v`
Expected: FAIL with `AttributeError: 'NoneType' object has no attribute 'send_error'`

- [ ] **Step 3: Add null guards to stop_monitor.py**

Find the line with `await self.telegram_bot.send_error(` (around line 137). Wrap it:

```python
        except Exception as exc:
            _LOG.error(
                "Broker close failed for %s — position left open: %s",
                position.ticker,
                exc,
            )
            if self.telegram_bot is not None:
                await self.telegram_bot.send_error(
                    f"Stop close failed for {position.ticker}: {exc}"
                )
            return
```

Find the Step 4 telegram notify block (around lines 180–196). Wrap the outer try with a None check:

```python
        # Step 4: notify Telegram (optional — None when running without Telegram).
        if self.telegram_bot is not None:
            try:
                if hasattr(self.telegram_bot, "send_position_closed"):
                    await self.telegram_bot.send_position_closed(
                        ticker=position.ticker,
                        close_price=current_price,
                        pnl_pct=pnl,
                        pnl_inr=pnl_inr,
                        market=market,
                        reason=reason,
                    )
                else:
                    await self.telegram_bot.send_stop_hit(
                        ticker=position.ticker,
                        close_price=current_price,
                        pnl_pct=pnl,
                        pnl_inr=pnl_inr,
                        market=market,
                    )
            except Exception as exc:
                _LOG.warning("Telegram notify failed for %s: %s", position.ticker, exc)
```

- [ ] **Step 4: Run the new test — expect PASS**

```bash
pytest tests/test_stop_monitor.py::test_close_position_tolerates_none_telegram_bot -v
```

- [ ] **Step 5: Remove TelegramBot from main.py**

Replace the current `main.py` `build_components()` function. The key changes:
- Remove `from tg_bot.bot import TelegramBot` import
- Remove `telegram_bot = TelegramBot(...)` creation
- Remove `india_intraday_harness.telegram_bot = telegram_bot` assignment
- Remove `india_positional_harness.telegram_bot = telegram_bot` assignment
- Return `None` for telegram_bot in the tuple
- In `main()`, remove `asyncio.to_thread(telegram_bot.run_polling)` task
- Pass `telegram_bot=None` to `StopMonitor`

New `build_components()` (replace entire function):

```python
def build_components():
    errors = config.validate_for_runtime()
    if errors:
        raise RuntimeError("Invalid runtime configuration:\n- " + "\n- ".join(errors))

    if config.run_startup_calibration:
        from training.calibrate_thresholds import calibrate
        calibrate()

    aggregator = SignalAggregator(signal_log_path=config.signal_log_path)
    harness_states: dict = {}

    india_intraday_harness = None
    india_positional_harness = None
    india_broker = None

    if config.enable_india:
        if config.enable_intraday:
            from harnesses.india_intraday_harness import IndiaIntradayHarness
            india_intraday_harness = IndiaIntradayHarness(
                telegram_bot=None, signal_aggregator=aggregator
            )
            harness_states[Market.INDIA] = india_intraday_harness.state
            india_broker = india_intraday_harness.broker

        if config.enable_positional:
            from harnesses.india_positional_harness import IndiaPositionalHarness
            india_positional_harness = IndiaPositionalHarness(
                telegram_bot=None, signal_aggregator=aggregator
            )
            if Market.INDIA not in harness_states:
                harness_states[Market.INDIA] = india_positional_harness.state
            if india_broker is None:
                india_broker = india_positional_harness.broker

    if india_broker is None and config.enable_india:
        raise RuntimeError(
            "ENABLE_INDIA=true but both ENABLE_INTRADAY and ENABLE_POSITIONAL are false."
        )

    return (
        india_intraday_harness,
        india_positional_harness,
        aggregator,
        harness_states,
        india_broker,
    )
```

New `main()` function — replace the task list and unpacking:

```python
async def main():
    (
        india_intraday_harness,
        india_positional_harness,
        aggregator,
        harness_states,
        india_broker,
    ) = build_components()

    data_scheduler = _build_data_scheduler()

    print("=" * 50)
    print("Trading Bot v2 — India Index Options")
    print(f"  Dashboard:  http://{config.dashboard_host}:{config.dashboard_port}")
    print(f"  Mode:       {'PAPER TRADING' if config.paper_trading else 'LIVE'}")
    print(f"  Telegram:   disabled (auto-execute mode)")
    print(f"  Strategy:   {config.strategy_profile}")
    print(
        f"  Harnesses:  "
        f"{'intraday ' if config.enable_intraday else ''}"
        f"{'positional' if config.enable_positional else ''}"
    )
    print("=" * 50)

    if india_intraday_harness:
        india_intraday_harness.start()
    if india_positional_harness:
        india_positional_harness.start()

    data_scheduler.start()

    tracker = SignalTracker(aggregator, india_broker)
    monitor = StopMonitor(
        india_broker=india_broker,
        harness_states=harness_states,
        telegram_bot=None,
        signal_aggregator=aggregator,
        intraday_force_exit_time=config.intraday_force_exit,
    )

    server = uvicorn.Server(
        uvicorn.Config(
            app=dashboard_app,
            host=config.dashboard_host,
            port=config.dashboard_port,
            log_level="warning",
        )
    )

    tasks = [
        asyncio.create_task(tracker.run_forever(900)),
        asyncio.create_task(monitor.run_forever(300)),
    ]

    try:
        await server.serve()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if data_scheduler.running:
            data_scheduler.shutdown(wait=False)
        if india_intraday_harness and india_intraday_harness.scheduler.running:
            india_intraday_harness.scheduler.shutdown(wait=False)
        if india_positional_harness and india_positional_harness.scheduler.running:
            india_positional_harness.scheduler.shutdown(wait=False)
        print("Trading Bot shut down.")
```

Also remove the `from tg_bot.bot import TelegramBot` import at the top of `main.py`.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -q --basetemp=.pytest-tmp
```

Expected: all 252 tests pass. If any fail with `TelegramBot`, check that test fixtures still pass `telegram_bot=MagicMock()` — those tests test `tg_bot/` directly and are unchanged.

- [ ] **Step 7: Commit**

```bash
git add main.py core/stop_monitor.py tests/test_stop_monitor.py
git commit -m "feat: remove Telegram, add StopMonitor null guards for telegram_bot=None"
```

---

### Task 2: Auto-execute signals in base_harness

**Files:**
- Modify: `harnesses/base_harness.py`
- Create: `tests/test_auto_execute.py`

**Interfaces:**
- Consumes: `Position` from `core.models` (fields: `signal_id`, `ticker`, `market`, `direction`, `entry_price`, `stop_price`, `target_price`, `size_inr`, `quantity`, `opened_at`, `broker_order_id`, `strike`, `expiry`, `option_type`)
- Consumes: `aggregator.mark_signal_executed(signal_id: str) -> bool` (already exists)
- Produces: `run_session()` that auto-executes and adds `Position` to `self.state.open_positions`

- [ ] **Step 1: Write failing tests**

Create `tests/test_auto_execute.py`:

```python
import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from core.models import Direction, Market, Signal, HarnessState


def _make_signal(ticker="NIFTY", direction=Direction.LONG):
    return Signal(
        id="sig_test_auto_001",
        market=Market.INDIA,
        ticker=ticker,
        direction=direction,
        entry_price=23000.0,
        target_price=46000.0,
        stop_price=14950.0,
        confidence=0.75,
        position_size_inr=2000.0,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def auto_harness(tmp_path):
    patches = {
        "harnesses.base_harness.DebateEngine": MagicMock,
        "harnesses.base_harness.RiskManager": MagicMock,
        "harnesses.base_harness.AgentPerformanceTracker": MagicMock,
        "harnesses.base_harness.PatternReputationTracker": MagicMock,
        "harnesses.base_harness.ClaudeCodeClient": MagicMock,
    }
    with patch.multiple("harnesses.base_harness", **patches):
        from harnesses.base_harness import BaseHarness

        class ConcreteHarness(BaseHarness):
            MARKET = Market.INDIA

        broker = MagicMock()
        broker.place_options_order.return_value = {
            "order_id": "paper-opt-test123",
            "status": "paper",
            "quantity": 65,
            "expiry": "2026-06-26",
            "strike": 23000.0,
        }
        broker.resolve_options_expiry.return_value = "2026-06-26"

        aggregator = MagicMock()
        aggregator.is_duplicate.return_value = False
        aggregator.get_all_signals.return_value = []

        harness = ConcreteHarness(
            state_path=str(tmp_path / "state.json"),
            broker=broker,
            agents=[],
            telegram_bot=None,
            signal_aggregator=aggregator,
        )
        return harness, broker, aggregator


@pytest.mark.asyncio
async def test_run_session_calls_place_options_order(auto_harness):
    harness, broker, aggregator = auto_harness
    signal = _make_signal()
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.LONG, "confidence": 0.75, "transcript": ""}
    )
    harness.risk_manager.approve.return_value = (True, "ok")
    harness.risk_manager.build_signal.return_value = signal

    await harness.run_session("NIFTY", [])

    broker.place_options_order.assert_called_once()
    call_kwargs = broker.place_options_order.call_args[1]
    assert call_kwargs["index"] == "NIFTY"
    assert call_kwargs["direction"] == "LONG"
    assert call_kwargs["size_inr"] == 2000.0


@pytest.mark.asyncio
async def test_run_session_adds_position_to_state(auto_harness):
    harness, broker, aggregator = auto_harness
    signal = _make_signal()
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.LONG, "confidence": 0.75, "transcript": ""}
    )
    harness.risk_manager.approve.return_value = (True, "ok")
    harness.risk_manager.build_signal.return_value = signal

    await harness.run_session("NIFTY", [])

    assert len(harness.state.open_positions) == 1
    pos = harness.state.open_positions[0]
    assert pos.ticker == "NIFTY"
    assert pos.direction == Direction.LONG
    assert pos.signal_id == "sig_test_auto_001"
    assert pos.broker_order_id == "paper-opt-test123"


@pytest.mark.asyncio
async def test_run_session_marks_signal_executed(auto_harness):
    harness, broker, aggregator = auto_harness
    signal = _make_signal()
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.LONG, "confidence": 0.75, "transcript": ""}
    )
    harness.risk_manager.approve.return_value = (True, "ok")
    harness.risk_manager.build_signal.return_value = signal

    await harness.run_session("NIFTY", [])

    aggregator.mark_signal_executed.assert_called_once_with("sig_test_auto_001")


@pytest.mark.asyncio
async def test_run_session_survives_broker_exception(auto_harness):
    """Broker crash must not propagate — signal is still returned."""
    harness, broker, aggregator = auto_harness
    broker.place_options_order.side_effect = RuntimeError("API down")
    signal = _make_signal()
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.LONG, "confidence": 0.75, "transcript": ""}
    )
    harness.risk_manager.approve.return_value = (True, "ok")
    harness.risk_manager.build_signal.return_value = signal

    result = await harness.run_session("NIFTY", [])

    assert result is not None  # signal returned even though broker failed
    assert len(harness.state.open_positions) == 0  # position NOT added on failure


@pytest.mark.asyncio
async def test_run_session_hold_skips_execution(auto_harness):
    harness, broker, aggregator = auto_harness
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.HOLD, "confidence": 0.5, "transcript": ""}
    )

    result = await harness.run_session("NIFTY", [])

    assert result is None
    broker.place_options_order.assert_not_called()
```

Run: `pytest tests/test_auto_execute.py -v`
Expected: 5 FAILs (auto-execute logic not implemented yet)

- [ ] **Step 2: Update base_harness.py imports**

In `harnesses/base_harness.py`, change line:
```python
from core.models import HarnessState, Direction
```
to:
```python
from core.models import HarnessState, Direction, Position
```

- [ ] **Step 3: Replace telegram send with auto-execute in run_session()**

In `harnesses/base_harness.py`, find this block (around lines 109–111):

```python
        if self.telegram_bot is not None:
            await self.telegram_bot.send_signal(signal)
        return signal
```

Replace with:

```python
        # Auto-execute: paper broker intercepts, logs to data/paper_trades.json.
        try:
            expiry = (
                self.broker.resolve_options_expiry()
                if hasattr(self.broker, "resolve_options_expiry")
                else None
            )
            order = self.broker.place_options_order(
                index=signal.ticker,
                direction=signal.direction.value,
                expiry=expiry,
                size_inr=signal.position_size_inr,
            )
            self.aggregator.mark_signal_executed(signal.id)
            pos = Position(
                signal_id=signal.id,
                ticker=signal.ticker,
                market=self.MARKET,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                size_inr=signal.position_size_inr,
                quantity=order.get("quantity"),
                opened_at=datetime.now(timezone.utc).isoformat(),
                broker_order_id=order.get("order_id", ""),
                strike=order.get("strike"),
                expiry=order.get("expiry"),
                option_type="CE" if signal.direction == Direction.LONG else "PE",
            )
            self.state.open_positions.append(pos)
            self._save_state()
        except Exception as exc:
            _LOG.warning("Auto-execute failed for %s: %s", signal.id, exc)
        return signal
```

- [ ] **Step 4: Run the new tests — expect all 5 PASS**

```bash
pytest tests/test_auto_execute.py -v
```

Expected:
```
PASSED test_run_session_calls_place_options_order
PASSED test_run_session_adds_position_to_state
PASSED test_run_session_marks_signal_executed
PASSED test_run_session_survives_broker_exception
PASSED test_run_session_hold_skips_execution
```

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -q --basetemp=.pytest-tmp
```

Expected: all tests pass (257+ total).

- [ ] **Step 6: Commit**

```bash
git add harnesses/base_harness.py tests/test_auto_execute.py
git commit -m "feat: auto-execute all signals in paper mode, no Telegram gate"
```

---

### Task 3: Dashboard API — three new endpoints

**Files:**
- Modify: `dashboard/server.py`
- Create: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `aggregator.get_all_signals() -> list[dict]` (existing)
- Produces:
  - `GET /api/trades` → `list[dict]` with keys: `id`, `ticker`, `direction`, `entry_price`, `target_price`, `stop_price`, `entry_time`, `exit_time`, `outcome`, `pnl_inr`, `confidence`
  - `GET /api/equity-curve` → `list[dict]` with keys: `t` (ISO timestamp), `pnl` (float, cumulative)
  - `GET /api/stats` → `dict` with keys: `total_trades`, `resolved`, `wins`, `losses`, `win_rate`, `total_pnl_inr`, `avg_win_inr`, `avg_loss_inr`, `max_drawdown_inr`

- [ ] **Step 1: Write failing tests**

Create `tests/test_dashboard.py`:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import dashboard.server as server_module
from dashboard.server import app

client = TestClient(app)

_SIGNAL_TARGET_HIT = {
    "id": "sig_001",
    "ticker": "NIFTY",
    "market": "india",
    "direction": "LONG",
    "entry_price": 23000.0,
    "target_price": 46000.0,
    "stop_price": 14950.0,
    "confidence": 0.75,
    "position_size_inr": 2000.0,
    "generated_at": "2026-06-19T09:20:00+00:00",
    "executed": True,
    "outcome": "TARGET_HIT",
    "outcome_at": "2026-06-19T13:15:00+00:00",
    "hypothetical_pnl_inr": 4000.0,
    "hypothetical_pnl_pct": 2.0,
    "rr_ratio": 1.8,
    "agent_votes": [],
    "debate_transcript": "",
}

_SIGNAL_STOP_HIT = {
    **_SIGNAL_TARGET_HIT,
    "id": "sig_002",
    "ticker": "BANKNIFTY",
    "direction": "SHORT",
    "outcome": "STOP_HIT",
    "outcome_at": "2026-06-19T11:00:00+00:00",
    "hypothetical_pnl_inr": -1500.0,
    "hypothetical_pnl_pct": -0.75,
}

_SIGNAL_PENDING = {
    **_SIGNAL_TARGET_HIT,
    "id": "sig_003",
    "outcome": "PENDING",
    "outcome_at": None,
    "hypothetical_pnl_inr": None,
}

_SIGNAL_NOT_EXECUTED = {
    **_SIGNAL_TARGET_HIT,
    "id": "sig_004",
    "executed": False,
}


def test_api_trades_returns_only_executed():
    signals = [_SIGNAL_TARGET_HIT, _SIGNAL_NOT_EXECUTED]
    with patch.object(server_module.aggregator, "get_all_signals", return_value=signals):
        resp = client.get("/api/trades")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "sig_001"


def test_api_trades_fields():
    sig = {**_SIGNAL_TARGET_HIT, "outcome_price": 46000.0}
    with patch.object(server_module.aggregator, "get_all_signals", return_value=[sig]):
        resp = client.get("/api/trades")
    trade = resp.json()[0]
    assert trade["ticker"] == "NIFTY"
    assert trade["direction"] == "LONG"
    assert trade["pnl_inr"] == 4000.0
    assert trade["outcome"] == "TARGET_HIT"
    assert trade["entry_time"] == "2026-06-19T09:20:00+00:00"
    assert trade["exit_time"] == "2026-06-19T13:15:00+00:00"
    assert trade["exit_price"] == 46000.0


def test_api_trades_empty():
    with patch.object(server_module.aggregator, "get_all_signals", return_value=[]):
        resp = client.get("/api/trades")
    assert resp.json() == []


def test_api_equity_curve_cumulative_order():
    signals = [_SIGNAL_TARGET_HIT, _SIGNAL_STOP_HIT]
    with patch.object(server_module.aggregator, "get_all_signals", return_value=signals):
        resp = client.get("/api/equity-curve")
    assert resp.status_code == 200
    points = resp.json()
    assert len(points) == 2
    assert points[0]["pnl"] == 4000.0   # first resolved: +4000
    assert points[1]["pnl"] == 2500.0   # cumulative: 4000 + (-1500)


def test_api_equity_curve_skips_pending():
    signals = [_SIGNAL_TARGET_HIT, _SIGNAL_PENDING]
    with patch.object(server_module.aggregator, "get_all_signals", return_value=signals):
        resp = client.get("/api/equity-curve")
    points = resp.json()
    assert len(points) == 1   # pending excluded


def test_api_equity_curve_empty():
    with patch.object(server_module.aggregator, "get_all_signals", return_value=[]):
        resp = client.get("/api/equity-curve")
    assert resp.json() == []


def test_api_stats_win_loss_counts():
    signals = [_SIGNAL_TARGET_HIT, _SIGNAL_STOP_HIT]
    with patch.object(server_module.aggregator, "get_all_signals", return_value=signals):
        resp = client.get("/api/stats")
    data = resp.json()
    assert data["wins"] == 1
    assert data["losses"] == 1
    assert data["win_rate"] == 50.0
    assert data["total_pnl_inr"] == 2500.0
    assert data["avg_win_inr"] == 4000.0
    assert data["avg_loss_inr"] == -1500.0


def test_api_stats_empty():
    with patch.object(server_module.aggregator, "get_all_signals", return_value=[]):
        resp = client.get("/api/stats")
    data = resp.json()
    assert data["win_rate"] == 0.0
    assert data["total_pnl_inr"] == 0.0
    assert data["wins"] == 0


def test_api_stats_max_drawdown():
    # two wins then a loss — max_dd = peak(5500) - trough(4000) = 1500
    s1 = {**_SIGNAL_TARGET_HIT, "id": "s1", "outcome_at": "2026-06-19T09:00:00+00:00", "hypothetical_pnl_inr": 3000.0, "outcome": "TARGET_HIT"}
    s2 = {**_SIGNAL_TARGET_HIT, "id": "s2", "outcome_at": "2026-06-19T10:00:00+00:00", "hypothetical_pnl_inr": 2500.0, "outcome": "TARGET_HIT"}
    s3 = {**_SIGNAL_TARGET_HIT, "id": "s3", "outcome_at": "2026-06-19T11:00:00+00:00", "hypothetical_pnl_inr": -1500.0, "outcome": "STOP_HIT"}
    with patch.object(server_module.aggregator, "get_all_signals", return_value=[s1, s2, s3]):
        resp = client.get("/api/stats")
    assert resp.json()["max_drawdown_inr"] == 1500.0
```

Run: `pytest tests/test_dashboard.py -v`
Expected: all tests FAIL (endpoints not implemented yet)

- [ ] **Step 2: Add three endpoints to dashboard/server.py**

Add after the existing `get_open_positions` endpoint:

```python
@app.get("/api/trades")
async def get_trades(limit: int = 200):
    signals = aggregator.get_all_signals()
    executed = [s for s in signals if s.get("executed")]
    trades = []
    for s in executed[-limit:]:
        trades.append({
            "id": s["id"],
            "ticker": s["ticker"],
            "direction": s["direction"],
            "entry_price": s["entry_price"],
            "target_price": s["target_price"],
            "stop_price": s["stop_price"],
            "entry_time": s.get("generated_at"),
            "exit_time": s.get("outcome_at"),
            "exit_price": s.get("outcome_price"),
            "outcome": s.get("outcome", "PENDING"),
            "pnl_inr": round(s.get("hypothetical_pnl_inr") or 0.0, 2),
            "confidence": s.get("confidence", 0.0),
        })
    return trades


@app.get("/api/equity-curve")
async def get_equity_curve():
    signals = aggregator.get_all_signals()
    resolved = [
        s for s in signals
        if s.get("executed")
        and s.get("outcome_at")
        and s.get("hypothetical_pnl_inr") is not None
        and s.get("outcome") not in ("PENDING", None)
    ]
    resolved.sort(key=lambda s: s["outcome_at"])
    cumulative = 0.0
    points = []
    for s in resolved:
        cumulative += s["hypothetical_pnl_inr"]
        points.append({"t": s["outcome_at"], "pnl": round(cumulative, 2)})
    return points


@app.get("/api/stats")
async def get_stats():
    signals = aggregator.get_all_signals()
    executed = [s for s in signals if s.get("executed")]
    resolved = [s for s in executed if s.get("outcome") not in ("PENDING", None)]
    wins = [s for s in resolved if s["outcome"] == "TARGET_HIT"]
    losses = [s for s in resolved if s["outcome"] == "STOP_HIT"]
    win_pnls = [s["hypothetical_pnl_inr"] for s in wins if s.get("hypothetical_pnl_inr") is not None]
    loss_pnls = [s["hypothetical_pnl_inr"] for s in losses if s.get("hypothetical_pnl_inr") is not None]
    total_pnl = sum(s.get("hypothetical_pnl_inr") or 0 for s in resolved)

    cumulative, peak, max_dd = 0.0, 0.0, 0.0
    for s in sorted(resolved, key=lambda x: x.get("outcome_at") or ""):
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

Also add a redirect for the old `/positions` route (being deleted in Task 6):

```python
from fastapi.responses import RedirectResponse

@app.get("/positions")
async def positions_redirect():
    return RedirectResponse(url="/", status_code=301)
```

Add the import at the top of `server.py` (with existing imports):
```python
from fastapi.responses import HTMLResponse, RedirectResponse
```

- [ ] **Step 3: Run dashboard tests — expect all PASS**

```bash
pytest tests/test_dashboard.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 4: Run full suite**

```bash
python -m pytest tests/ -q --basetemp=.pytest-tmp
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/server.py tests/test_dashboard.py
git commit -m "feat: add /api/trades, /api/equity-curve, /api/stats dashboard endpoints"
```

---

### Task 4: Rebuild index.html — Overview page

**Files:**
- Modify: `dashboard/static/index.html` (full rewrite)

**Interfaces:**
- Consumes: `GET /api/open-positions` (existing), `GET /api/trades` (Task 3), `GET /api/performance` (existing)
- Consumes: `WebSocket /ws/signals` for live feed

No automated tests for HTML. Verified manually by loading `http://localhost:5000`.

- [ ] **Step 1: Replace dashboard/static/index.html with this complete file**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot — Overview</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}
nav{display:flex;align-items:center;gap:20px;padding:0 24px;height:52px;background:#1a1d2e;border-bottom:1px solid #2a2d3e}
.brand{font-weight:700;color:#fff;font-size:15px;margin-right:8px}
nav a{color:#8888aa;text-decoration:none;font-size:13px;padding-bottom:3px}
nav a.active{color:#00d4aa;border-bottom:2px solid #00d4aa}
.spacer{flex:1}
.live-status{display:flex;align-items:center;gap:6px;font-size:12px;color:#00d4aa}
.live-dot{width:7px;height:7px;border-radius:50%;background:#00d4aa;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.paper-badge{background:#ff8c0018;color:#ff8c00;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;border:1px solid #ff8c0033}
#clock{font-size:11px;color:#555;font-family:monospace}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;padding:20px 24px}
.card{background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:18px 20px}
.stat-label{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.stat-value{font-size:26px;font-weight:700}
.pos{color:#00d4aa}.neg{color:#ff4d6d}.neu{color:#e0e0e0}
section{padding:0 24px 20px}
section h2{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
.pos-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.pos-card{background:#1a1d2e;border:1px solid #2a2d3e;border-radius:6px;padding:14px 16px}
.pos-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.pos-ticker{font-weight:700;font-size:15px}
.dir{font-size:11px;font-weight:600;padding:2px 7px;border-radius:3px}
.dir-l{background:#00d4aa18;color:#00d4aa}.dir-s{background:#ff4d6d18;color:#ff4d6d}
.pos-line{font-size:12px;color:#666;margin-top:3px}
.feed-wrap{background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;overflow:hidden}
.feed-head{display:grid;grid-template-columns:65px 90px 60px 1fr 130px;padding:10px 16px;border-bottom:1px solid #2a2d3e;font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px}
.feed-row{display:grid;grid-template-columns:65px 90px 60px 1fr 130px;padding:10px 16px;border-bottom:1px solid #1e2130;font-size:12px;font-family:monospace}
.feed-row:last-child{border-bottom:none}
.ft{color:#555}.fl{color:#00d4aa}.fs{color:#ff4d6d}.fh{color:#444}
.fex{color:#00d4aa;text-align:right}.fno{color:#444;text-align:right}
.empty{padding:28px;text-align:center;color:#444;font-size:13px}
</style>
</head>
<body>
<nav>
  <span class="brand">Trading Bot</span>
  <a href="/" class="active">Overview</a>
  <a href="/trades">Trades</a>
  <a href="/performance">Performance</a>
  <span class="spacer"></span>
  <div class="live-status"><div class="live-dot"></div>LIVE</div>
  <span id="clock"></span>
  <span class="paper-badge">PAPER</span>
</nav>

<div class="stat-grid">
  <div class="card"><div class="stat-label">Portfolio Value</div><div class="stat-value neu" id="stat-port">₹1,00,000</div></div>
  <div class="card"><div class="stat-label">Today P&amp;L</div><div class="stat-value neu" id="stat-today">₹—</div></div>
  <div class="card"><div class="stat-label">Open Positions</div><div class="stat-value neu" id="stat-open">—</div></div>
  <div class="card"><div class="stat-label">Signals Today</div><div class="stat-value neu" id="stat-sigs">—</div></div>
</div>

<section>
  <h2>Open Positions</h2>
  <div class="pos-grid" id="pos-grid"><div class="empty">No open positions</div></div>
</section>

<section>
  <h2>Live Signal Feed</h2>
  <div class="feed-wrap">
    <div class="feed-head"><span>Time</span><span>Ticker</span><span>Dir</span><span>Conf</span><span style="text-align:right">Status</span></div>
    <div id="feed-body"><div class="empty">Waiting for signals…</div></div>
  </div>
</section>

<script>
const IST = 5.5 * 3600000;
const fmt = v => v == null ? '—' : (v >= 0 ? '+₹' : '-₹') + Math.abs(v).toLocaleString('en-IN', {maximumFractionDigits:0});
const fmtT = iso => { if(!iso) return '—'; const d = new Date(new Date(iso).getTime()+IST); return d.toISOString().slice(11,16); };
const today = () => new Date().toISOString().slice(0,10);

function tick() {
  const d = new Date(Date.now()+IST);
  document.getElementById('clock').textContent = d.toISOString().slice(11,19)+' IST';
}
setInterval(tick,1000); tick();

async function refresh() {
  try {
    const [pos, trades] = await Promise.all([
      fetch('/api/open-positions').then(r=>r.json()),
      fetch('/api/trades').then(r=>r.json()),
    ]);
    document.getElementById('stat-open').textContent = pos.length;
    const t = today();
    const todayPnl = trades.filter(x=>x.entry_time&&x.entry_time.startsWith(t)&&x.outcome!=='PENDING').reduce((s,x)=>s+(x.pnl_inr||0),0);
    const todayEl = document.getElementById('stat-today');
    todayEl.textContent = fmt(todayPnl);
    todayEl.className = 'stat-value '+(todayPnl>=0?'pos':'neg');
    document.getElementById('stat-sigs').textContent = trades.filter(x=>x.entry_time&&x.entry_time.startsWith(t)).length;

    const grid = document.getElementById('pos-grid');
    if (!pos.length) { grid.innerHTML='<div class="empty">No open positions</div>'; return; }
    grid.innerHTML = pos.map(p=>`
      <div class="pos-card">
        <div class="pos-head"><span class="pos-ticker">${p.ticker}</span>
        <span class="dir ${p.direction==='LONG'?'dir-l':'dir-s'}">${p.direction}</span></div>
        <div class="pos-line">Entry ₹${(p.entry_price||0).toLocaleString('en-IN')}</div>
        <div class="pos-line">Target ₹${(p.target_price||0).toLocaleString('en-IN')} · Stop ₹${(p.stop_price||0).toLocaleString('en-IN')}</div>
        <div class="pos-line">Opened ${fmtT(p.opened_at)} IST &nbsp;·&nbsp; <span style="color:#00d4aa">OPEN</span></div>
      </div>`).join('');
  } catch(e) { console.error(e); }
}

const rows = [];
function addRow(sig) {
  const dir = sig.direction||'HOLD';
  const dc = dir==='LONG'?'fl':dir==='SHORT'?'fs':'fh';
  const conf = sig.confidence?(sig.confidence*100).toFixed(0)+'%':'—';
  const st = dir!=='HOLD'?'<span class="fex">✅ AUTO-EXECUTED</span>':'<span class="fno">—</span>';
  rows.unshift(`<div class="feed-row"><span class="ft">${fmtT(sig.generated_at||new Date().toISOString())}</span><span>${sig.ticker||'—'}</span><span class="${dc}">${dir}</span><span style="color:#666">${conf}</span>${st}</div>`);
  if(rows.length>50) rows.pop();
  document.getElementById('feed-body').innerHTML = rows.join('');
}

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws/signals`);
  ws.onmessage = e => { const m=JSON.parse(e.data); if(m.type==='signal') addRow(m.data); };
  ws.onclose = () => setTimeout(connectWS,3000);
}

refresh(); setInterval(refresh,30000); connectWS();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify manually**

Start the server: `python main.py`
Open: `http://localhost:5000`

Check:
- Nav bar visible with active "Overview" link
- 4 stat cards render
- "No open positions" shown when none exist
- "Waiting for signals…" in feed
- Clock ticks in top right

- [ ] **Step 3: Commit**

```bash
git add dashboard/static/index.html
git commit -m "feat: rebuild Overview dashboard page with live signal feed"
```

---

### Task 5: Rebuild trades.html — Trade History page

**Files:**
- Modify: `dashboard/static/trades.html` (full rewrite)

**Interfaces:**
- Consumes: `GET /api/trades` (Task 3)

- [ ] **Step 1: Replace dashboard/static/trades.html with this complete file**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot — Trades</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}
nav{display:flex;align-items:center;gap:20px;padding:0 24px;height:52px;background:#1a1d2e;border-bottom:1px solid #2a2d3e}
.brand{font-weight:700;color:#fff;font-size:15px;margin-right:8px}
nav a{color:#8888aa;text-decoration:none;font-size:13px;padding-bottom:3px}
nav a.active{color:#00d4aa;border-bottom:2px solid #00d4aa}
.spacer{flex:1}
.paper-badge{background:#ff8c0018;color:#ff8c00;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;border:1px solid #ff8c0033}
.toolbar{display:flex;gap:8px;align-items:center;padding:16px 24px;flex-wrap:wrap}
.fb{background:#1a1d2e;border:1px solid #2a2d3e;color:#666;padding:5px 14px;border-radius:4px;font-size:12px;cursor:pointer}
.fb.on{background:#00d4aa18;border-color:#00d4aa;color:#00d4aa}
.fb.on-r{background:#ff4d6d18;border-color:#ff4d6d;color:#ff4d6d}
.sep{color:#333;margin:0 4px}
.tw{padding:0 24px 8px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:#555;font-size:10px;text-transform:uppercase;letter-spacing:1px;padding:8px 12px;border-bottom:1px solid #2a2d3e;text-align:left;white-space:nowrap}
td{padding:10px 12px;border-bottom:1px solid #1a1d24;white-space:nowrap}
tr:hover td{background:#1e2130}
.pos{color:#00d4aa}.neg{color:#ff4d6d}
.badge{display:inline-block;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:600}
.b-win{background:#00d4aa18;color:#00d4aa}
.b-loss{background:#ff4d6d18;color:#ff4d6d}
.b-pend{background:#88888818;color:#888}
.pag{display:flex;gap:8px;align-items:center;padding:12px 24px;color:#555;font-size:12px}
.pag button{background:#1a1d2e;border:1px solid #2a2d3e;color:#aaa;padding:4px 12px;border-radius:4px;cursor:pointer}
.pag button:disabled{opacity:.3;cursor:default}
.empty{padding:32px;text-align:center;color:#444;font-size:13px}
</style>
</head>
<body>
<nav>
  <span class="brand">Trading Bot</span>
  <a href="/">Overview</a>
  <a href="/trades" class="active">Trades</a>
  <a href="/performance">Performance</a>
  <span class="spacer"></span>
  <span class="paper-badge">PAPER</span>
</nav>

<div class="toolbar">
  <span style="font-size:12px;color:#555;margin-right:4px">Direction:</span>
  <button class="fb dir-f on" data-val="all" onclick="setF(this,'dir')">All</button>
  <button class="fb dir-f" data-val="LONG" onclick="setF(this,'dir')">LONG</button>
  <button class="fb dir-f" data-val="SHORT" onclick="setF(this,'dir')">SHORT</button>
  <span class="sep">|</span>
  <span style="font-size:12px;color:#555;margin-right:4px">Outcome:</span>
  <button class="fb out-f on" data-val="all" onclick="setF(this,'out')">All</button>
  <button class="fb out-f" data-val="TARGET_HIT" onclick="setF(this,'out')">✅ Win</button>
  <button class="fb out-f" data-val="STOP_HIT" onclick="setF(this,'out')">❌ Loss</button>
  <button class="fb out-f" data-val="PENDING" onclick="setF(this,'out')">⏳ Open</button>
  <span class="spacer"></span>
  <span id="count-label" style="font-size:12px;color:#555"></span>
</div>

<div class="tw">
  <table>
    <thead>
      <tr>
        <th>Opened (IST)</th>
        <th>Ticker</th>
        <th>Dir</th>
        <th>Entry ₹</th>
        <th>Exit ₹</th>
        <th>Duration</th>
        <th>P&amp;L ₹</th>
        <th>Conf</th>
        <th>Outcome</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  <div id="empty-msg" class="empty" style="display:none">No trades match the filter.</div>
</div>

<div class="pag">
  <button id="prev-btn" onclick="goPage(-1)" disabled>← Prev</button>
  <span id="pag-info">—</span>
  <button id="next-btn" onclick="goPage(1)" disabled>Next →</button>
</div>

<script>
const IST = 5.5*3600000;
const PAGE = 50;
let all=[], filtered=[], page=0, dirF='all', outF='all';

const fmtIST = iso => { if(!iso) return '—'; const d=new Date(new Date(iso).getTime()+IST); return d.toISOString().slice(0,16).replace('T',' '); };
const fmtDur = (a,b) => { if(!a||!b) return '—'; const ms=new Date(b)-new Date(a); const h=Math.floor(ms/3600000),m=Math.floor((ms%3600000)/60000); return h?`${h}h ${m}m`:`${m}m`; };
const fmtPnl = v => v==null?'—':(v>=0?'+':'')+'₹'+Math.abs(v).toLocaleString('en-IN',{maximumFractionDigits:0});

function badge(o) {
  if(o==='TARGET_HIT') return '<span class="badge b-win">TARGET HIT</span>';
  if(o==='STOP_HIT')   return '<span class="badge b-loss">STOP HIT</span>';
  if(o==='EXPIRED')    return '<span class="badge b-pend">EXPIRED</span>';
  return '<span class="badge b-pend">OPEN</span>';
}

function render() {
  filtered = all.filter(t => (dirF==='all'||t.direction===dirF) && (outF==='all'||t.outcome===outF));
  const total = filtered.length;
  const start = page*PAGE, end = Math.min(start+PAGE, total);
  const rows = filtered.slice(start, end);

  document.getElementById('count-label').textContent = `${total} trade${total!==1?'s':''}`;
  document.getElementById('pag-info').textContent = total ? `${start+1}–${end} of ${total}` : '0';
  document.getElementById('prev-btn').disabled = page===0;
  document.getElementById('next-btn').disabled = end>=total;

  if (!rows.length) {
    document.getElementById('tbody').innerHTML='';
    document.getElementById('empty-msg').style.display='block';
    return;
  }
  document.getElementById('empty-msg').style.display='none';
  document.getElementById('tbody').innerHTML = rows.map(t => {
    const pnl = t.pnl_inr;
    const pc = pnl==null?'':pnl>=0?'pos':'neg';
    return `<tr>
      <td>${fmtIST(t.entry_time)}</td>
      <td style="font-weight:600">${t.ticker}</td>
      <td class="${t.direction==='LONG'?'pos':'neg'}">${t.direction}</td>
      <td>₹${(t.entry_price||0).toLocaleString('en-IN')}</td>
      <td>${t.exit_price!=null?'₹'+t.exit_price.toLocaleString('en-IN'):'—'}</td>
      <td style="color:#666">${fmtDur(t.entry_time,t.exit_time)}</td>
      <td class="${pc}">${fmtPnl(pnl)}</td>
      <td style="color:#666">${t.confidence?(t.confidence*100).toFixed(0)+'%':'—'}</td>
      <td>${badge(t.outcome)}</td>
    </tr>`;
  }).join('');
}

function setF(btn, type) {
  document.querySelectorAll('.'+type+'-f').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  if(type==='dir') dirF=btn.dataset.val;
  else outF=btn.dataset.val;
  page=0; render();
}
function goPage(d) { page=Math.max(0,page+d); render(); }

fetch('/api/trades').then(r=>r.json()).then(data=>{
  all = data.reverse(); // newest first
  render();
}).catch(e=>console.error(e));
</script>
</body>
</html>
```

- [ ] **Step 2: Verify manually**

Open: `http://localhost:5000/trades`

Check:
- Table renders with correct columns
- Filter buttons highlight correctly when clicked
- "No trades match the filter." shows when filter has no results
- Pagination controls appear when >50 trades

- [ ] **Step 3: Commit**

```bash
git add dashboard/static/trades.html
git commit -m "feat: rebuild Trades dashboard page with filter and pagination"
```

---

### Task 6: Rebuild performance.html + cleanup

**Files:**
- Modify: `dashboard/static/performance.html` (full rewrite)
- Delete: `dashboard/static/positions.html`
- Modify: `dashboard/server.py` (add `/trades` route — currently only `/history` and `/performance` routes exist for HTML; verify `/trades` is wired)

**Interfaces:**
- Consumes: `GET /api/equity-curve` (Task 3), `GET /api/stats` (Task 3)

- [ ] **Step 1: Add /trades HTML route to dashboard/server.py**

In `dashboard/server.py`, after the existing `/performance` route, add:

```python
@app.get("/trades", response_class=HTMLResponse)
async def trades_page():
    return Path("dashboard/static/trades.html").read_text()
```

- [ ] **Step 2: Replace dashboard/static/performance.html with this complete file**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot — Performance</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}
nav{display:flex;align-items:center;gap:20px;padding:0 24px;height:52px;background:#1a1d2e;border-bottom:1px solid #2a2d3e}
.brand{font-weight:700;color:#fff;font-size:15px;margin-right:8px}
nav a{color:#8888aa;text-decoration:none;font-size:13px;padding-bottom:3px}
nav a.active{color:#00d4aa;border-bottom:2px solid #00d4aa}
.spacer{flex:1}
.paper-badge{background:#ff8c0018;color:#ff8c00;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;border:1px solid #ff8c0033}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;padding:20px 24px}
.card{background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:16px 18px}
.stat-label{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px}
.stat-value{font-size:22px;font-weight:700}
.pos{color:#00d4aa}.neg{color:#ff4d6d}.neu{color:#e0e0e0}
.charts{display:grid;grid-template-columns:2fr 1fr;gap:16px;padding:0 24px 24px}
@media(max-width:700px){.charts{grid-template-columns:1fr}}
.chart-card{background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:18px}
.chart-title{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px}
.empty{padding:32px;text-align:center;color:#444;font-size:13px}
canvas{max-height:280px}
</style>
</head>
<body>
<nav>
  <span class="brand">Trading Bot</span>
  <a href="/">Overview</a>
  <a href="/trades">Trades</a>
  <a href="/performance" class="active">Performance</a>
  <span class="spacer"></span>
  <span class="paper-badge">PAPER</span>
</nav>

<div class="stat-grid">
  <div class="card"><div class="stat-label">Win Rate</div><div class="stat-value neu" id="s-wr">—</div></div>
  <div class="card"><div class="stat-label">Total P&amp;L</div><div class="stat-value neu" id="s-pnl">—</div></div>
  <div class="card"><div class="stat-label">Avg Win</div><div class="stat-value pos" id="s-aw">—</div></div>
  <div class="card"><div class="stat-label">Avg Loss</div><div class="stat-value neg" id="s-al">—</div></div>
  <div class="card"><div class="stat-label">Max Drawdown</div><div class="stat-value neg" id="s-dd">—</div></div>
  <div class="card"><div class="stat-label">Total Trades</div><div class="stat-value neu" id="s-tt">—</div></div>
</div>

<div class="charts">
  <div class="chart-card">
    <div class="chart-title">Equity Curve (Cumulative P&amp;L ₹)</div>
    <div id="eq-empty" class="empty" style="display:none">No closed trades yet — equity curve will appear here.</div>
    <canvas id="eq-chart"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-title">Win / Loss Split</div>
    <div id="dl-empty" class="empty" style="display:none">No resolved trades yet.</div>
    <canvas id="dl-chart"></canvas>
  </div>
</div>

<script>
const fmtINR = v => v==null?'—':(v>=0?'+₹':'-₹')+Math.abs(v).toLocaleString('en-IN',{maximumFractionDigits:0});
const fmtIST = iso => { if(!iso) return ''; const d=new Date(new Date(iso).getTime()+5.5*3600000); return d.toISOString().slice(0,10); };

let eqChart=null, dlChart=null;

async function load() {
  try {
    const [stats, curve] = await Promise.all([
      fetch('/api/stats').then(r=>r.json()),
      fetch('/api/equity-curve').then(r=>r.json()),
    ]);

    // Stat cards
    const wr = stats.win_rate;
    const wrEl = document.getElementById('s-wr');
    wrEl.textContent = stats.resolved ? wr.toFixed(1)+'%' : '—';
    wrEl.className = 'stat-value '+(wr>=50?'pos':wr>0?'neg':'neu');

    const pEl = document.getElementById('s-pnl');
    pEl.textContent = fmtINR(stats.total_pnl_inr);
    pEl.className = 'stat-value '+(stats.total_pnl_inr>=0?'pos':'neg');

    document.getElementById('s-aw').textContent = stats.avg_win_inr ? fmtINR(stats.avg_win_inr) : '—';
    document.getElementById('s-al').textContent = stats.avg_loss_inr ? fmtINR(stats.avg_loss_inr) : '—';
    document.getElementById('s-dd').textContent = stats.max_drawdown_inr ? '-₹'+stats.max_drawdown_inr.toLocaleString('en-IN',{maximumFractionDigits:0}) : '₹0';
    document.getElementById('s-tt').textContent = stats.total_trades || 0;

    // Equity curve
    if (!curve.length) {
      document.getElementById('eq-empty').style.display='block';
      document.getElementById('eq-chart').style.display='none';
    } else {
      const labels = curve.map(p=>fmtIST(p.t));
      const data = curve.map(p=>p.pnl);
      const ctx = document.getElementById('eq-chart').getContext('2d');
      if (eqChart) eqChart.destroy();
      const grad = ctx.createLinearGradient(0,0,0,280);
      grad.addColorStop(0,'rgba(0,212,170,0.25)');
      grad.addColorStop(1,'rgba(0,212,170,0.01)');
      eqChart = new Chart(ctx, {
        type:'line',
        data:{
          labels,
          datasets:[{
            data,
            borderColor:'#00d4aa',
            backgroundColor:grad,
            borderWidth:2,
            fill:true,
            tension:0.3,
            pointRadius:data.length>60?0:3,
            pointHoverRadius:5,
          }]
        },
        options:{
          responsive:true,
          plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>fmtINR(c.raw)}}},
          scales:{
            x:{ticks:{color:'#555',maxTicksLimit:8},grid:{color:'#1e2130'}},
            y:{ticks:{color:'#555',callback:v=>'₹'+v.toLocaleString('en-IN')},grid:{color:'#1e2130'}}
          }
        }
      });
    }

    // Donut
    if (!stats.wins && !stats.losses) {
      document.getElementById('dl-empty').style.display='block';
      document.getElementById('dl-chart').style.display='none';
    } else {
      const ctx2 = document.getElementById('dl-chart').getContext('2d');
      if (dlChart) dlChart.destroy();
      dlChart = new Chart(ctx2, {
        type:'doughnut',
        data:{
          labels:['Wins','Losses'],
          datasets:[{data:[stats.wins,stats.losses],backgroundColor:['#00d4aa','#ff4d6d'],borderWidth:0}]
        },
        options:{
          responsive:true,
          plugins:{
            legend:{labels:{color:'#aaa',padding:16}},
            tooltip:{callbacks:{label:c=>`${c.label}: ${c.raw} (${(c.raw/(stats.wins+stats.losses)*100).toFixed(1)}%)`}}
          }
        }
      });
    }
  } catch(e) { console.error(e); }
}

load(); setInterval(load,60000);
</script>
</body>
</html>
```

- [ ] **Step 3: Delete positions.html**

```bash
Remove-Item dashboard/static/positions.html
```

Or on bash: `rm dashboard/static/positions.html`

- [ ] **Step 4: Verify manually**

Open: `http://localhost:5000/performance`

Check:
- Stat cards render (show `—` when no data)
- "No closed trades yet" message shown in equity curve when no data
- "No resolved trades yet." in donut when no data
- With data: equity curve line chart and donut both render
- Page refreshes automatically every 60s

Open: `http://localhost:5000/trades`

Check: trades page loads correctly.

- [ ] **Step 5: Run full test suite one final time**

```bash
python -m pytest tests/ -q --basetemp=.pytest-tmp
```

Expected: all tests pass (no test referenced `positions.html`).

- [ ] **Step 6: Commit**

```bash
git add dashboard/static/performance.html dashboard/server.py
git rm dashboard/static/positions.html
git commit -m "feat: rebuild Performance dashboard with Chart.js equity curve and donut chart"
```

---

## Summary of all file changes

| File | Change |
|------|--------|
| `main.py` | Remove TelegramBot; tasks list has no polling task |
| `core/stop_monitor.py` | Add null guards around both telegram call sites |
| `harnesses/base_harness.py` | Import Position; replace telegram send with auto-execute block |
| `dashboard/server.py` | Add `/api/trades`, `/api/equity-curve`, `/api/stats`, `/trades` HTML route, `/positions` redirect |
| `dashboard/static/index.html` | Full rewrite |
| `dashboard/static/trades.html` | Full rewrite |
| `dashboard/static/performance.html` | Full rewrite with Chart.js |
| `dashboard/static/positions.html` | Deleted |
| `tests/test_stop_monitor.py` | +1 null guard test |
| `tests/test_auto_execute.py` | New — 5 tests |
| `tests/test_dashboard.py` | New — 10 tests |
