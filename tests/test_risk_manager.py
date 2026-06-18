import pytest
from config import config
from core.models import Direction, HarnessState, Market, Position
from core.risk_manager import RiskManager


def _klines(n=20, close=100.0, spread=2.0):
    return [
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 1000,
        }
        for _ in range(n)
    ]


def _state(**kw):
    defaults = dict(
        portfolio_value_inr=100_000.0,
        daily_realized_pnl_inr=0.0,
        weekly_realized_pnl_inr=0.0,
        consecutive_losses=0,
        open_positions=[],
    )
    defaults.update(kw)
    return HarnessState(**defaults)


def test_size_position_uses_notional_cap_when_stop_is_near():
    manager = RiskManager(Market.INDIA)
    state = HarnessState(portfolio_value_inr=100000.0)

    size = manager.size_position(state, entry=100.0, stop=98.0)

    assert size == 2000.0


def test_size_position_uses_risk_budget_when_stop_is_wide():
    manager = RiskManager(Market.INDIA)
    state = HarnessState(portfolio_value_inr=100000.0)

    size = manager.size_position(state, entry=100.0, stop=50.0)

    assert size == 1000.0


def test_approve_blocks_after_consecutive_loss_limit():
    manager = RiskManager(Market.INDIA)
    state = HarnessState(consecutive_losses=3)
    klines = [
        {"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0, "volume": 1.0}
        for _ in range(20)
    ]

    approved, reason = manager.approve(
        state=state,
        ticker="BTC/USDT",
        direction=Direction.LONG,
        confidence=0.9,
        klines=klines,
    )

    assert approved is False
    assert "Consecutive loss limit" in reason


def test_calc_atr_fewer_bars_than_period_uses_last_bar_range():
    rm = RiskManager(Market.INDIA)
    klines = _klines(3, close=100.0, spread=5.0)
    atr = rm._calc_atr(klines, period=14)
    assert atr == pytest.approx(10.0)


def test_size_position_zero_portfolio():
    rm = RiskManager(Market.INDIA)
    state = _state(portfolio_value_inr=0.0)
    assert rm.size_position(state) == 0.0


def test_approve_max_positions_gate():
    rm = RiskManager(Market.INDIA)
    positions = [
        Position(
            signal_id=f"s{i}",
            ticker="NIFTY",
            market=Market.INDIA,
            direction=Direction.LONG,
            entry_price=100.0,
            stop_price=90.0,
            target_price=120.0,
            size_inr=2000.0,
            opened_at="2026-01-01T00:00:00+00:00",
            broker_order_id=f"ord{i}",
        )
        for i in range(config.max_open_positions_per_market)
    ]
    state = _state(open_positions=positions)
    ok, reason = rm.approve(state, "BTC", Direction.LONG, 0.9, _klines())
    assert not ok
    assert "Max" in reason


def test_approve_daily_loss_gate():
    rm = RiskManager(Market.INDIA)
    loss = -(config.max_daily_loss_pct + 0.01) * 100_000.0
    state = _state(daily_realized_pnl_inr=loss)
    ok, reason = rm.approve(state, "BTC", Direction.LONG, 0.9, _klines())
    assert not ok
    assert "Daily" in reason


def test_approve_low_confidence_gate():
    rm = RiskManager(Market.INDIA)
    ok, reason = rm.approve(_state(), "BTC", Direction.LONG, 0.3, _klines())
    assert not ok
    assert "Confidence" in reason


def test_approve_hold_direction_gate():
    rm = RiskManager(Market.INDIA)
    ok, reason = rm.approve(_state(), "BTC", Direction.HOLD, 0.9, _klines())
    assert not ok
    assert "HOLD" in reason


def test_approve_passes_clean_state():
    rm = RiskManager(Market.INDIA)
    # spread=1.0 → ATR≈2, stop=97, target=105.4, R:R=2.7 — well above 1.8 min
    ok, _ = rm.approve(_state(), "BTC", Direction.LONG, 0.9, _klines(20, spread=1.0))
    assert ok
