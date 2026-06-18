from core.risk_manager import RiskManager
from core.models import HarnessState, Direction, Market


def _klines(n=20):
    # close != mid(high, low) so R:R calculation avoids floating-point underflow
    return [
        {"open": 100.0, "high": 105.0, "low": 95.0, "close": 101.0, "volume": 1000}
        for _ in range(n)
    ]


def _state():
    return HarnessState(portfolio_value_inr=100_000.0)


def test_vix_above_gate_blocks_signal():
    rm = RiskManager(Market.INDIA)
    ok, reason = rm.approve(_state(), "NIFTY", Direction.LONG, 0.9, _klines(), vix=26.0)
    assert not ok
    assert "VIX" in reason


def test_vix_below_gate_passes():
    rm = RiskManager(Market.INDIA)
    ok, _ = rm.approve(_state(), "NIFTY", Direction.LONG, 0.9, _klines(), vix=18.0)
    assert ok


def test_vix_none_skips_gate():
    rm = RiskManager(Market.INDIA)
    ok, _ = rm.approve(_state(), "NIFTY", Direction.LONG, 0.9, _klines(), vix=None)
    assert ok


def test_expiry_proximity_under_2hr_blocks():
    rm = RiskManager(Market.INDIA)
    ok, reason = rm.approve(
        _state(), "NIFTY", Direction.LONG, 0.9, _klines(), minutes_to_expiry=90
    )
    assert not ok
    assert "expiry" in reason.lower()


def test_expiry_proximity_over_2hr_passes():
    rm = RiskManager(Market.INDIA)
    ok, _ = rm.approve(
        _state(), "NIFTY", Direction.LONG, 0.9, _klines(), minutes_to_expiry=180
    )
    assert ok


def test_expiry_proximity_none_skips_gate():
    rm = RiskManager(Market.INDIA)
    ok, _ = rm.approve(
        _state(), "NIFTY", Direction.LONG, 0.9, _klines(), minutes_to_expiry=None
    )
    assert ok
