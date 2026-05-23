from unittest.mock import MagicMock
from core.signal_tracker import SignalTracker
from core.models import SignalOutcome


def make_pending_entry(
    direction="LONG", entry=62400, target=65100, stop=61000, size_inr=2000
):
    return {
        "id": "sig_001",
        "market": "crypto",
        "ticker": "BTC/USDT",
        "direction": direction,
        "entry_price": entry,
        "target_price": target,
        "stop_price": stop,
        "confidence": 0.74,
        "position_size_inr": size_inr,
        "generated_at": "2026-05-23T08:00:00Z",
        "executed": False,
        "outcome": "PENDING",
        "outcome_price": None,
        "outcome_at": None,
        "hypothetical_pnl_pct": None,
        "hypothetical_pnl_inr": None,
    }


def test_target_hit_long():
    tracker = SignalTracker(
        aggregator=MagicMock(), crypto_broker=MagicMock(), india_broker=MagicMock()
    )
    entry = make_pending_entry("LONG", 62400, 65100, 61000, 2000)
    outcome = tracker._resolve_outcome(entry, current_price=65200.0)
    assert outcome == SignalOutcome.TARGET_HIT


def test_stop_hit_long():
    tracker = SignalTracker(
        aggregator=MagicMock(), crypto_broker=MagicMock(), india_broker=MagicMock()
    )
    entry = make_pending_entry("LONG", 62400, 65100, 61000, 2000)
    outcome = tracker._resolve_outcome(entry, current_price=60900.0)
    assert outcome == SignalOutcome.STOP_HIT


def test_hypothetical_pnl_target_hit():
    tracker = SignalTracker(
        aggregator=MagicMock(), crypto_broker=MagicMock(), india_broker=MagicMock()
    )
    pnl_pct, pnl_inr = tracker._calc_pnl(
        direction="LONG", entry_price=62400, outcome_price=65100, position_size_inr=2000
    )
    assert pnl_pct > 0
    assert round(pnl_inr, 2) == round(2000 * pnl_pct, 2)


def test_hypothetical_pnl_stop_hit():
    tracker = SignalTracker(
        aggregator=MagicMock(), crypto_broker=MagicMock(), india_broker=MagicMock()
    )
    pnl_pct, pnl_inr = tracker._calc_pnl(
        direction="LONG", entry_price=62400, outcome_price=61000, position_size_inr=2000
    )
    assert pnl_pct < 0
