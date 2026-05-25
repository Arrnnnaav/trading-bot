import pytest
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


def test_chronos_outcome_written_to_jsonl(tmp_path, monkeypatch):
    import json
    from core.signal_aggregator import SignalAggregator
    from core.models import SignalOutcome

    outcomes_path = tmp_path / "chronos_outcomes.jsonl"
    monkeypatch.setenv("CHRONOS_OUTCOMES_PATH", str(outcomes_path))
    # Reload module so CHRONOS_OUTCOMES_PATH picks up new env var
    import importlib
    import core.signal_tracker as st_mod

    importlib.reload(st_mod)

    agg = SignalAggregator(signal_log_path=str(tmp_path / "signal_log.json"))
    tracker = st_mod.SignalTracker(agg, MagicMock(), MagicMock())

    entry = {
        "id": "sig_test_001",
        "ticker": "BTC/USDT",
        "market": "crypto",
        "direction": "LONG",
        "entry_price": 62000.0,
        "target_price": 63200.0,
        "stop_price": 61400.0,
        "position_size_inr": 2000.0,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "agent_votes": [
            {
                "agent_name": "ChronosTechnical",
                "direction": "LONG",
                "confidence": 0.71,
                "reasoning": "test",
            }
        ],
    }
    tracker._write_chronos_outcome(entry, SignalOutcome.TARGET_HIT)

    lines = outcomes_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticker"] == "BTC/USDT"
    assert record["predicted"] == "LONG"
    assert record["confidence"] == pytest.approx(0.71)
    assert record["actual"] == "TARGET_HIT"
