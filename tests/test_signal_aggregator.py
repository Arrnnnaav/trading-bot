import json
from datetime import datetime, timezone
from core.signal_aggregator import SignalAggregator
from core.models import Signal, Market, Direction


def make_signal(id="s1", market=Market.INDIA, ticker="NIFTY", direction=Direction.LONG):
    return Signal(
        id=id,
        market=market,
        ticker=ticker,
        direction=direction,
        entry_price=62400.0,
        target_price=65100.0,
        stop_price=61000.0,
        confidence=0.74,
        position_size_inr=2000.0,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def test_log_signal_writes_to_file(tmp_path):
    log_path = str(tmp_path / "signal_log.json")
    agg = SignalAggregator(signal_log_path=log_path)
    sig = make_signal()
    agg.log_signal(sig)
    with open(log_path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["id"] == "s1"


def test_log_signal_appends(tmp_path):
    log_path = str(tmp_path / "signal_log.json")
    agg = SignalAggregator(signal_log_path=log_path)
    agg.log_signal(make_signal("s1"))
    agg.log_signal(make_signal("s2"))
    with open(log_path) as f:
        data = json.load(f)
    assert len(data) == 2


def test_dedup_same_ticker_returns_true(tmp_path):
    log_path = str(tmp_path / "signal_log.json")
    agg = SignalAggregator(signal_log_path=log_path)
    agg.log_signal(make_signal("s1", ticker="NIFTY"))
    assert agg.is_duplicate(make_signal("s2", ticker="NIFTY")) == True


def test_dedup_different_ticker_returns_false(tmp_path):
    log_path = str(tmp_path / "signal_log.json")
    agg = SignalAggregator(signal_log_path=log_path)
    agg.log_signal(make_signal("s1", ticker="NIFTY"))
    assert agg.is_duplicate(make_signal("s2", ticker="BANKNIFTY")) == False
