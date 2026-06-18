import json
from pathlib import Path
from agents.fii_dii import FIIDIIAgent
from core.models import Direction, Market


def _write_fii_files(tmp_path: Path, records: list[dict]) -> Path:
    for r in records:
        f = tmp_path / f"{r['date']}.json"
        f.write_text(json.dumps(r))
    return tmp_path


def _klines():
    return [
        {"open": 100, "high": 105, "low": 95, "close": 100, "volume": 1000}
        for _ in range(10)
    ]


def test_fii_strong_buying_is_long(tmp_path):
    records = [
        {"date": f"2026-06-1{i}", "fii_net_cr": 800.0, "dii_net_cr": 200.0}
        for i in range(1, 6)
    ]
    _write_fii_files(tmp_path, records)
    agent = FIIDIIAgent(data_dir=tmp_path)
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.LONG
    assert vote.confidence >= 0.5


def test_fii_sustained_selling_is_short(tmp_path):
    records = [
        {"date": f"2026-06-1{i}", "fii_net_cr": -1200.0, "dii_net_cr": 300.0}
        for i in range(1, 6)
    ]
    _write_fii_files(tmp_path, records)
    agent = FIIDIIAgent(data_dir=tmp_path)
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.SHORT


def test_fii_no_data_returns_hold(tmp_path):
    agent = FIIDIIAgent(data_dir=tmp_path)
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0


def test_fii_dii_floor_is_hold(tmp_path):
    records = [
        {"date": f"2026-06-1{i}", "fii_net_cr": -500.0, "dii_net_cr": 600.0}
        for i in range(1, 6)
    ]
    _write_fii_files(tmp_path, records)
    agent = FIIDIIAgent(data_dir=tmp_path)
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.HOLD


def test_fii_agent_never_raises(tmp_path):
    agent = FIIDIIAgent(data_dir=tmp_path)
    vote = agent.analyze("NIFTY", [], Market.INDIA)
    assert vote.direction == Direction.HOLD
