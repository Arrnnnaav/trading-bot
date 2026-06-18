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
    with patch.object(
        server_module.aggregator, "get_all_signals", return_value=signals
    ):
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
    with patch.object(
        server_module.aggregator, "get_all_signals", return_value=signals
    ):
        resp = client.get("/api/equity-curve")
    assert resp.status_code == 200
    points = resp.json()
    assert len(points) == 2
    assert points[0]["pnl"] == 4000.0  # first resolved: +4000
    assert points[1]["pnl"] == 2500.0  # cumulative: 4000 + (-1500)


def test_api_equity_curve_skips_pending():
    signals = [_SIGNAL_TARGET_HIT, _SIGNAL_PENDING]
    with patch.object(
        server_module.aggregator, "get_all_signals", return_value=signals
    ):
        resp = client.get("/api/equity-curve")
    points = resp.json()
    assert len(points) == 1  # pending excluded


def test_api_equity_curve_empty():
    with patch.object(server_module.aggregator, "get_all_signals", return_value=[]):
        resp = client.get("/api/equity-curve")
    assert resp.json() == []


def test_api_stats_win_loss_counts():
    signals = [_SIGNAL_TARGET_HIT, _SIGNAL_STOP_HIT]
    with patch.object(
        server_module.aggregator, "get_all_signals", return_value=signals
    ):
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
    s1 = {
        **_SIGNAL_TARGET_HIT,
        "id": "s1",
        "outcome_at": "2026-06-19T09:00:00+00:00",
        "hypothetical_pnl_inr": 3000.0,
        "outcome": "TARGET_HIT",
    }
    s2 = {
        **_SIGNAL_TARGET_HIT,
        "id": "s2",
        "outcome_at": "2026-06-19T10:00:00+00:00",
        "hypothetical_pnl_inr": 2500.0,
        "outcome": "TARGET_HIT",
    }
    s3 = {
        **_SIGNAL_TARGET_HIT,
        "id": "s3",
        "outcome_at": "2026-06-19T11:00:00+00:00",
        "hypothetical_pnl_inr": -1500.0,
        "outcome": "STOP_HIT",
    }
    with patch.object(
        server_module.aggregator, "get_all_signals", return_value=[s1, s2, s3]
    ):
        resp = client.get("/api/stats")
    assert resp.json()["max_drawdown_inr"] == 1500.0
