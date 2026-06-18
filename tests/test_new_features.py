"""Tests for AgentPerformanceTracker, PatternReputationTracker, PaperBroker."""

import json
from unittest.mock import MagicMock
from pathlib import Path

from core.models import Direction


# ── helpers ───────────────────────────────────────────────────────────────────


def make_klines(n=100):
    return [
        {
            "open": 62000.0 + i,
            "high": 63000.0 + i,
            "low": 61500.0 + i,
            "close": 62400.0 + i * 10,
            "volume": 1200.0 + i,
        }
        for i in range(n)
    ]


def write_signal_log(tmp_path, signals):
    p = tmp_path / "signal_log.json"
    p.write_text(json.dumps(signals))
    return str(p)


def make_signal(direction="LONG", outcome="TARGET_HIT", ticker="BTC/USDT", agents=None):
    if agents is None:
        agents = [{"agent_name": "AgentA", "direction": direction, "confidence": 0.75}]
    return {
        "id": "sig_test",
        "direction": direction,
        "outcome": outcome,
        "ticker": ticker,
        "agent_votes": agents,
    }


# ── AgentPerformanceTracker ────────────────────────────────────────────────────


class TestAgentPerformanceTracker:
    def test_empty_log_no_weights(self, tmp_path):
        from core.agent_tracker import AgentPerformanceTracker

        log = write_signal_log(tmp_path, [])
        tracker = AgentPerformanceTracker(log)
        assert tracker.get_weights() == {}

    def test_weight_above_baseline_for_winning_agent(self, tmp_path):
        from core.agent_tracker import AgentPerformanceTracker, MIN_SAMPLES

        signals = [make_signal("LONG", "TARGET_HIT") for _ in range(MIN_SAMPLES)]
        log = write_signal_log(tmp_path, signals)
        tracker = AgentPerformanceTracker(log)
        weights = tracker.get_weights()
        assert "AgentA" in weights
        assert weights["AgentA"] > 1.0  # win rate 100% → weight 2.0 (max)

    def test_weight_below_baseline_for_losing_agent(self, tmp_path):
        from core.agent_tracker import AgentPerformanceTracker, MIN_SAMPLES

        signals = [make_signal("LONG", "STOP_HIT") for _ in range(MIN_SAMPLES)]
        log = write_signal_log(tmp_path, signals)
        tracker = AgentPerformanceTracker(log)
        weights = tracker.get_weights()
        assert "AgentA" in weights
        assert weights["AgentA"] < 1.0  # win rate 0% → weight 0.5 (min)

    def test_ignores_agents_below_min_samples(self, tmp_path):
        from core.agent_tracker import AgentPerformanceTracker, MIN_SAMPLES

        signals = [make_signal("LONG", "TARGET_HIT") for _ in range(MIN_SAMPLES - 1)]
        log = write_signal_log(tmp_path, signals)
        tracker = AgentPerformanceTracker(log)
        assert tracker.get_weights() == {}

    def test_ignores_hold_votes(self, tmp_path):
        from core.agent_tracker import AgentPerformanceTracker, MIN_SAMPLES

        agents = [{"agent_name": "AgentA", "direction": "HOLD", "confidence": 0.75}]
        signals = [
            make_signal("LONG", "TARGET_HIT", agents=agents) for _ in range(MIN_SAMPLES)
        ]
        log = write_signal_log(tmp_path, signals)
        tracker = AgentPerformanceTracker(log)
        assert tracker.get_weights() == {}  # HOLD votes excluded

    def test_refresh_reloads(self, tmp_path):
        from core.agent_tracker import AgentPerformanceTracker, MIN_SAMPLES

        log_path = tmp_path / "signal_log.json"
        log_path.write_text("[]")
        tracker = AgentPerformanceTracker(str(log_path))
        assert tracker.get_weights() == {}

        signals = [make_signal("LONG", "TARGET_HIT") for _ in range(MIN_SAMPLES)]
        log_path.write_text(json.dumps(signals))
        tracker.refresh()
        assert "AgentA" in tracker.get_weights()


# ── PatternReputationTracker ───────────────────────────────────────────────────


class TestPatternReputationTracker:
    def test_new_pattern_always_allowed(self, tmp_path):
        from core.pattern_reputation import PatternReputationTracker

        log = write_signal_log(tmp_path, [])
        tracker = PatternReputationTracker(log)
        votes = [{"agent_name": "A", "direction": "LONG", "confidence": 0.8}]
        allowed, _ = tracker.is_pattern_allowed("LONG", "BTC/USDT", votes)
        assert allowed

    def test_insufficient_samples_allowed(self, tmp_path):
        from core.pattern_reputation import PatternReputationTracker, MIN_SAMPLES

        signals = [make_signal("LONG", "STOP_HIT") for _ in range(MIN_SAMPLES - 1)]
        log = write_signal_log(tmp_path, signals)
        tracker = PatternReputationTracker(log)
        votes = [{"agent_name": "AgentA", "direction": "LONG", "confidence": 0.75}]
        allowed, _ = tracker.is_pattern_allowed("LONG", "BTC/USDT", votes)
        assert allowed

    def test_low_win_rate_pattern_blocked(self, tmp_path):
        from core.pattern_reputation import PatternReputationTracker, MIN_SAMPLES

        signals = [make_signal("LONG", "STOP_HIT") for _ in range(MIN_SAMPLES)]
        log = write_signal_log(tmp_path, signals)
        tracker = PatternReputationTracker(log)
        votes = [{"agent_name": "AgentA", "direction": "LONG", "confidence": 0.75}]
        allowed, reason = tracker.is_pattern_allowed("LONG", "BTC/USDT", votes)
        assert not allowed
        assert "blocked" in reason

    def test_high_win_rate_pattern_allowed(self, tmp_path):
        from core.pattern_reputation import PatternReputationTracker, MIN_SAMPLES

        signals = [make_signal("LONG", "TARGET_HIT") for _ in range(MIN_SAMPLES)]
        log = write_signal_log(tmp_path, signals)
        tracker = PatternReputationTracker(log)
        votes = [{"agent_name": "AgentA", "direction": "LONG", "confidence": 0.75}]
        allowed, _ = tracker.is_pattern_allowed("LONG", "BTC/USDT", votes)
        assert allowed


# ── PaperBroker ───────────────────────────────────────────────────────────────


class TestPaperBroker:
    def test_place_order_logs_trade(self, tmp_path):
        from brokers.paper_broker import PaperBroker

        mock_broker = MagicMock()
        log_path = str(tmp_path / "paper.json")
        broker = PaperBroker(mock_broker, log_path)

        result = broker.place_order("BTCUSDT", Direction.LONG, 100.0)
        assert result["status"] == "paper"
        assert result["order_id"].startswith("paper-")

        trades = json.loads(Path(log_path).read_text())
        assert len(trades) == 1
        assert trades[0]["ticker"] == "BTCUSDT"
        assert trades[0]["direction"] == "LONG"

    def test_place_options_order_logs_trade(self, tmp_path):
        from brokers.paper_broker import PaperBroker

        mock_broker = MagicMock()
        log_path = str(tmp_path / "paper.json")
        broker = PaperBroker(mock_broker, log_path)

        result = broker.place_options_order("NIFTY", Direction.LONG, 1, "2026-05-29")
        assert result["status"] == "paper"

        trades = json.loads(Path(log_path).read_text())
        assert trades[0]["type"] == "options"
        assert trades[0]["expiry"] == "2026-05-29"

    def test_get_price_passes_through(self, tmp_path):
        from brokers.paper_broker import PaperBroker

        mock_broker = MagicMock()
        mock_broker.get_price.return_value = 62000.0
        broker = PaperBroker(mock_broker, str(tmp_path / "paper.json"))

        assert broker.get_price("BTCUSDT") == 62000.0
        mock_broker.get_price.assert_called_once_with("BTCUSDT")

    def test_get_ohlcv_passes_through(self, tmp_path):
        from brokers.paper_broker import PaperBroker

        mock_broker = MagicMock()
        mock_broker.get_ohlcv.return_value = make_klines(10)
        broker = PaperBroker(mock_broker, str(tmp_path / "paper.json"))

        result = broker.get_ohlcv("BTCUSDT", interval="15m", limit=10)
        assert len(result) == 10
        mock_broker.get_ohlcv.assert_called_once()

    def test_multiple_trades_accumulate(self, tmp_path):
        from brokers.paper_broker import PaperBroker

        mock_broker = MagicMock()
        log_path = str(tmp_path / "paper.json")
        broker = PaperBroker(mock_broker, log_path)

        broker.place_order("BTCUSDT", Direction.LONG, 100.0)
        broker.place_order("ETHUSDT", Direction.SHORT, 50.0)

        trades = json.loads(Path(log_path).read_text())
        assert len(trades) == 2
