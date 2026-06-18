import pytest
from unittest.mock import MagicMock
from agents.kronos_india import KronosAgent
from core.models import Direction, Market


def _klines(n=70):
    base = 22000.0
    return [
        {
            "open": base + i,
            "high": base + i + 20,
            "low": base + i - 20,
            "close": base + i + 5,
            "volume": 1_000_000,
        }
        for i in range(n)
    ]


def test_kronos_no_model_returns_hold(tmp_path):
    agent = KronosAgent(model_path=str(tmp_path / "nonexistent.pt"))
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.HOLD
    assert "model" in vote.reasoning.lower()


def test_kronos_insufficient_data_returns_hold():
    agent = KronosAgent(model_path="models/kronos_india/best.pt")
    vote = agent.analyze("NIFTY", _klines(10), Market.INDIA)
    assert vote.direction == Direction.HOLD


def test_kronos_with_mock_model_returns_long():
    agent = KronosAgent(model_path="models/kronos_india/best.pt")
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "long_prob": 0.75,
        "short_prob": 0.15,
        "hold_prob": 0.10,
    }
    agent._model = mock_model
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.LONG
    assert vote.confidence == pytest.approx(0.75)


def test_kronos_low_confidence_returns_hold():
    agent = KronosAgent(model_path="models/kronos_india/best.pt")
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "long_prob": 0.40,
        "short_prob": 0.35,
        "hold_prob": 0.25,
    }
    agent._model = mock_model
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.HOLD


def test_kronos_never_raises():
    agent = KronosAgent(model_path="nonexistent_path")
    vote = agent.analyze("NIFTY", [], Market.INDIA)
    assert vote.direction == Direction.HOLD
