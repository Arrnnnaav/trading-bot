import torch
import pytest
from unittest.mock import MagicMock
from core.models import Direction, Market


def make_klines(n=100):
    return [
        {
            "open": 62000.0,
            "high": 63000.0,
            "low": 61500.0,
            "close": 62400.0 + i * 10,
            "volume": 1200.0,
        }
        for i in range(n)
    ]


def test_chronos_agent_returns_agent_vote():
    from agents.chronos_technical import ChronosTechnicalAgent

    agent = ChronosTechnicalAgent.__new__(ChronosTechnicalAgent)
    mock_model = MagicMock()
    mock_model.predict.return_value = (Direction.LONG, 0.72)
    mock_model.eval = MagicMock()

    logits = torch.tensor([[2.0, -1.0, 0.5]])
    mock_model.return_value = logits
    agent.model = mock_model
    agent.device = torch.device("cpu")

    vote = agent.analyze(ticker="BTC/USDT", klines=make_klines(), market=Market.CRYPTO)
    assert vote.direction == Direction.LONG
    assert vote.confidence == pytest.approx(0.72)
    assert vote.agent_name == "ChronosTechnical"
    assert "Chronos-2" in vote.reasoning


def test_chronos_agent_hold_on_low_confidence():
    from agents.chronos_technical import ChronosTechnicalAgent

    agent = ChronosTechnicalAgent.__new__(ChronosTechnicalAgent)
    mock_model = MagicMock()
    mock_model.predict.return_value = (Direction.HOLD, 0.0)
    mock_model.eval = MagicMock()
    logits = torch.tensor([[0.3, 0.3, 0.4]])
    mock_model.return_value = logits
    agent.model = mock_model
    agent.device = torch.device("cpu")

    vote = agent.analyze(ticker="BTC/USDT", klines=make_klines(), market=Market.CRYPTO)
    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0


def test_chronos_agent_fails_loudly_if_model_missing(monkeypatch, tmp_path):
    from agents import chronos_technical

    # Point both model paths to non-existent locations
    monkeypatch.setattr(chronos_technical, "MODEL_PATH_V2", str(tmp_path / "v2.pt"))
    monkeypatch.setattr(chronos_technical, "MODEL_PATH_V1", str(tmp_path / "v1.pt"))

    with pytest.raises(FileNotFoundError, match="Train first"):
        chronos_technical.ChronosTechnicalAgent()


def test_chronos_agent_needs_96_candles():
    from agents.chronos_technical import ChronosTechnicalAgent

    agent = ChronosTechnicalAgent.__new__(ChronosTechnicalAgent)
    mock_model = MagicMock()
    mock_model.predict.return_value = (Direction.HOLD, 0.0)
    mock_model.eval = MagicMock()
    agent.model = mock_model
    agent.device = torch.device("cpu")

    vote = agent.analyze(
        ticker="BTC/USDT", klines=make_klines(50), market=Market.CRYPTO
    )
    assert vote.direction == Direction.HOLD
    assert "insufficient" in vote.reasoning.lower()
