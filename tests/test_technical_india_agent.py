from agents.technical_india import TechnicalIndiaAgent
from core.models import Direction, Market


def _klines(n=60, trend="up"):
    """Generate synthetic klines."""
    base = 22000.0
    result = []
    for i in range(n):
        if trend == "up":
            close = base + i * 10
        elif trend == "down":
            close = base - i * 10
        else:
            close = base + (i % 5) * 2
        result.append(
            {
                "open": close - 5,
                "high": close + 20,
                "low": close - 20,
                "close": close,
                "volume": 1_000_000 + i * 1000,
            }
        )
    return result


def test_technical_agent_returns_agent_vote():
    agent = TechnicalIndiaAgent()
    vote = agent.analyze("NIFTY", _klines(60, "up"), Market.INDIA)
    assert vote.agent_name == "technical_india"
    assert vote.direction in (Direction.LONG, Direction.SHORT, Direction.HOLD)
    assert 0.0 <= vote.confidence <= 1.0
    assert isinstance(vote.reasoning, str)


def test_technical_agent_strong_uptrend_is_long():
    agent = TechnicalIndiaAgent()
    vote = agent.analyze("NIFTY", _klines(60, "up"), Market.INDIA)
    assert vote.direction == Direction.LONG
    assert vote.confidence >= 0.5


def test_technical_agent_strong_downtrend_is_short():
    agent = TechnicalIndiaAgent()
    vote = agent.analyze("NIFTY", _klines(60, "down"), Market.INDIA)
    assert vote.direction == Direction.SHORT
    assert vote.confidence >= 0.5


def test_technical_agent_insufficient_data_returns_hold():
    agent = TechnicalIndiaAgent()
    vote = agent.analyze("NIFTY", _klines(5), Market.INDIA)
    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0


def test_technical_agent_never_raises():
    agent = TechnicalIndiaAgent()
    # Empty klines — must not raise
    vote = agent.analyze("NIFTY", [], Market.INDIA)
    assert vote.direction == Direction.HOLD
