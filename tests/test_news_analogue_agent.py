from unittest.mock import patch
from agents.news_analogue import NewsAnalogueAgent
from core.models import Direction, Market


def _klines():
    return [
        {"open": 100, "high": 105, "low": 95, "close": 100, "volume": 1000}
        for _ in range(10)
    ]


def test_news_agent_no_chroma_returns_hold(tmp_path):
    agent = NewsAnalogueAgent(
        chroma_path=str(tmp_path / "chroma"), news_db_path=str(tmp_path / "news.db")
    )
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0


def test_news_agent_positive_analogue_is_long(tmp_path):
    agent = NewsAnalogueAgent(
        chroma_path=str(tmp_path / "chroma"), news_db_path=str(tmp_path / "news.db")
    )
    with patch.object(
        agent, "_fetch_headlines", return_value=["Market rally on RBI rate cut"]
    ):
        with patch.object(
            agent,
            "_query_analogues",
            return_value=[
                {"similarity": 0.85, "nifty_pct_change_5d": 3.5},
                {"similarity": 0.80, "nifty_pct_change_5d": 2.1},
            ],
        ):
            vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.LONG
    assert vote.confidence > 0.0


def test_news_agent_negative_analogue_is_short(tmp_path):
    agent = NewsAnalogueAgent(
        chroma_path=str(tmp_path / "chroma"), news_db_path=str(tmp_path / "news.db")
    )
    with patch.object(
        agent, "_fetch_headlines", return_value=["Market crash on global sell-off"]
    ):
        with patch.object(
            agent,
            "_query_analogues",
            return_value=[
                {"similarity": 0.88, "nifty_pct_change_5d": -4.2},
                {"similarity": 0.76, "nifty_pct_change_5d": -1.8},
            ],
        ):
            vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.SHORT


def test_news_agent_low_similarity_returns_hold(tmp_path):
    agent = NewsAnalogueAgent(
        chroma_path=str(tmp_path / "chroma"), news_db_path=str(tmp_path / "news.db")
    )
    with patch.object(
        agent, "_fetch_headlines", return_value=["routine market update"]
    ):
        with patch.object(
            agent,
            "_query_analogues",
            return_value=[
                {
                    "similarity": 0.60,
                    "nifty_pct_change_5d": 2.0,
                },  # below 0.75 threshold
            ],
        ):
            vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.HOLD


def test_news_agent_never_raises(tmp_path):
    agent = NewsAnalogueAgent(
        chroma_path=str(tmp_path / "chroma"), news_db_path=str(tmp_path / "news.db")
    )
    vote = agent.analyze("NIFTY", [], Market.INDIA)
    assert vote.direction == Direction.HOLD
