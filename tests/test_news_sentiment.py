from unittest.mock import patch, MagicMock
from core.models import Direction, Market


def make_articles(sentiments: list[str]) -> dict:
    """Helper to create mock article response."""
    return {
        "Data": [{"SENTIMENT": s, "TITLE": f"Article about {s}"} for s in sentiments]
    }


def test_positive_articles_return_long():
    from agents.news_sentiment import NewsSentimentAgent

    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = make_articles(["POSITIVE"] * 8 + ["NEUTRAL"] * 2)
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.LONG
    assert vote.confidence > 0.5


def test_negative_articles_return_short():
    from agents.news_sentiment import NewsSentimentAgent

    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = make_articles(["NEGATIVE"] * 8 + ["NEUTRAL"] * 2)
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.SHORT
    assert vote.confidence > 0.5


def test_mixed_articles_return_hold():
    from agents.news_sentiment import NewsSentimentAgent

    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = make_articles(
        ["POSITIVE"] * 3 + ["NEGATIVE"] * 3 + ["NEUTRAL"] * 4
    )
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0


def test_api_failure_returns_hold():
    from agents.news_sentiment import NewsSentimentAgent

    agent = NewsSentimentAgent()

    with patch("httpx.get", side_effect=Exception("connection error")):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0
    assert "failed" in vote.reasoning.lower()


def test_empty_articles_return_hold():
    from agents.news_sentiment import NewsSentimentAgent

    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"Data": []}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        vote = agent.analyze("BTC/USDT", [], Market.CRYPTO)

    assert vote.direction == Direction.HOLD


def test_ticker_to_currency_mapping():
    from agents.news_sentiment import NewsSentimentAgent

    agent = NewsSentimentAgent()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"Data": []}
    mock_resp.raise_for_status = MagicMock()

    captured_params = {}

    def capture_get(url, params=None, timeout=None):
        captured_params.update(params or {})
        return mock_resp

    with patch("httpx.get", side_effect=capture_get):
        agent.analyze("ETH/USDT", [], Market.CRYPTO)

    assert captured_params.get("categories") == "ETH"
