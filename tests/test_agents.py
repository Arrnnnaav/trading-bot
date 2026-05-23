from core.models import Direction, Market
from agents.kronos_technical import KronosTechnicalAgent


def test_kronos_returns_agent_vote(monkeypatch):
    agent = KronosTechnicalAgent()
    fake_embedding = [0.1] * 768
    monkeypatch.setattr(agent, "_encode_klines", lambda klines: fake_embedding)
    monkeypatch.setattr(
        agent,
        "_pattern_match",
        lambda emb: {
            "direction": "LONG",
            "confidence": 0.72,
            "matched": 8,
            "total": 10,
        },
    )
    monkeypatch.setattr(
        agent, "_forecast", lambda emb: [62500, 63000, 63800, 62900, 64000]
    )

    klines = [
        {"open": 62000, "high": 63000, "low": 61500, "close": 62400, "volume": 1200}
        for _ in range(60)
    ]
    vote = agent.analyze(ticker="BTC/USDT", klines=klines, market=Market.CRYPTO)

    assert vote.direction == Direction.LONG
    assert 0.0 <= vote.confidence <= 1.0
    assert vote.agent_name == "KronosTechnical"
    assert "pattern" in vote.reasoning.lower() or "match" in vote.reasoning.lower()


def test_kronos_low_confidence_returns_hold(monkeypatch):
    agent = KronosTechnicalAgent()
    monkeypatch.setattr(agent, "_encode_klines", lambda klines: [0.0] * 768)
    monkeypatch.setattr(
        agent,
        "_pattern_match",
        lambda emb: {
            "direction": "LONG",
            "confidence": 0.30,
            "matched": 3,
            "total": 10,
        },
    )
    monkeypatch.setattr(
        agent, "_forecast", lambda emb: [62000, 61900, 61800, 61700, 61600]
    )

    klines = [
        {"open": 62000, "high": 62100, "low": 61900, "close": 62000, "volume": 500}
        for _ in range(60)
    ]
    vote = agent.analyze(ticker="BTC/USDT", klines=klines, market=Market.CRYPTO)
    assert vote.direction == Direction.HOLD
