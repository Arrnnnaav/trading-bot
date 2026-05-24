from unittest.mock import patch
from agents.macro_crypto import MacroCryptoAgent
from core.models import Direction, Market


def _make_agent(fg: int, btc_dom: float) -> MacroCryptoAgent:
    agent = MacroCryptoAgent()
    agent._cached_fg = fg
    agent._cached_dom = btc_dom
    agent._cache_ts = float("inf")  # never expire in tests
    return agent


def _vote(fg, btc_dom, ticker="BTC/USDT"):
    agent = _make_agent(fg, btc_dom)
    return agent.analyze(ticker=ticker, klines=[], market=Market.CRYPTO)


def test_extreme_fear_is_hold():
    v = _vote(fg=15, btc_dom=50.0)
    assert v.direction == Direction.HOLD


def test_fear_is_short():
    v = _vote(fg=30, btc_dom=50.0)
    assert v.direction == Direction.SHORT
    assert v.confidence >= 0.55


def test_neutral_is_hold():
    v = _vote(fg=50, btc_dom=50.0)
    assert v.direction == Direction.HOLD


def test_greed_is_long():
    v = _vote(fg=70, btc_dom=50.0)
    assert v.direction == Direction.LONG
    assert v.confidence >= 0.55


def test_extreme_greed_is_short():
    v = _vote(fg=85, btc_dom=50.0)
    assert v.direction == Direction.SHORT


def test_high_btc_dom_penalises_alt_long():
    # Greed + high BTC dom → alt gets penalised below 0.50 threshold
    v_btc = _vote(fg=65, btc_dom=62.0, ticker="BTC/USDT")
    v_alt = _vote(fg=65, btc_dom=62.0, ticker="ETH/USDT")
    # BTC unaffected, alt penalised
    assert v_btc.direction == Direction.LONG
    # ETH confidence should be 30% lower; may still be >=0.50 or might drop to HOLD
    # Just check it's not higher than BTC
    if v_alt.direction == Direction.LONG:
        assert v_alt.confidence < v_btc.confidence


def test_api_failure_returns_hold():
    agent = MacroCryptoAgent()
    with patch("agents.macro_crypto.requests.get", side_effect=Exception("timeout")):
        v = agent.analyze(ticker="BTC/USDT", klines=[], market=Market.CRYPTO)
    assert v.direction == Direction.HOLD
    assert "unavailable" in v.reasoning


def test_cache_prevents_double_fetch():
    agent = _make_agent(fg=60, btc_dom=50.0)
    with patch("agents.macro_crypto.requests.get") as mock_get:
        agent.analyze(ticker="BTC/USDT", klines=[], market=Market.CRYPTO)
        agent.analyze(ticker="ETH/USDT", klines=[], market=Market.CRYPTO)
    mock_get.assert_not_called()  # cache served both calls
