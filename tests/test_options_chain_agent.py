from unittest.mock import MagicMock
from agents.options_chain import OptionsChainAgent
from core.models import Direction, Market


def _chain_bullish():
    """High PCR (put-heavy) → contrarian LONG."""
    base_call_oi = 1000
    base_put_oi = 1400  # PCR = 1.4 > 1.3
    chain = []
    for i, strike in enumerate(range(22000, 22600, 50)):
        chain.append(
            {
                "strike_price": strike,
                "call_options": {
                    "market_data": {"oi": base_call_oi - i * 10},
                    "option_type": "CE",
                },
                "put_options": {
                    "market_data": {"oi": base_put_oi - i * 5},
                    "option_type": "PE",
                },
            }
        )
    return chain, 22200.0  # current price


def _chain_bearish():
    """Low PCR (call-heavy) → contrarian SHORT."""
    base_call_oi = 1400
    base_put_oi = 900  # PCR = 0.64 < 0.7
    chain = []
    for i, strike in enumerate(range(22000, 22600, 50)):
        chain.append(
            {
                "strike_price": strike,
                "call_options": {
                    "market_data": {"oi": base_call_oi - i * 10},
                    "option_type": "CE",
                },
                "put_options": {
                    "market_data": {"oi": base_put_oi - i * 5},
                    "option_type": "PE",
                },
            }
        )
    return chain, 22200.0


def _klines(current=22200.0):
    return [
        {
            "open": current - 5,
            "high": current + 10,
            "low": current - 10,
            "close": current,
            "volume": 1000,
        }
        for _ in range(10)
    ]


def test_options_high_pcr_is_long():
    chain, price = _chain_bullish()
    agent = OptionsChainAgent()
    vote = agent.analyze("NIFTY", _klines(price), Market.INDIA, options_chain=chain)
    assert vote.direction == Direction.LONG


def test_options_low_pcr_is_short():
    chain, price = _chain_bearish()
    agent = OptionsChainAgent()
    vote = agent.analyze("NIFTY", _klines(price), Market.INDIA, options_chain=chain)
    assert vote.direction == Direction.SHORT


def test_options_no_chain_no_broker_returns_hold():
    agent = OptionsChainAgent(broker=None)
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA)
    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0


def test_options_broker_called_when_no_chain(monkeypatch):
    mock_broker = MagicMock()
    chain, _ = _chain_bullish()
    mock_broker.get_options_chain.return_value = chain
    mock_broker.get_price.return_value = 22200.0
    agent = OptionsChainAgent(broker=mock_broker)
    vote = agent.analyze("NIFTY", _klines(), Market.INDIA, expiry="2026-06-26")
    mock_broker.get_options_chain.assert_called_once_with("NIFTY", "2026-06-26")
    assert vote.direction == Direction.LONG


def test_options_agent_never_raises():
    agent = OptionsChainAgent()
    vote = agent.analyze("NIFTY", [], Market.INDIA, options_chain=[])
    assert vote.direction == Direction.HOLD


def test_options_max_pain_below_price_is_short():
    """PCR neutral, but price above max_pain → score -1 total (not enough for SHORT alone, so HOLD)."""
    # Use a chain where PCR is neutral (0.7-1.3) to isolate max pain
    chain = []
    for i, strike in enumerate(range(22000, 22600, 50)):
        chain.append(
            {
                "strike_price": strike,
                "call_options": {"market_data": {"oi": 1000}},
                "put_options": {"market_data": {"oi": 1000}},  # PCR = 1.0 neutral
            }
        )
    agent = OptionsChainAgent()
    # Current price well above max pain → score = -1 → HOLD (not enough for SHORT)
    vote = agent.analyze("NIFTY", _klines(22550.0), Market.INDIA, options_chain=chain)
    # Max pain with equal OI at all strikes → max pain at lowest strike (22000)
    # price 22550 > 22000 * 1.01 → score -1 → HOLD
    assert vote.direction == Direction.HOLD
