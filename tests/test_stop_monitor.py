from unittest.mock import MagicMock
from core.stop_monitor import StopMonitor
from core.models import Position, Market, Direction


def make_position(ticker="BTC/USDT", entry=62400, stop=61000, market=Market.CRYPTO):
    return Position(
        signal_id="sig_001",
        ticker=ticker,
        market=market,
        direction=Direction.LONG,
        entry_price=entry,
        stop_price=stop,
        target_price=65000,
        size_inr=2000.0,
        opened_at="2026-05-23T08:00:00Z",
        broker_order_id="ord_001",
    )


def test_should_close_when_stop_hit():
    monitor = StopMonitor(
        crypto_broker=MagicMock(),
        india_broker=MagicMock(),
        harness_states={},
        telegram_bot=MagicMock(),
    )
    pos = make_position(stop=61000)
    assert monitor._should_close(pos, current_price=60900.0) == True


def test_should_not_close_above_stop():
    monitor = StopMonitor(
        crypto_broker=MagicMock(),
        india_broker=MagicMock(),
        harness_states={},
        telegram_bot=MagicMock(),
    )
    pos = make_position(stop=61000)
    assert monitor._should_close(pos, current_price=62000.0) == False


def test_options_stop_40pct():
    monitor = StopMonitor(
        crypto_broker=MagicMock(),
        india_broker=MagicMock(),
        harness_states={},
        telegram_bot=MagicMock(),
    )
    pos = Position(
        signal_id="s1",
        ticker="NSE_FO|NIFTY...",
        market=Market.INDIA,
        direction=Direction.LONG,
        entry_price=100.0,  # premium paid
        stop_price=60.0,  # -40%
        target_price=200.0,
        size_inr=2000.0,
        opened_at="2026-05-23T08:00:00Z",
        broker_order_id="ord_002",
        option_type="CE",
    )
    assert monitor._should_close(pos, current_price=59.0) == True
    assert monitor._should_close(pos, current_price=61.0) == False
