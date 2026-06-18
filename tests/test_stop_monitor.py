from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
import pytest
from core.stop_monitor import StopMonitor
from core.models import Position, Market, Direction, HarnessState


def make_position(
    ticker="NIFTY",
    entry=22400,
    stop=22000,
    target=23000,
    market=Market.INDIA,
):
    return Position(
        signal_id="sig_001",
        ticker=ticker,
        market=market,
        direction=Direction.LONG,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        size_inr=2000.0,
        opened_at=datetime.now(timezone.utc).isoformat(),
        broker_order_id="ord_001",
    )


def test_should_close_when_stop_hit():
    monitor = StopMonitor(
        india_broker=MagicMock(),
        harness_states={},
        telegram_bot=MagicMock(),
        intraday_force_exit_time="23:59",
    )
    pos = make_position(entry=22400, stop=22000, target=23200)
    assert monitor._should_close(pos, current_price=21900.0) == True


def test_should_not_close_above_stop():
    monitor = StopMonitor(
        india_broker=MagicMock(),
        harness_states={},
        telegram_bot=MagicMock(),
        intraday_force_exit_time="23:59",
    )
    pos = make_position(entry=22400, stop=22000, target=23200)
    assert monitor._should_close(pos, current_price=22500.0) == False


def test_should_close_when_target_hit():
    monitor = StopMonitor(
        india_broker=MagicMock(),
        harness_states={},
        telegram_bot=MagicMock(),
        intraday_force_exit_time="23:59",
    )
    pos = make_position(entry=22400, stop=22000, target=23200)
    assert monitor._should_close(pos, current_price=23300.0) == True


def test_options_stop_40pct():
    monitor = StopMonitor(
        india_broker=MagicMock(),
        harness_states={},
        telegram_bot=MagicMock(),
        intraday_force_exit_time="23:59",
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
        opened_at=datetime.now(timezone.utc).isoformat(),
        broker_order_id="ord_002",
        option_type="CE",
    )
    assert monitor._should_close(pos, current_price=59.0) == True
    assert monitor._should_close(pos, current_price=61.0) == False


@pytest.mark.asyncio
async def test_close_persists_state_and_updates_signal(monkeypatch, tmp_path):
    pos = make_position(stop=61000)
    state = HarnessState(open_positions=[pos])
    broker = MagicMock()
    broker.get_price.return_value = 60900.0
    aggregator = MagicMock()
    telegram = MagicMock()
    telegram.send_position_closed = AsyncMock()

    from config import config

    monkeypatch.setattr(config, "india_progress_path", str(tmp_path / "india.json"))
    monitor = StopMonitor(
        india_broker=broker,
        harness_states={Market.INDIA: state},
        telegram_bot=telegram,
        signal_aggregator=aggregator,
        intraday_force_exit_time="23:59",
    )

    await monitor._check_position(pos, Market.INDIA)

    assert state.open_positions == []
    aggregator.update_signal_outcome.assert_called_once()
    assert (tmp_path / "india.json").exists()


@pytest.mark.asyncio
async def test_close_position_tolerates_none_telegram_bot(tmp_path):
    """StopMonitor must not crash when telegram_bot is None."""
    state = HarnessState()
    pos = make_position(entry=22400, stop=22000, target=23200)
    state.open_positions.append(pos)

    broker = MagicMock()
    broker.get_price.return_value = 21900.0
    broker.close_position.return_value = {"order_id": "ord_close", "status": "closed"}

    aggregator = MagicMock()
    state_path = str(tmp_path / "state.json")

    monitor = StopMonitor(
        india_broker=broker,
        harness_states={Market.INDIA: state},
        telegram_bot=None,  # key: None, not a mock
        signal_aggregator=aggregator,
        intraday_force_exit_time="23:59",
    )
    monitor._state_paths = {Market.INDIA: state_path}

    # Must not raise AttributeError
    await monitor._check_position(pos, Market.INDIA)
