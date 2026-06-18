from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from core.stop_monitor import StopMonitor
from core.models import Position, Market, Direction


def _position(ticker="NIFTY", is_intraday=True):
    opened = datetime.now(timezone.utc).isoformat()
    return Position(
        signal_id="sig_x",
        ticker=ticker,
        market=Market.INDIA,
        direction=Direction.LONG,
        entry_price=22000.0,
        stop_price=21800.0,
        target_price=22400.0,
        size_inr=10000.0,
        opened_at=opened,
        broker_order_id="ord_x",
        # intraday positions have no expires_at (positional positions do)
        expires_at=None if is_intraday else "2026-06-25T15:30:00+05:30",
    )


def _monitor(force_exit="15:15"):
    m = StopMonitor(
        crypto_broker=None,
        india_broker=MagicMock(),
        harness_states={},
        telegram_bot=MagicMock(),
        intraday_force_exit_time=force_exit,
    )
    return m


def test_past_force_exit_returns_true_after_1515():
    monitor = _monitor(force_exit="15:15")
    position = _position(is_intraday=True)
    # Simulate IST time at 15:20
    ist_now = datetime(
        2026, 6, 18, 15, 20, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    with patch("core.stop_monitor.datetime") as mock_dt:
        mock_dt.now.return_value = ist_now
        mock_dt.fromisoformat = datetime.fromisoformat
        assert monitor._is_past_force_exit(position) is True


def test_past_force_exit_returns_false_before_1515():
    monitor = _monitor(force_exit="15:15")
    position = _position(is_intraday=True)
    ist_now = datetime(
        2026, 6, 18, 14, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    with patch("core.stop_monitor.datetime") as mock_dt:
        mock_dt.now.return_value = ist_now
        mock_dt.fromisoformat = datetime.fromisoformat
        assert monitor._is_past_force_exit(position) is False


def test_positional_position_not_force_exited():
    monitor = _monitor(force_exit="15:15")
    position = _position(is_intraday=False)
    ist_now = datetime(
        2026, 6, 18, 15, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    with patch("core.stop_monitor.datetime") as mock_dt:
        mock_dt.now.return_value = ist_now
        mock_dt.fromisoformat = datetime.fromisoformat
        # positional has expires_at set — not subject to force exit
        assert monitor._is_past_force_exit(position) is False
