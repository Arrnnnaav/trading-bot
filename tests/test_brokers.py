import pytest

from brokers.upstox import UpstoxBroker


def test_upstox_get_price_returns_float(monkeypatch):
    broker = UpstoxBroker(api_key="test", access_token="test")
    monkeypatch.setattr(broker, "_fetch_ltp", lambda token: 22450.5)
    price = broker.get_price("NSE_INDEX|Nifty 50")
    assert price == 22450.5


def test_upstox_get_atm_strike_nifty():
    broker = UpstoxBroker(api_key="test", access_token="test")
    strike = broker._calc_atm_strike(spot=22483.0, step=50)
    assert strike == 22500.0


def test_upstox_resolves_next_weekly_expiry():
    from datetime import date

    assert UpstoxBroker._next_weekly_expiry(date(2026, 6, 14)) == "2026-06-18"


def test_upstox_place_options_order_returns_order_id(monkeypatch):
    broker = UpstoxBroker(api_key="test", access_token="test")
    monkeypatch.setattr(
        broker,
        "_place_options_order",
        lambda **kw: {"order_id": "upstox_456", "fill_price": 120.0},
    )
    monkeypatch.setattr(broker, "_fetch_ltp", lambda token: 120.0)
    monkeypatch.setattr(broker, "get_price", lambda token: 22500.0)
    result = broker.place_options_order(
        index="NIFTY", direction="LONG", expiry="2026-05-30", size_inr=2000.0
    )
    assert result["order_id"] == "upstox_456"


def test_upstox_get_ohlcv_intraday_returns_candles(monkeypatch):
    broker = UpstoxBroker(api_key="test", access_token="test")
    mock_candles = [
        ["2026-06-18T09:20:00+05:30", 22400.0, 22450.0, 22380.0, 22430.0, 12345]
    ]
    monkeypatch.setattr(
        broker,
        "_get_intraday_candles",
        lambda instrument, interval, from_date, to_date: mock_candles,
    )
    result = broker.get_ohlcv_intraday("NSE_INDEX|Nifty 50", "15minute", days_back=2)
    assert len(result) == 1
    assert result[0]["close"] == 22430.0
    assert "time" in result[0]


def test_upstox_get_ohlcv_intraday_invalid_interval(monkeypatch):
    broker = UpstoxBroker(api_key="test", access_token="test")
    with pytest.raises(ValueError, match="interval"):
        broker.get_ohlcv_intraday("NSE_INDEX|Nifty 50", "3minute")


# ──────────────────────────────────────────────────────────────────────────────
# ZerodhaBroker tests
# ──────────────────────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch  # noqa: E402

from brokers.zerodha import ZerodhaBroker, _INDEX_SPEC  # noqa: E402


def _make_zerodha() -> ZerodhaBroker:
    with patch("brokers.zerodha.KiteConnect") as MockKite:
        MockKite.return_value = MagicMock()
        broker = ZerodhaBroker(api_key="test_key", access_token="test_token")
    return broker


def test_zerodha_index_spec_has_all_four_indices():
    assert set(_INDEX_SPEC.keys()) == {"NIFTY", "BANKNIFTY", "SENSEX", "NIFTYIT"}


def test_zerodha_index_spec_lot_sizes():
    assert _INDEX_SPEC["NIFTY"]["lot"] == 65
    assert _INDEX_SPEC["BANKNIFTY"]["lot"] == 30
    assert _INDEX_SPEC["SENSEX"]["lot"] == 20
    assert _INDEX_SPEC["NIFTYIT"]["lot"] == 35


def test_zerodha_next_weekly_expiry_lands_on_thursday():
    from datetime import date

    result = ZerodhaBroker._next_weekly_expiry(date(2026, 6, 16))  # Monday
    assert date.fromisoformat(result).weekday() == 3  # Thursday


def test_zerodha_calc_atm_strike_rounds_to_step():
    broker = _make_zerodha()
    assert broker._calc_atm_strike(22483.0, 50) == 22500.0
    assert broker._calc_atm_strike(22424.0, 50) == 22400.0
    assert broker._calc_atm_strike(46340.0, 100) == 46300.0


def test_zerodha_get_price_returns_float(monkeypatch):
    broker = _make_zerodha()
    broker.kite.ltp = MagicMock(return_value={"NSE:NIFTY 50": {"last_price": 22500.75}})
    price = broker.get_price("NSE:NIFTY 50")
    assert price == 22500.75


def test_zerodha_get_ohlcv_returns_candle_list(monkeypatch):
    from datetime import datetime

    broker = _make_zerodha()
    broker.kite.historical_data = MagicMock(
        return_value=[
            {
                "date": datetime(2026, 6, 18, 9, 15),
                "open": 22400.0,
                "high": 22450.0,
                "low": 22380.0,
                "close": 22430.0,
                "volume": 12345,
            }
        ]
    )
    result = broker.get_ohlcv(256265, "day", limit=10)
    assert len(result) == 1
    assert result[0]["close"] == 22430.0
    assert "time" in result[0]


def test_zerodha_get_ohlcv_intraday_invalid_interval():
    broker = _make_zerodha()
    with pytest.raises(ValueError, match="interval"):
        broker.get_ohlcv_intraday(256265, "3minute")


def test_zerodha_require_spec_raises_on_unknown_index():
    with pytest.raises(ValueError, match="Unknown index"):
        ZerodhaBroker._require_spec("MIDCAP")


def test_zerodha_place_order_raises_not_implemented():
    broker = _make_zerodha()
    with pytest.raises(NotImplementedError):
        broker.place_order("NSE:NIFTY 50", "BUY", 10000.0)
