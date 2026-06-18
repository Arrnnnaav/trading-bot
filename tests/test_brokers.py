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
