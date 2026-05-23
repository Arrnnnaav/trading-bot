from brokers.coindcx import CoinDCXBroker
from brokers.upstox import UpstoxBroker


def test_coindcx_get_price_returns_float(monkeypatch):
    broker = CoinDCXBroker(api_key="test", api_secret="test")
    monkeypatch.setattr(
        broker, "_fetch_ticker", lambda ticker: {"last_price": "62400.5"}
    )
    price = broker.get_price("BTCUSDT")
    assert isinstance(price, float)
    assert price == 62400.5


def test_coindcx_place_order_returns_order_id(monkeypatch):
    broker = CoinDCXBroker(api_key="test", api_secret="test")
    monkeypatch.setattr(
        broker, "_fetch_ticker", lambda ticker: {"last_price": "62400.0"}
    )
    monkeypatch.setattr(
        broker,
        "_create_order",
        lambda **kw: {"id": "order_123", "status": "filled", "avg_price": 62400.0},
    )
    result = broker.place_order(ticker="BTCUSDT", side="buy", size_inr=2000.0)
    assert result["order_id"] == "order_123"
    assert result["fill_price"] == 62400.0


def test_upstox_get_price_returns_float(monkeypatch):
    broker = UpstoxBroker(api_key="test", access_token="test")
    monkeypatch.setattr(broker, "_fetch_ltp", lambda token: 22450.5)
    price = broker.get_price("NSE_INDEX|Nifty 50")
    assert price == 22450.5


def test_upstox_get_atm_strike_nifty():
    broker = UpstoxBroker(api_key="test", access_token="test")
    strike = broker._calc_atm_strike(spot=22483.0, step=50)
    assert strike == 22500.0


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
