"""Broker error-case tests — timeouts raise, bad responses raise, never return None."""

import httpx
import pytest
from unittest.mock import MagicMock

from brokers.upstox import UpstoxBroker
from brokers.paper_broker import PaperBroker


# ── Upstox ───────────────────────────────────────────────────────────────────


def test_upstox_get_price_raises_on_timeout(monkeypatch):
    broker = UpstoxBroker(api_key="k", access_token="t")
    monkeypatch.setattr(
        "brokers.upstox.httpx.get",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.TimeoutException("timeout")),
    )
    with pytest.raises(httpx.TimeoutException):
        broker.get_price("NSE_INDEX|Nifty 50")


def test_upstox_get_price_raises_on_401(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401", request=MagicMock(), response=MagicMock()
    )
    monkeypatch.setattr("brokers.upstox.httpx.get", lambda *a, **kw: mock_resp)
    broker = UpstoxBroker(api_key="k", access_token="bad_token")
    with pytest.raises(httpx.HTTPStatusError):
        broker.get_price("NSE_INDEX|Nifty 50")


# ── PaperBroker ──────────────────────────────────────────────────────────────


def test_paper_broker_place_order_raises_when_price_unavailable(tmp_path, monkeypatch):
    real = MagicMock()
    real.get_price.side_effect = RuntimeError("exchange down")
    broker = PaperBroker(real, log_path=str(tmp_path / "trades.json"))
    with pytest.raises(RuntimeError, match="exchange down"):
        broker.place_order("BTCUSDT", "buy", 5000.0)


def test_paper_broker_place_order_uses_provided_price(tmp_path):
    real = MagicMock()
    real.get_price.return_value = 100.0
    broker = PaperBroker(real, log_path=str(tmp_path / "trades.json"))
    result = broker.place_order("BTCUSDT", "buy", 1000.0, price=200.0)
    assert result["fill_price"] == 200.0
    assert result["quantity"] == pytest.approx(5.0)
    real.get_price.assert_not_called()
