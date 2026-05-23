import hashlib
import hmac
import json
import time
import httpx
from brokers.base_broker import BrokerBase


class CoinDCXBroker(BrokerBase):
    BASE_URL = "https://api.coindcx.com"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def _sign(self, body: dict) -> str:
        body_str = json.dumps(body, separators=(",", ":"))
        return hmac.new(
            self.api_secret.encode(), body_str.encode(), hashlib.sha256
        ).hexdigest()

    def _fetch_ticker(self, ticker: str) -> dict:
        resp = httpx.get(f"{self.BASE_URL}/exchange/ticker")
        resp.raise_for_status()
        for t in resp.json():
            if t["market"] == ticker:
                return t
        raise ValueError(f"Ticker {ticker} not found")

    def get_price(self, ticker: str) -> float:
        data = self._fetch_ticker(ticker)
        return float(data["last_price"])

    def get_ohlcv(
        self, ticker: str, interval: str = "5m", limit: int = 60
    ) -> list[dict]:
        candle_resp = httpx.get(
            "https://public.coindcx.com/market_data/candles",
            params={"pair": ticker, "interval": interval, "limit": limit},
        )
        candle_resp.raise_for_status()
        return candle_resp.json()

    def _create_order(self, **kwargs) -> dict:
        body = {"timestamp": int(time.time() * 1000), **kwargs}
        sig = self._sign(body)
        resp = httpx.post(
            f"{self.BASE_URL}/exchange/v1/orders/create",
            json=body,
            headers={"X-AUTH-APIKEY": self.api_key, "X-AUTH-SIGNATURE": sig},
        )
        resp.raise_for_status()
        return resp.json()

    def place_order(self, ticker: str, side: str, size_inr: float) -> dict:
        price = self.get_price(ticker)
        quantity = round(size_inr / price, 6)
        result = self._create_order(
            market=ticker, side=side, order_type="market_order", total_quantity=quantity
        )
        return {
            "order_id": result["id"],
            "fill_price": float(result.get("avg_price", price)),
            "status": result.get("status", "filled"),
        }

    def close_position(
        self, broker_order_id: str, ticker: str, side: str, quantity: float = 0.0
    ) -> dict:
        close_side = "sell" if side == "buy" else "buy"
        result = self._create_order(
            market=ticker,
            side=close_side,
            order_type="market_order",
            total_quantity=quantity,
            client_order_id=f"close_{broker_order_id}",
        )
        return {"order_id": result["id"], "status": result.get("status")}
