import httpx
from brokers.base_broker import BrokerBase


class UpstoxBroker(BrokerBase):
    BASE_URL = "https://api.upstox.com/v2"

    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    def _fetch_ltp(self, instrument_token: str) -> float:
        resp = httpx.get(
            f"{self.BASE_URL}/market-quote/ltp",
            params={"instrument_key": instrument_token},
            headers=self._headers,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        key = list(data.keys())[0]
        return float(data[key]["last_price"])

    def get_price(self, instrument_token: str) -> float:
        return self._fetch_ltp(instrument_token)

    def _calc_atm_strike(self, spot: float, step: int = 50) -> float:
        return round(spot / step) * step

    def get_ohlcv(
        self, instrument_token: str, interval: str = "1d", limit: int = 60
    ) -> list[dict]:
        resp = httpx.get(
            f"{self.BASE_URL}/historical-candle/{instrument_token}/{interval}/2024-01-01/2026-12-31",
            headers=self._headers,
        )
        resp.raise_for_status()
        candles = resp.json()["data"]["candles"]
        return [
            {
                "time": c[0],
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            }
            for c in candles[-limit:]
        ]

    def get_options_chain(self, index: str, expiry: str) -> list[dict]:
        token = "NSE_INDEX|Nifty 50" if index == "NIFTY" else "NSE_INDEX|Nifty Bank"
        resp = httpx.get(
            f"{self.BASE_URL}/option/chain",
            params={"instrument_key": token, "expiry_date": expiry},
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def _place_options_order(self, **kwargs) -> dict:
        resp = httpx.post(
            f"{self.BASE_URL}/order/place",
            json=kwargs,
            headers={**self._headers, "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return {
            "order_id": data["order_id"],
            "fill_price": float(data.get("average_price", 0)),
        }

    def place_options_order(
        self, index: str, direction: str, expiry: str, size_inr: float
    ) -> dict:
        spot = self.get_price(
            "NSE_INDEX|Nifty 50" if index == "NIFTY" else "NSE_INDEX|Nifty Bank"
        )
        step = 50 if index == "NIFTY" else 100
        strike = self._calc_atm_strike(spot, step)
        option_type = "CE" if direction == "LONG" else "PE"
        token = f"NSE_FO|{index}{expiry.replace('-', '')}{int(strike):05d}{option_type}"
        ltp = self._fetch_ltp(token)
        lot_size = 50 if index == "NIFTY" else 15
        lots = max(1, int(size_inr / (ltp * lot_size)))
        result = self._place_options_order(
            instrument_token=token,
            transaction_type="BUY",
            order_type="LIMIT",
            price=round(ltp * 1.005, 1),
            quantity=lots * lot_size,
            product="D",
            validity="DAY",
        )
        result["strike"] = strike
        result["option_type"] = option_type
        result["instrument_token"] = token
        result["expiry"] = expiry
        return result

    def place_order(self, ticker: str, side: str, size_inr: float) -> dict:
        raise NotImplementedError("Use place_options_order for India F&O")

    def close_position(
        self, broker_order_id: str, ticker: str, side: str, quantity: float = 50.0
    ) -> dict:
        result = self._place_options_order(
            instrument_token=ticker,
            transaction_type="SELL",
            order_type="MARKET",
            quantity=int(quantity),
            product="D",
            validity="DAY",
        )
        return {"order_id": result["order_id"], "status": "closed"}
