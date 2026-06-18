import time
from datetime import date, timedelta

import httpx
from brokers.base_broker import BrokerBase
from config import config

_HTTP_TIMEOUT = 10  # seconds per request
_MAX_RETRIES = 3
_RETRY_BACKOFF = (1, 2)  # seconds between attempts 1→2 and 2→3


def _get(url: str, **kwargs) -> httpx.Response:
    kwargs.setdefault("timeout", _HTTP_TIMEOUT)
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = httpx.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < len(_RETRY_BACKOFF):
                time.sleep(_RETRY_BACKOFF[attempt])
    raise last_exc  # type: ignore[misc]


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
        resp = _get(
            f"{self.BASE_URL}/market-quote/ltp",
            params={"instrument_key": instrument_token},
            headers=self._headers,
        )
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
        today = date.today()
        from_date = "2024-01-01"
        to_date = today.isoformat()
        resp = _get(
            f"{self.BASE_URL}/historical-candle/{instrument_token}/{interval}/{from_date}/{to_date}",
            headers=self._headers,
        )
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

    _VALID_INTRADAY_INTERVALS = {
        "1minute",
        "5minute",
        "15minute",
        "30minute",
        "60minute",
    }

    def _get_intraday_candles(
        self, instrument: str, interval: str, from_date: str, to_date: str
    ) -> list:
        resp = _get(
            f"{self.BASE_URL}/historical-candle/intraday/{instrument}/{interval}/{from_date}/{to_date}",
            headers=self._headers,
        )
        return resp.json()["data"]["candles"]

    def get_ohlcv_intraday(
        self, instrument: str, interval: str = "15minute", days_back: int = 5
    ) -> list[dict]:
        if interval not in self._VALID_INTRADAY_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. Valid: {sorted(self._VALID_INTRADAY_INTERVALS)}"
            )
        to_date = date.today().isoformat()
        from_date = (date.today() - timedelta(days=days_back)).isoformat()
        candles = self._get_intraday_candles(instrument, interval, from_date, to_date)
        return [
            {
                "time": c[0],
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            }
            for c in candles
        ]

    def get_options_chain(self, index: str, expiry: str) -> list[dict]:
        spec = self._INDEX_SPEC.get(index.upper())
        if spec is None:
            raise ValueError(
                f"Unknown index '{index}'. Supported: {list(self._INDEX_SPEC)}"
            )
        token = spec["spot"]
        resp = _get(
            f"{self.BASE_URL}/option/chain",
            params={"instrument_key": token, "expiry_date": expiry},
            headers=self._headers,
        )
        return resp.json()["data"]

    @staticmethod
    def _next_weekly_expiry(today: date | None = None) -> str:
        today = today or date.today()
        # Default to the next Thursday weekly expiry. Exchange holidays still need
        # broker-chain validation before live trading.
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0:
            days_until_thursday = 7
        return (today + timedelta(days=days_until_thursday)).isoformat()

    def resolve_options_expiry(self, expiry: str | None = None) -> str:
        return expiry or config.india_options_expiry or self._next_weekly_expiry()

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

    # Lot sizes and strike steps per SEBI/NSE effective 2025-2026.
    # Verify at https://www.nseindia.com before live trading — SEBI revises periodically.
    _INDEX_SPEC: dict[str, dict] = {
        "NIFTY": {
            "spot": "NSE_INDEX|Nifty 50",
            "step": 50,
            "lot": 65,
            "exchange": "NSE_FO",
        },
        "BANKNIFTY": {
            "spot": "NSE_INDEX|Nifty Bank",
            "step": 100,
            "lot": 30,
            "exchange": "NSE_FO",
        },
        "SENSEX": {
            "spot": "BSE_INDEX|SENSEX",
            "step": 100,
            "lot": 20,
            "exchange": "BSE_FO",
        },
        "NIFTYIT": {
            "spot": "NSE_INDEX|Nifty IT",
            "step": 50,
            "lot": 35,
            "exchange": "NSE_FO",
        },
    }

    def place_options_order(
        self, index: str, direction: str, expiry: str | None, size_inr: float
    ) -> dict:
        spec = self._INDEX_SPEC.get(index.upper())
        if spec is None:
            raise ValueError(
                f"Unknown index '{index}'. Supported: {list(self._INDEX_SPEC)}"
            )
        expiry = self.resolve_options_expiry(expiry)
        spot = self.get_price(spec["spot"])
        strike = self._calc_atm_strike(spot, spec["step"])
        option_type = "CE" if direction == "LONG" else "PE"
        token = f"{spec['exchange']}|{index.upper()}{expiry.replace('-', '')}{int(strike):05d}{option_type}"
        ltp = self._fetch_ltp(token)
        lot_size = spec["lot"]
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
