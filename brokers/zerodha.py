"""
Zerodha Kite Connect broker adapter.

Requires kiteconnect Python SDK (pip install kiteconnect).
Daily OAuth token must be refreshed each morning before 9:20 IST
using scripts/kite_auth.py.
"""

import logging
import threading
from datetime import date, timedelta
from typing import Optional

from kiteconnect import KiteConnect

from brokers.base_broker import BrokerBase
from config import config

_LOG = logging.getLogger(__name__)

# Kite Connect interval strings
_INTERVAL_MAP = {
    "1d": "day",
    "day": "day",
    "15minute": "15minute",
    "5minute": "5minute",
    "1minute": "minute",
    "minute": "minute",
}

# Lot sizes and strike steps per SEBI/NSE 2025-2026.
# Verify at https://www.nseindia.com before live trading.
_INDEX_SPEC: dict[str, dict] = {
    "NIFTY": {
        "exchange": "NSE",
        "tradingsymbol": "NIFTY 50",
        "spot_token": 256265,
        "fo_name": "NIFTY",  # as used in NFO instruments list
        "fo_exchange": "NFO",
        "step": 50,
        "lot": 65,
    },
    "BANKNIFTY": {
        "exchange": "NSE",
        "tradingsymbol": "NIFTY BANK",
        "spot_token": 260105,
        "fo_name": "BANKNIFTY",
        "fo_exchange": "NFO",
        "step": 100,
        "lot": 30,
    },
    "SENSEX": {
        "exchange": "BSE",
        "tradingsymbol": "SENSEX",
        "spot_token": 1,
        "fo_name": "SENSEX",
        "fo_exchange": "BFO",
        "step": 100,
        "lot": 20,
    },
    "NIFTYIT": {
        "exchange": "NSE",
        "tradingsymbol": "NIFTY IT",
        "spot_token": 259849,
        "fo_name": "NIFTYIT",
        "fo_exchange": "NFO",
        "step": 50,
        "lot": 35,
    },
}


class ZerodhaBroker(BrokerBase):
    """BrokerBase implementation for Zerodha Kite Connect."""

    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self._instruments_cache: dict[str, list] = {}
        self._instruments_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────
    # Instrument cache — downloaded once per session from Kite
    # ──────────────────────────────────────────────────────────────────────

    def _load_instruments(self, exchange: str) -> list:
        with self._instruments_lock:
            if exchange not in self._instruments_cache:
                _LOG.info("Fetching Kite instruments for %s", exchange)
                self._instruments_cache[exchange] = self.kite.instruments(exchange)
            return self._instruments_cache[exchange]

    def _find_option_instrument(
        self, fo_name: str, exchange: str, expiry: date, strike: float, option_type: str
    ) -> Optional[dict]:
        instruments = self._load_instruments(exchange)
        for inst in instruments:
            if (
                inst["name"] == fo_name
                and inst["instrument_type"] == option_type  # "CE" or "PE"
                and inst["expiry"] == expiry
                and abs(inst["strike"] - strike) < 0.5
            ):
                return inst
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Price
    # ──────────────────────────────────────────────────────────────────────

    def get_price(self, ticker: str) -> float:
        """
        ticker: "NSE:NIFTY 50" | "BSE:SENSEX" | NFO instrument symbol
        Returns last traded price.
        """
        quotes = self.kite.ltp([ticker])
        return float(quotes[ticker]["last_price"])

    def _get_spot_price(self, index: str) -> float:
        spec = self._require_spec(index)
        ticker = f"{spec['exchange']}:{spec['tradingsymbol']}"
        return self.get_price(ticker)

    # ──────────────────────────────────────────────────────────────────────
    # OHLCV — daily
    # ──────────────────────────────────────────────────────────────────────

    def get_ohlcv(
        self, instrument_token: int, interval: str = "day", limit: int = 60
    ) -> list[dict]:
        kite_interval = _INTERVAL_MAP.get(interval, interval)
        today = date.today()
        from_date = today - timedelta(days=max(limit * 2, 365))
        candles = self.kite.historical_data(
            instrument_token,
            from_date=from_date.isoformat(),
            to_date=today.isoformat(),
            interval=kite_interval,
        )
        return [
            {
                "time": str(c["date"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": int(c["volume"]),
            }
            for c in candles[-limit:]
        ]

    # ──────────────────────────────────────────────────────────────────────
    # OHLCV — intraday
    # ──────────────────────────────────────────────────────────────────────

    _VALID_INTRADAY_INTERVALS = {
        "minute",
        "5minute",
        "15minute",
        "30minute",
        "60minute",
    }

    def get_ohlcv_intraday(
        self,
        instrument_token: int,
        interval: str = "15minute",
        days_back: int = 5,
    ) -> list[dict]:
        kite_interval = _INTERVAL_MAP.get(interval, interval)
        if kite_interval not in self._VALID_INTRADAY_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. Valid: {sorted(self._VALID_INTRADAY_INTERVALS)}"
            )
        today = date.today()
        from_date = today - timedelta(days=days_back)
        candles = self.kite.historical_data(
            instrument_token,
            from_date=from_date.isoformat(),
            to_date=today.isoformat(),
            interval=kite_interval,
        )
        return [
            {
                "time": str(c["date"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": int(c["volume"]),
            }
            for c in candles
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Options chain
    # ──────────────────────────────────────────────────────────────────────

    def get_options_chain(self, index: str, expiry: str) -> list[dict]:
        spec = self._require_spec(index)
        expiry_date = date.fromisoformat(expiry)
        instruments = self._load_instruments(spec["fo_exchange"])
        chain = []
        for inst in instruments:
            if (
                inst["name"] == spec["fo_name"]
                and inst["expiry"] == expiry_date
                and inst["instrument_type"] in ("CE", "PE")
            ):
                chain.append(
                    {
                        "strike": inst["strike"],
                        "option_type": inst["instrument_type"],
                        "instrument_token": inst["instrument_token"],
                        "tradingsymbol": inst["tradingsymbol"],
                        "lot_size": inst["lot_size"],
                    }
                )
        return chain

    # ──────────────────────────────────────────────────────────────────────
    # Options order
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _next_weekly_expiry(today: Optional[date] = None) -> str:
        today = today or date.today()
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0:
            days_until_thursday = 7
        return (today + timedelta(days=days_until_thursday)).isoformat()

    def resolve_options_expiry(self, expiry: Optional[str] = None) -> str:
        return expiry or config.india_options_expiry or self._next_weekly_expiry()

    def _calc_atm_strike(self, spot: float, step: int) -> float:
        return round(spot / step) * step

    def place_options_order(
        self,
        index: str,
        direction: str,
        expiry: Optional[str],
        size_inr: float,
    ) -> dict:
        spec = self._require_spec(index)
        expiry_str = self.resolve_options_expiry(expiry)
        expiry_date = date.fromisoformat(expiry_str)

        spot = self._get_spot_price(index)
        strike = self._calc_atm_strike(spot, spec["step"])
        option_type = "CE" if direction == "LONG" else "PE"

        inst = self._find_option_instrument(
            spec["fo_name"], spec["fo_exchange"], expiry_date, strike, option_type
        )
        if inst is None:
            raise ValueError(
                f"No {option_type} instrument found for {index} strike={strike} expiry={expiry_str}"
            )

        # Fetch LTP for the option itself
        opt_ticker = f"{spec['fo_exchange']}:{inst['tradingsymbol']}"
        ltp = self.get_price(opt_ticker)
        lot_size = spec["lot"]
        lots = max(1, int(size_inr / (ltp * lot_size)))
        qty = lots * lot_size

        order_id = self.kite.place_order(
            tradingsymbol=inst["tradingsymbol"],
            exchange=spec["fo_exchange"],
            transaction_type=KiteConnect.TRANSACTION_TYPE_BUY,
            quantity=qty,
            order_type=KiteConnect.ORDER_TYPE_LIMIT,
            price=round(ltp * 1.005, 1),
            product=KiteConnect.PRODUCT_MIS,  # MIS = intraday margin; use NRML for positional
            variety=KiteConnect.VARIETY_REGULAR,
        )

        return {
            "order_id": order_id,
            "fill_price": ltp,
            "strike": strike,
            "option_type": option_type,
            "tradingsymbol": inst["tradingsymbol"],
            "instrument_token": inst["instrument_token"],
            "expiry": expiry_str,
            "lots": lots,
            "quantity": qty,
        }

    # ──────────────────────────────────────────────────────────────────────
    # BrokerBase generic methods
    # ──────────────────────────────────────────────────────────────────────

    def place_order(self, ticker: str, side: str, size_inr: float) -> dict:
        raise NotImplementedError("Use place_options_order for India F&O")

    def close_position(
        self,
        broker_order_id: str,
        ticker: str,
        side: str,
        quantity: float = 0.0,
    ) -> dict:
        # ticker must be the NSE/NFO tradingsymbol, e.g. "NIFTY24JAN23000CE"
        # Determine exchange from ticker prefix stored in open position
        exchange = "NFO"
        if ticker.startswith("BSE") or "SENSEX" in ticker.upper():
            exchange = "BFO"

        order_id = self.kite.place_order(
            tradingsymbol=ticker,
            exchange=exchange,
            transaction_type=KiteConnect.TRANSACTION_TYPE_SELL,
            quantity=int(quantity),
            order_type=KiteConnect.ORDER_TYPE_MARKET,
            product=KiteConnect.PRODUCT_MIS,
            variety=KiteConnect.VARIETY_REGULAR,
        )
        return {"order_id": order_id, "status": "closed"}

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _require_spec(index: str) -> dict:
        spec = _INDEX_SPEC.get(index.upper())
        if spec is None:
            raise ValueError(f"Unknown index '{index}'. Supported: {list(_INDEX_SPEC)}")
        return spec
