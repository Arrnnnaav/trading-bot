"""
PaperBroker — wraps any BrokerBase and intercepts place_order / place_options_order.
Instead of hitting the real API, it logs the order to paper_trades.json and returns
a fake order ID.  get_price / get_ohlcv / close_position pass through to the real broker
so signals are based on live data.

Usage:
    real_broker = UpstoxBroker(api_key, access_token)
    broker = PaperBroker(real_broker, log_path="data/paper_trades.json")
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from brokers.base_broker import BrokerBase
from core.json_store import read_json, write_json_atomic
from core.models import Direction


class PaperBroker(BrokerBase):
    def __init__(
        self, real_broker: BrokerBase, log_path: str = "data/paper_trades.json"
    ):
        self._real = real_broker
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_path.exists():
            write_json_atomic(self._log_path, [])

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    # ── pass-through methods ──────────────────────────────────────────────────

    def get_price(self, ticker: str) -> float:
        return self._real.get_price(ticker)

    def get_ohlcv(self, ticker: str, interval: str = "15m", limit: int = 100) -> list:
        return self._real.get_ohlcv(ticker, interval=interval, limit=limit)

    def close_position(
        self, broker_order_id: str, ticker: str, side: str, quantity: float = 0.0
    ) -> dict:
        order_id = f"paper-close-{uuid.uuid4().hex[:8]}"
        self._append_trade(
            {
                "order_id": order_id,
                "broker_order_id": broker_order_id,
                "type": "close",
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
            }
        )
        return {"order_id": order_id, "status": "paper"}

    # ── intercepted methods ───────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: Direction | str,
        size_inr: float,
        price: float | None = None,
    ) -> dict:
        order_id = f"paper-{uuid.uuid4().hex[:8]}"
        side_value = side.value if hasattr(side, "value") else str(side)
        direction_value = {
            "buy": "LONG",
            "sell": "SHORT",
        }.get(side_value.lower(), side_value.upper())
        fill_price = (
            float(price) if price is not None else float(self._real.get_price(ticker))
        )
        quantity = round(size_inr / fill_price, 6)
        self._append_trade(
            {
                "order_id": order_id,
                "type": "spot",
                "ticker": ticker,
                "side": side_value,
                "direction": direction_value,
                "size_inr": size_inr,
                "quantity": quantity,
                "price": fill_price,
            }
        )
        return {
            "order_id": order_id,
            "status": "paper",
            "fill_price": fill_price,
            "quantity": quantity,
            "side": side_value,
        }

    def place_options_order(self, *args, **kwargs) -> dict:
        if kwargs:
            ticker = kwargs.get("index") or kwargs.get("ticker")
            direction = kwargs.get("direction")
            expiry = kwargs.get("expiry")
            size_inr = kwargs.get("size_inr")
            quantity = kwargs.get("quantity")
            strike = kwargs.get("strike")
        else:
            ticker = args[0] if len(args) > 0 else ""
            direction = args[1] if len(args) > 1 else Direction.LONG
            quantity = args[2] if len(args) > 2 else None
            expiry = args[3] if len(args) > 3 else None
            size_inr = None
            strike = args[4] if len(args) > 4 else None

        if not expiry and hasattr(self._real, "resolve_options_expiry"):
            expiry = self._real.resolve_options_expiry()

        order_id = f"paper-opt-{uuid.uuid4().hex[:8]}"
        self._append_trade(
            {
                "order_id": order_id,
                "type": "options",
                "ticker": ticker,
                "direction": direction.value
                if hasattr(direction, "value")
                else str(direction),
                "quantity": quantity,
                "size_inr": size_inr,
                "expiry": expiry,
                "strike": strike,
            }
        )
        return {
            "order_id": order_id,
            "status": "paper",
            "quantity": quantity,
            "expiry": expiry,
            "strike": strike,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _append_trade(self, entry: dict) -> None:
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        trades = read_json(self._log_path, [])
        if not isinstance(trades, list):
            trades = []
        trades.append(entry)
        write_json_atomic(self._log_path, trades)
