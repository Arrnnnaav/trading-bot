"""
PaperBroker — wraps any BrokerBase and intercepts place_order / place_options_order.
Instead of hitting the real API, it logs the order to paper_trades.json and returns
a fake order ID.  get_price / get_ohlcv / close_position pass through to the real broker
so signals are based on live data.

Usage:
    real_broker = CoinDCXBroker(key, secret)
    broker = PaperBroker(real_broker, log_path="data/paper_trades.json")
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from brokers.base_broker import BrokerBase
from core.models import Direction


class PaperBroker(BrokerBase):
    def __init__(
        self, real_broker: BrokerBase, log_path: str = "data/paper_trades.json"
    ):
        self._real = real_broker
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_path.exists():
            self._log_path.write_text("[]")

    # ── pass-through methods ──────────────────────────────────────────────────

    def get_price(self, ticker: str) -> float:
        return self._real.get_price(ticker)

    def get_ohlcv(self, ticker: str, interval: str = "15m", limit: int = 100) -> list:
        return self._real.get_ohlcv(ticker, interval=interval, limit=limit)

    def close_position(
        self, ticker: str, direction: Direction, quantity: float
    ) -> dict:
        order_id = f"paper-close-{uuid.uuid4().hex[:8]}"
        self._append_trade(
            {
                "order_id": order_id,
                "type": "close",
                "ticker": ticker,
                "direction": direction.value
                if hasattr(direction, "value")
                else str(direction),
                "quantity": quantity,
            }
        )
        return {"order_id": order_id, "status": "paper"}

    # ── intercepted methods ───────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        direction: Direction,
        quantity: float,
        price: float | None = None,
    ) -> dict:
        order_id = f"paper-{uuid.uuid4().hex[:8]}"
        self._append_trade(
            {
                "order_id": order_id,
                "type": "spot",
                "ticker": ticker,
                "direction": direction.value
                if hasattr(direction, "value")
                else str(direction),
                "quantity": quantity,
                "price": price,
            }
        )
        return {"order_id": order_id, "status": "paper"}

    def place_options_order(
        self,
        ticker: str,
        direction: Direction,
        quantity: int,
        expiry: str,
        strike: float | None = None,
    ) -> dict:
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
                "expiry": expiry,
                "strike": strike,
            }
        )
        return {"order_id": order_id, "status": "paper"}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _append_trade(self, entry: dict) -> None:
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            trades = json.loads(self._log_path.read_text())
        except Exception:
            trades = []
        trades.append(entry)
        self._log_path.write_text(json.dumps(trades, indent=2))
