import asyncio
import json
import os
from datetime import datetime, timezone, timedelta

from config import config
from core.models import Market, SignalOutcome
from core.signal_aggregator import SignalAggregator

CHRONOS_OUTCOMES_PATH = os.environ.get(
    "CHRONOS_OUTCOMES_PATH", "data/chronos_outcomes.jsonl"
)


class SignalTracker:
    def __init__(self, aggregator: SignalAggregator, crypto_broker, india_broker):
        self.aggregator = aggregator
        self.brokers = {Market.CRYPTO: crypto_broker, Market.INDIA: india_broker}

    def _resolve_outcome(self, entry: dict, current_price: float) -> SignalOutcome:
        direction = entry["direction"]
        target = entry["target_price"]
        stop = entry["stop_price"]

        if direction == "LONG":
            if current_price >= target:
                return SignalOutcome.TARGET_HIT
            if current_price <= stop:
                return SignalOutcome.STOP_HIT
        else:  # SHORT
            if current_price <= target:
                return SignalOutcome.TARGET_HIT
            if current_price >= stop:
                return SignalOutcome.STOP_HIT
        return SignalOutcome.PENDING

    def _is_expired(self, entry: dict) -> bool:
        try:
            generated = datetime.fromisoformat(entry["generated_at"])
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=timezone.utc)
            market = entry["market"]
            if market == "crypto":
                expiry = generated + timedelta(hours=config.crypto_signal_expiry_hours)
            else:
                expiry = generated + timedelta(days=config.india_signal_expiry_days)
            return datetime.now(timezone.utc) > expiry
        except Exception:
            return False

    def _calc_pnl(
        self,
        direction: str,
        entry_price: float,
        outcome_price: float,
        position_size_inr: float,
    ) -> tuple[float, float]:
        if direction == "LONG":
            pct = (outcome_price - entry_price) / entry_price
        else:
            pct = (entry_price - outcome_price) / entry_price
        inr = position_size_inr * pct
        return pct, inr

    def _get_chronos_confidence(self, entry: dict) -> float:
        """Extract confidence from ChronosTechnical agent vote if present."""
        for vote in entry.get("agent_votes", []):
            if isinstance(vote, dict) and vote.get("agent_name") == "ChronosTechnical":
                return float(vote.get("confidence", 0.0))
            if hasattr(vote, "agent_name") and vote.agent_name == "ChronosTechnical":
                return float(vote.confidence)
        return 0.0

    def _write_chronos_outcome(self, entry: dict, outcome: "SignalOutcome"):
        """Append one record to chronos_outcomes.jsonl for model retraining."""
        confidence = self._get_chronos_confidence(entry)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticker": entry.get("ticker", ""),
            "predicted": entry.get("direction", "HOLD"),
            "confidence": confidence,
            "actual": outcome.value,
        }
        with open(CHRONOS_OUTCOMES_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    async def run_once(self):
        pending = self.aggregator.get_pending_signals()
        for entry in pending:
            try:
                market = Market(entry["market"])
                broker = self.brokers[market]
                current_price = broker.get_price(entry["ticker"])

                if self._is_expired(entry):
                    outcome = SignalOutcome.EXPIRED
                    outcome_price = current_price
                else:
                    outcome = self._resolve_outcome(entry, current_price)
                    outcome_price = current_price

                if outcome != SignalOutcome.PENDING:
                    pnl_pct, pnl_inr = self._calc_pnl(
                        entry["direction"],
                        entry["entry_price"],
                        outcome_price,
                        entry["position_size_inr"],
                    )
                    self.aggregator.update_signal_outcome(
                        entry["id"], outcome.value, outcome_price, pnl_pct, pnl_inr
                    )
                    # Persist for Chronos retraining
                    if entry.get("market") == "crypto":
                        self._write_chronos_outcome(entry, outcome)
            except Exception:
                continue

    async def run_forever(self, interval_seconds: int = 900):
        while True:
            await self.run_once()
            await asyncio.sleep(interval_seconds)
