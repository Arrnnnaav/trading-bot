import json
import os
from datetime import datetime, timezone, timedelta
from core.models import Signal


class SignalAggregator:
    DEDUP_WINDOW_HOURS = 4  # don't send same ticker signal within 4hrs

    def __init__(self, signal_log_path: str = "data/signal_log.json"):
        self.signal_log_path = signal_log_path
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        os.makedirs(os.path.dirname(self.signal_log_path), exist_ok=True)
        if not os.path.exists(self.signal_log_path):
            with open(self.signal_log_path, "w") as f:
                json.dump([], f)

    def _load_log(self) -> list[dict]:
        with open(self.signal_log_path) as f:
            return json.load(f)

    def _save_log(self, data: list[dict]):
        with open(self.signal_log_path, "w") as f:
            json.dump(data, f, indent=2)

    def log_signal(self, signal: Signal):
        data = self._load_log()
        data.append(signal.model_dump())
        self._save_log(data)

    def is_duplicate(self, signal: Signal) -> bool:
        data = self._load_log()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.DEDUP_WINDOW_HOURS)
        for entry in data:
            if entry["ticker"] != signal.ticker:
                continue
            try:
                entry_time = datetime.fromisoformat(entry["generated_at"])
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                # Check if within window and outcome is PENDING
                if entry_time > cutoff and entry["outcome"] == "PENDING":
                    return True
            except Exception:
                continue
        return False

    def update_signal_outcome(
        self,
        signal_id: str,
        outcome: str,
        outcome_price: float,
        hypothetical_pnl_pct: float,
        hypothetical_pnl_inr: float,
    ):
        data = self._load_log()
        for entry in data:
            if entry["id"] == signal_id:
                entry["outcome"] = outcome
                entry["outcome_price"] = outcome_price
                entry["outcome_at"] = datetime.now(timezone.utc).isoformat()
                entry["hypothetical_pnl_pct"] = round(hypothetical_pnl_pct, 4)
                entry["hypothetical_pnl_inr"] = round(hypothetical_pnl_inr, 2)
                break
        self._save_log(data)

    def get_all_signals(self) -> list[dict]:
        return self._load_log()

    def get_pending_signals(self) -> list[dict]:
        return [s for s in self._load_log() if s["outcome"] == "PENDING"]
