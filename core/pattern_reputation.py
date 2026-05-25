"""
PatternReputationTracker — fingerprints signal patterns and tracks their historical
win rates. A pattern is a stable descriptor of a signal's context, independent of
the specific price level. Two signals with the same pattern represent the same
"setup" repeating across time.

Pattern fingerprint: "<direction>|<ticker_group>|<agent_votes_sorted>"
  direction:    LONG / SHORT
  ticker_group: BTC, ETH, ALT (SOL/BNB/XRP), INDIA
  agent_votes:  sorted list of "AgentName:DIRECTION" for agents that voted
                in agreement with the signal direction (confidence >= 0.50)

Win:  pattern direction matches signal direction AND outcome == TARGET_HIT
Loss: pattern direction matches signal direction AND outcome == STOP_HIT
HOLD votes, EXPIRED outcomes excluded.

Gate: pattern_win_rate < MIN_WIN_RATE → signal blocked if SAMPLE_THRESHOLD met.
"""

import json
from pathlib import Path
from collections import defaultdict


MIN_WIN_RATE = 0.40  # patterns below this rate get blocked
MIN_SAMPLES = 10  # need at least 10 resolved signals before blocking


_TICKER_GROUPS = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "ALT",
    "BNB": "ALT",
    "XRP": "ALT",
    "NIFTY50": "INDIA",
    "NIFTYBANK": "INDIA",
    "HDFC": "INDIA",
    "INFOSYS": "INDIA",
    "RELIANCE": "INDIA",
}


def _ticker_group(ticker: str) -> str:
    base = ticker.split("/")[0].upper()
    return _TICKER_GROUPS.get(base, "OTHER")


def _build_fingerprint(direction: str, ticker: str, agent_votes: list[dict]) -> str:
    group = _ticker_group(ticker)
    agreeing = sorted(
        f"{v['agent_name']}:{v['direction']}"
        for v in agent_votes
        if v.get("direction") == direction and v.get("confidence", 0) >= 0.50
    )
    vote_str = ",".join(agreeing) if agreeing else "none"
    return f"{direction}|{group}|{vote_str}"


class PatternReputationTracker:
    def __init__(self, log_path: str = "data/signal_log.json"):
        self.log_path = log_path
        self._stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0})
        self._load()

    def _load(self) -> None:
        path = Path(self.log_path)
        if not path.exists():
            return
        try:
            signals = json.loads(path.read_text())
        except Exception:
            return

        for sig in signals:
            outcome = sig.get("outcome", "PENDING")
            if outcome not in ("TARGET_HIT", "STOP_HIT"):
                continue
            direction = sig.get("direction", "")
            if not direction or direction == "HOLD":
                continue
            ticker = sig.get("ticker", "")
            votes = sig.get("agent_votes", [])
            fp = _build_fingerprint(direction, ticker, votes)

            if outcome == "TARGET_HIT":
                self._stats[fp]["wins"] += 1
            else:
                self._stats[fp]["losses"] += 1

    def refresh(self) -> None:
        self._stats.clear()
        self._load()

    def is_pattern_allowed(
        self, direction: str, ticker: str, agent_votes: list[dict]
    ) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        Blocks if pattern has >= MIN_SAMPLES and win_rate < MIN_WIN_RATE.
        """
        fp = _build_fingerprint(direction, ticker, agent_votes)
        s = self._stats.get(fp)
        if s is None:
            return True, "pattern new — no history"
        total = s["wins"] + s["losses"]
        if total < MIN_SAMPLES:
            return (
                True,
                f"pattern has {total}/{MIN_SAMPLES} samples — not enough history",
            )
        wr = s["wins"] / total
        if wr < MIN_WIN_RATE:
            return (
                False,
                f"pattern win rate {wr:.0%} < {MIN_WIN_RATE:.0%} over {total} signals — blocked",
            )
        return True, f"pattern win rate {wr:.0%} over {total} signals — allowed"

    def get_stats(self) -> dict:
        out = {}
        for fp, s in self._stats.items():
            total = s["wins"] + s["losses"]
            out[fp] = {
                "wins": s["wins"],
                "losses": s["losses"],
                "total": total,
                "win_rate": round(s["wins"] / total, 3) if total else None,
            }
        return out
