"""
AgentPerformanceTracker — reads signal_log.json, computes per-agent win rates,
returns vote weights and a text summary for DebateEngine.

Win definition: agent voted the same direction as the signal AND signal resolved
TARGET_HIT.  Loss: agent agreed on direction AND signal resolved STOP_HIT.
HOLD votes and EXPIRED signals are excluded from stats.

Weight formula: clip(win_rate / BASELINE, MIN_W, MAX_W)
  BASELINE = 0.50 (random binary direction on agreed signals)
  So an agent right 65% → weight 1.30; right 40% → weight 0.80.
"""

import json
from pathlib import Path
from collections import defaultdict


BASELINE = 0.50
MIN_W = 0.50
MAX_W = 2.00
MIN_SAMPLES = 20  # ignore agents with fewer resolved signals


class AgentPerformanceTracker:
    def __init__(self, log_path: str = "data/signal_log.json"):
        self.log_path = log_path
        # {agent_name: {"wins": int, "losses": int}}
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
            sig_direction = sig.get("direction", "")
            for vote in sig.get("agent_votes", []):
                name = vote.get("agent_name", "")
                vote_dir = vote.get("direction", "HOLD")
                if vote_dir == "HOLD" or vote_dir != sig_direction:
                    continue  # only count votes that agreed with the signal
                if outcome == "TARGET_HIT":
                    self._stats[name]["wins"] += 1
                else:
                    self._stats[name]["losses"] += 1

    def refresh(self) -> None:
        self._stats.clear()
        self._load()

    def get_weights(self) -> dict[str, float]:
        weights = {}
        for name, s in self._stats.items():
            total = s["wins"] + s["losses"]
            if total < MIN_SAMPLES:
                continue
            wr = s["wins"] / total
            weight = round(min(max(wr / BASELINE, MIN_W), MAX_W), 3)
            weights[name] = weight
        return weights

    def get_performance_notes(self) -> str:
        lines = []
        for name, s in self._stats.items():
            total = s["wins"] + s["losses"]
            if total < MIN_SAMPLES:
                continue
            wr = s["wins"] / total
            rating = "strong" if wr >= 0.60 else ("weak" if wr < 0.40 else "average")
            lines.append(f"  {name}: {wr:.0%} win rate over {total} signals ({rating})")
        if not lines:
            return ""
        return "Agent track records:\n" + "\n".join(lines)

    def stats_summary(self) -> dict:
        out = {}
        for name, s in self._stats.items():
            total = s["wins"] + s["losses"]
            out[name] = {
                "wins": s["wins"],
                "losses": s["losses"],
                "total": total,
                "win_rate": round(s["wins"] / total, 3) if total else None,
                "weight": self.get_weights().get(name, 1.0),
            }
        return out
