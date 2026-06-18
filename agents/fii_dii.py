"""FIIDIIAgent — reads cached nsefin daily FII/DII flow files."""

import json
import logging
from pathlib import Path

from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

_LOG = logging.getLogger(__name__)
_DEFAULT_DATA_DIR = Path("data/fii_dii")


class FIIDIIAgent(BaseAgent):
    name = "fii_dii"

    def __init__(self, data_dir: Path | str = _DEFAULT_DATA_DIR):
        self._data_dir = Path(data_dir)

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        try:
            return self._compute()
        except Exception as exc:
            _LOG.warning("FIIDIIAgent failed: %s", exc)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"error: {exc}",
            )

    def _load_recent(self, days: int = 5) -> list[dict]:
        files = sorted(self._data_dir.glob("*.json"))[-days:]
        records = []
        for f in files:
            try:
                records.append(json.loads(f.read_text()))
            except Exception:
                pass
        return records

    def _compute(self) -> AgentVote:
        records = self._load_recent(5)
        if not records:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="no FII/DII data available",
            )

        fii_flows = [r["fii_net_cr"] for r in records]
        dii_flows = [r["dii_net_cr"] for r in records]
        rolling_fii = sum(fii_flows)
        rolling_dii = sum(dii_flows)
        avg_fii = rolling_fii / len(fii_flows)
        today_fii = fii_flows[-1]

        # Strong FII buying + accelerating
        if rolling_fii > 2000 and today_fii >= avg_fii:
            confidence = min(0.5 + rolling_fii / 10000, 0.85)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.LONG,
                confidence=round(confidence, 2),
                reasoning=f"FII 5d net={rolling_fii:.0f}Cr, accelerating (today={today_fii:.0f}Cr)",
            )

        # Sustained FII selling ≥3 days
        consecutive_selling = sum(1 for f in fii_flows if f < 0)
        if rolling_fii < -2000 and consecutive_selling >= 3:
            # But check if DII is providing strong floor support
            if rolling_dii <= abs(rolling_fii) / 2:
                return AgentVote(
                    agent_name=self.name,
                    direction=Direction.SHORT,
                    confidence=0.72,
                    reasoning=f"FII sustained selling: {consecutive_selling} days, net={rolling_fii:.0f}Cr",
                )

        # DII buying while FII selling — floor signal
        if rolling_fii < 0 and rolling_dii > 0:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"DII floor: FII={rolling_fii:.0f}Cr, DII={rolling_dii:.0f}Cr",
            )

        return AgentVote(
            agent_name=self.name,
            direction=Direction.HOLD,
            confidence=0.0,
            reasoning=f"FII 5d net={rolling_fii:.0f}Cr — no clear signal",
        )
