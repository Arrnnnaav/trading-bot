import httpx
from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market


class FIIDIIAgent(BaseAgent):
    name = "FIIDII"

    def _fetch_fii_data(self) -> dict:
        resp = httpx.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        try:
            data = self._fetch_fii_data()
            entries = data.get("data", [])
            if not entries:
                raise ValueError("Empty FII data")
            latest = entries[0]
            fii_net = float(latest.get("fiiNet", 0) or 0)
            dii_net = float(latest.get("diiNet", 0) or 0)
        except Exception:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="FII/DII data unavailable",
            )

        net_flow = fii_net + dii_net
        if net_flow > 500:
            direction, confidence = Direction.LONG, 0.70
        elif net_flow < -500:
            direction, confidence = Direction.SHORT, 0.65
        else:
            direction, confidence = Direction.HOLD, 0.0

        reasoning = f"FII net: ₹{fii_net:.0f}Cr | DII net: ₹{dii_net:.0f}Cr | Total: ₹{net_flow:.0f}Cr"
        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
        )
