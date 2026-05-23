import httpx
from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market


class FundamentalsAgent(BaseAgent):
    name = "Fundamentals"

    def _fetch_nse_data(self, ticker: str) -> dict:
        symbol = ticker.replace(".NS", "").replace("/", "")
        resp = httpx.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={symbol}",
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
            data = self._fetch_nse_data(ticker)
            price_info = data.get("priceInfo", {})
            pe = float(data.get("metadata", {}).get("pe", 0) or 0)
            delivery_pct = float(price_info.get("deliveryToTradedQuantity", 0) or 0)
        except Exception:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="NSE data fetch failed",
            )

        signals = []
        if delivery_pct > 60:
            signals.append(("LONG", 0.65))
        if pe > 0 and pe < 25:
            signals.append(("LONG", 0.60))
        if delivery_pct < 30:
            signals.append(("SHORT", 0.60))

        if not signals:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"P/E: {pe:.1f} | Delivery: {delivery_pct:.1f}%",
            )

        longs = [(d, c) for d, c in signals if d == "LONG"]
        shorts = [(d, c) for d, c in signals if d == "SHORT"]
        if len(longs) >= len(shorts):
            direction = Direction.LONG
            confidence = max(c for _, c in longs)
        else:
            direction = Direction.SHORT
            confidence = max(c for _, c in shorts)

        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=f"P/E: {pe:.1f} | Delivery: {delivery_pct:.1f}%",
        )
