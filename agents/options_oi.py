from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market


class OptionsOIAgent(BaseAgent):
    name = "OptionsOI"

    def __init__(self, upstox_broker):
        self.broker = upstox_broker

    def _calc_pcr(self, chain: list[dict]) -> float:
        total_put_oi = sum(
            float(c.get("put_options", {}).get("open_interest", 0) or 0) for c in chain
        )
        total_call_oi = sum(
            float(c.get("call_options", {}).get("open_interest", 0) or 0) for c in chain
        )
        if total_call_oi == 0:
            return 1.0
        return total_put_oi / total_call_oi

    def _calc_max_pain(self, chain: list[dict]) -> float:
        min_pain, max_pain_strike = float("inf"), 0.0
        for strike_data in chain:
            strike = float(strike_data.get("strike_price", 0))
            pain = sum(
                max(0, (s - strike))
                * float(c.get("call_options", {}).get("open_interest", 0) or 0)
                + max(0, (strike - s))
                * float(c.get("put_options", {}).get("open_interest", 0) or 0)
                for sc in chain
                for s in [float(sc.get("strike_price", 0))]
            )
            if pain < min_pain:
                min_pain, max_pain_strike = pain, strike
        return max_pain_strike

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        expiry = kwargs.get("expiry", "")
        index = (
            "NIFTY"
            if "NIFTY" in ticker.upper() and "BANK" not in ticker.upper()
            else "BANKNIFTY"
        )
        try:
            chain = self.broker.get_options_chain(index, expiry)
            pcr = self._calc_pcr(chain)
            spot = self.broker.get_price(
                "NSE_INDEX|Nifty 50" if index == "NIFTY" else "NSE_INDEX|Nifty Bank"
            )
            max_pain = self._calc_max_pain(chain)
        except Exception:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="Options chain fetch failed",
            )

        # PCR > 1.2 = heavy put writing = bullish; < 0.8 = heavy call writing = bearish
        if pcr > 1.2 and spot < max_pain:
            direction, confidence = Direction.LONG, 0.68
        elif pcr < 0.8 and spot > max_pain:
            direction, confidence = Direction.SHORT, 0.65
        else:
            direction, confidence = Direction.HOLD, 0.0

        reasoning = f"PCR: {pcr:.2f} | Max Pain: {max_pain:.0f} | Spot: {spot:.0f}"
        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
        )
