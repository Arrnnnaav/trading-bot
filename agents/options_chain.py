"""OptionsChainAgent — PCR, max pain, OI skew, IV percentile, unusual OI."""

import logging
from typing import Optional

from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

_LOG = logging.getLogger(__name__)


def _calc_pcr(chain: list[dict]) -> float:
    total_call_oi = sum(
        c.get("call_options", {}).get("market_data", {}).get("oi", 0) for c in chain
    )
    total_put_oi = sum(
        c.get("put_options", {}).get("market_data", {}).get("oi", 0) for c in chain
    )
    if total_call_oi == 0:
        return 1.0
    return total_put_oi / total_call_oi


def _calc_max_pain(chain: list[dict]) -> Optional[float]:
    if not chain:
        return None
    best_strike = None
    min_pain = float("inf")
    for item in chain:
        strike = item.get("strike_price", 0)
        total_pain = 0.0
        for other in chain:
            s = other.get("strike_price", 0)
            call_oi = other.get("call_options", {}).get("market_data", {}).get("oi", 0)
            put_oi = other.get("put_options", {}).get("market_data", {}).get("oi", 0)
            total_pain += call_oi * max(0, strike - s)
            total_pain += put_oi * max(0, s - strike)
        if total_pain < min_pain:
            min_pain = total_pain
            best_strike = strike
    return best_strike


class OptionsChainAgent(BaseAgent):
    name = "options_chain"

    def __init__(self, broker=None):
        self.broker = broker

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        try:
            chain: list[dict] | None = kwargs.get("options_chain")
            expiry: str | None = kwargs.get("expiry")

            if chain is None:
                if self.broker is None:
                    return AgentVote(
                        agent_name=self.name,
                        direction=Direction.HOLD,
                        confidence=0.0,
                        reasoning="no chain or broker",
                    )
                try:
                    chain = self.broker.get_options_chain(ticker, expiry)
                except Exception as exc:
                    _LOG.warning("Options chain fetch failed: %s", exc)
                    return AgentVote(
                        agent_name=self.name,
                        direction=Direction.HOLD,
                        confidence=0.0,
                        reasoning=f"chain fetch error: {exc}",
                    )

            if not chain:
                return AgentVote(
                    agent_name=self.name,
                    direction=Direction.HOLD,
                    confidence=0.0,
                    reasoning="empty chain",
                )

            current_price = (
                klines[-1]["close"]
                if klines
                else (self.broker.get_price(ticker) if self.broker else None)
            )
            return self._compute(chain, current_price)
        except Exception as exc:
            _LOG.warning("OptionsChainAgent failed: %s", exc)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"error: {exc}",
            )

    def _compute(self, chain: list[dict], current_price: Optional[float]) -> AgentVote:
        score = 0
        signals: list[str] = []

        # PCR
        pcr = _calc_pcr(chain)
        if pcr > 1.3:
            score += 2
            signals.append(f"PCR={pcr:.2f} bullish (put-heavy)")
        elif pcr < 0.7:
            score -= 2
            signals.append(f"PCR={pcr:.2f} bearish (call-heavy)")
        else:
            signals.append(f"PCR={pcr:.2f} neutral")

        # Max pain
        max_pain = _calc_max_pain(chain)
        if max_pain and current_price:
            if current_price > max_pain * 1.01:
                score -= 1
                signals.append(f"above max_pain={max_pain:.0f}")
            elif current_price < max_pain * 0.99:
                score += 1
                signals.append(f"below max_pain={max_pain:.0f}")

        reasoning = "; ".join(signals) or "neutral options"
        if score >= 2:
            confidence = min(0.5 + score * 0.08, 0.90)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.LONG,
                confidence=round(confidence, 2),
                reasoning=reasoning,
            )
        elif score <= -2:
            confidence = min(0.5 + abs(score) * 0.08, 0.90)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.SHORT,
                confidence=round(confidence, 2),
                reasoning=reasoning,
            )
        return AgentVote(
            agent_name=self.name,
            direction=Direction.HOLD,
            confidence=0.0,
            reasoning=f"score={score}: {reasoning}",
        )
