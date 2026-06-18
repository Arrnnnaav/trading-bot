"""TechnicalIndiaAgent — RSI/MACD/BB/VWAP/EMA + fractal S/R zones."""

import logging

from agents.base import BaseAgent
from agents._indicators import rsi, macd, bollinger, vwap, ema, support_resistance
from core.models import AgentVote, Direction, Market

_LOG = logging.getLogger(__name__)
_MIN_KLINES = 30


class TechnicalIndiaAgent(BaseAgent):
    name = "technical_india"

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        if len(klines) < _MIN_KLINES:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"Insufficient data: {len(klines)} klines (need {_MIN_KLINES})",
            )
        try:
            return self._compute(ticker, klines)
        except Exception as exc:
            _LOG.warning("TechnicalIndiaAgent failed for %s: %s", ticker, exc)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"Computation error: {exc}",
            )

    def _compute(self, ticker: str, klines: list[dict]) -> AgentVote:
        closes = [k["close"] for k in klines]
        volumes = [k["volume"] for k in klines]
        avg_volume = (
            sum(volumes[-20:]) / 20
            if len(volumes) >= 20
            else (volumes[-1] if volumes else 1)
        )
        current = closes[-1]
        current_vol = volumes[-1] if volumes else 0

        score = 0
        signals: list[str] = []

        # RSI — informational filter; extreme oversold/overbought noted in reasoning
        # Not scored directly: trend-following signals (MACD/EMA/VWAP) handle direction;
        # RSI extremes in strong trends are momentum, not reversal.
        rsi_val = rsi(closes, 14)
        if rsi_val < 30:
            signals.append(f"RSI={rsi_val:.1f} oversold")
        elif rsi_val > 70:
            signals.append(f"RSI={rsi_val:.1f} overbought")

        # MACD
        macd_val, signal_val = macd(closes)
        if macd_val > signal_val:
            score += 1
            signals.append("MACD bullish")
        elif macd_val < signal_val:
            score -= 1
            signals.append("MACD bearish")

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = bollinger(closes)
        if current < bb_lower:
            score += 1
            signals.append("price below BB lower")
        elif current > bb_upper:
            score -= 1
            signals.append("price above BB upper")

        # VWAP
        vwap_val = vwap(klines)
        if current > vwap_val:
            score += 1
            signals.append(f"price above VWAP {vwap_val:.0f}")
        else:
            score -= 1
            signals.append(f"price below VWAP {vwap_val:.0f}")

        # EMA cross
        ema9 = ema(closes, 9)
        ema21 = ema(closes, 21)
        if ema9 and ema21:
            if ema9[-1] > ema21[-1]:
                score += 1
                signals.append("EMA9 > EMA21")
            else:
                score -= 1
                signals.append("EMA9 < EMA21")

        # S/R breakout/breakdown
        supports, resistances = support_resistance(klines)
        vol_surge = current_vol > avg_volume * 1.2
        if resistances and current > resistances[0] and vol_surge:
            score += 1
            signals.append(f"breakout above R={resistances[0]:.0f}")
        if supports and current < supports[0] and vol_surge:
            score -= 1
            signals.append(f"breakdown below S={supports[0]:.0f}")

        reasoning = "; ".join(signals) or "no clear signal"
        if score >= 3:
            confidence = min(0.5 + score * 0.08, 0.95)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.LONG,
                confidence=round(confidence, 2),
                reasoning=reasoning,
            )
        elif score <= -3:
            confidence = min(0.5 + abs(score) * 0.08, 0.95)
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
