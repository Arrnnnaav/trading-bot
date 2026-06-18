"""MacroIndiaAgent — India VIX, SGX Nifty gap, USD/INR, RBI policy day."""

import logging
import time
from typing import Optional

from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

_LOG = logging.getLogger(__name__)

# VIX cache: (value, fetched_at_timestamp)
_vix_cache: tuple[float, float] | None = None
_VIX_TTL = 3600  # 1 hour


def _fetch_vix() -> Optional[float]:
    global _vix_cache
    now = time.time()
    if _vix_cache and now - _vix_cache[1] < _VIX_TTL:
        return _vix_cache[0]
    try:
        import yfinance as yf

        ticker = yf.Ticker("^INDIAVIX")
        hist = ticker.history(period="1d", interval="1h")
        if hist.empty:
            return None
        val = float(hist["Close"].iloc[-1])
        _vix_cache = (val, now)
        return val
    except Exception as exc:
        _LOG.warning("VIX fetch failed: %s", exc)
        return None


class MacroIndiaAgent(BaseAgent):
    name = "macro_india"

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        try:
            vix: Optional[float] = kwargs.get("vix")
            if vix is None:
                vix = _fetch_vix()
            sgx_gap: float = kwargs.get("sgx_gap_pct", 0.0)
            usdinr_chg: float = kwargs.get("usdinr_change_pct", 0.0)
            is_rbi_day: bool = kwargs.get("is_rbi_day", False)
            return self._compute(vix, sgx_gap, usdinr_chg, is_rbi_day)
        except Exception as exc:
            _LOG.warning("MacroIndiaAgent failed: %s", exc)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"error: {exc}",
            )

    def _compute(
        self,
        vix: Optional[float],
        sgx_gap: float,
        usdinr_chg: float,
        is_rbi_day: bool,
    ) -> AgentVote:
        if is_rbi_day:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="RBI policy day — widening stops",
            )

        score = 0
        signals: list[str] = []

        if sgx_gap > 1.0:
            score += 1
            signals.append(f"SGX gap +{sgx_gap:.1f}%")
        elif sgx_gap < -1.0:
            score -= 1
            signals.append(f"SGX gap {sgx_gap:.1f}%")

        if usdinr_chg > 0.5:
            score -= 1
            signals.append(f"USD/INR +{usdinr_chg:.2f}% (FII outflow)")

        vix_penalty = 0.0
        vix_note = ""
        if vix is not None:
            if vix > 20:
                vix_penalty = 0.10
                vix_note = f" (VIX={vix:.1f} penalty)"
            if vix > 25:
                return AgentVote(
                    agent_name=self.name,
                    direction=Direction.HOLD,
                    confidence=0.0,
                    reasoning=f"VIX={vix:.1f} >25 — blocked",
                )

        reasoning = ("; ".join(signals) or "neutral macro") + vix_note
        base_confidence = 0.65

        if score >= 1:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.LONG,
                confidence=round(max(0.0, base_confidence - vix_penalty), 2),
                reasoning=reasoning,
            )
        elif score <= -1:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.SHORT,
                confidence=round(max(0.0, base_confidence - vix_penalty), 2),
                reasoning=reasoning,
            )
        return AgentVote(
            agent_name=self.name,
            direction=Direction.HOLD,
            confidence=0.0,
            reasoning=reasoning,
        )
