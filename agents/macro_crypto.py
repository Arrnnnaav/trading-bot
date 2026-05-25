"""
MacroCryptoAgent — Fear & Greed index + BTC dominance macro context.

Data sources (both free, no API key):
  - Fear & Greed: https://api.alternative.me/fng/
  - BTC dominance: https://api.coingecko.com/api/v3/global

Signal logic:
  Fear/Greed 0-24  (extreme fear)  -> HOLD  (ambiguous: bottom or continued drop)
  Fear/Greed 25-44 (fear)          -> SHORT
  Fear/Greed 45-55 (neutral)       -> HOLD
  Fear/Greed 56-74 (greed)         -> LONG
  Fear/Greed 75-100 (extreme greed)-> SHORT (contrarian, overbought)

BTC dominance modifier:
  dom > 58% AND ticker != BTC -> LONG confidence penalised 30%
  (capital rotating INTO BTC = altcoins bearish)

Cache: macro data fetched once per 15-min cycle (TTL=900s) to respect
CoinGecko free-tier rate limits when harness runs 5 tickers per cycle.
"""

import time
import requests

from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

_FNG_URL = "https://api.alternative.me/fng/?limit=1"
_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
_TIMEOUT = 6
_CACHE_TTL = 900  # seconds — matches 15-min harness cycle


class MacroCryptoAgent(BaseAgent):
    name = "MacroCrypto"

    def __init__(self):
        self._cached_fg: int | None = None
        self._cached_dom: float | None = None
        self._cache_ts: float = 0.0

    def analyze(self, ticker: str, klines: list, market: Market, **kwargs) -> AgentVote:
        try:
            fg, btc_dom = self._fetch_macro()
        except Exception as exc:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"Macro data unavailable: {exc}",
            )

        direction, confidence, fg_note = self._fg_signal(fg)

        dom_note = f"BTC dom={btc_dom:.1f}%"
        if btc_dom > 58 and "BTC" not in ticker and direction == Direction.LONG:
            confidence *= 0.70
            dom_note += " >58% — alt penalised"

        reasoning = f"Macro: {fg_note} | {dom_note}"

        if direction == Direction.HOLD or confidence < 0.50:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=reasoning,
            )

        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=round(min(confidence, 0.75), 3),
            reasoning=reasoning,
        )

    def _fetch_macro(self) -> tuple[int, float]:
        now = time.time()
        if now - self._cache_ts < _CACHE_TTL and self._cached_fg is not None:
            return self._cached_fg, self._cached_dom

        r = requests.get(_FNG_URL, timeout=_TIMEOUT)
        r.raise_for_status()
        fg = int(r.json()["data"][0]["value"])

        r2 = requests.get(_GLOBAL_URL, timeout=_TIMEOUT)
        r2.raise_for_status()
        btc_dom = float(r2.json()["data"]["market_cap_percentage"].get("btc", 50.0))

        self._cached_fg = fg
        self._cached_dom = btc_dom
        self._cache_ts = now
        return fg, btc_dom

    @staticmethod
    def _fg_signal(fg: int) -> tuple[Direction, float, str]:
        if fg <= 24:
            return Direction.HOLD, 0.0, f"Extreme Fear ({fg}) — ambiguous HOLD"
        if fg <= 44:
            conf = 0.55 + (44 - fg) / 100
            return Direction.SHORT, conf, f"Fear ({fg}) -> SHORT"
        if fg <= 55:
            return Direction.HOLD, 0.0, f"Neutral ({fg})"
        if fg <= 74:
            conf = 0.55 + (fg - 55) / 100
            return Direction.LONG, conf, f"Greed ({fg}) -> LONG"
        return Direction.SHORT, 0.62, f"Extreme Greed ({fg}) — overbought SHORT"
