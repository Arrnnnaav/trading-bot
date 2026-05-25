import json
import httpx
from pathlib import Path
from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

THRESHOLDS_PATH = "data/calibrated_thresholds.json"


class OnChainAgent(BaseAgent):
    name = "OnChain"
    BASE_URL = "https://open-api.coinglass.com/public/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key
        thresholds = self._load_thresholds()
        self._funding_threshold = thresholds.get("onchain_funding_threshold", -0.001)
        self._ls_ratio_threshold = thresholds.get("onchain_ls_ratio_threshold", 0.9)

    def _load_thresholds(self) -> dict:
        try:
            return json.loads(Path(THRESHOLDS_PATH).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _fetch_funding(self, symbol: str) -> float:
        resp = httpx.get(
            f"{self.BASE_URL}/funding",
            params={"symbol": symbol},
            headers={"coinglassSecret": self.api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return 0.0
        rates = [float(x.get("fundingRate", 0)) for x in data[:5]]
        return sum(rates) / len(rates)

    def _fetch_ls_ratio(self, symbol: str) -> float:
        resp = httpx.get(
            f"{self.BASE_URL}/globalLongShortAccountRatio",
            params={"symbol": symbol, "interval": "1h", "limit": 1},
            headers={"coinglassSecret": self.api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return 1.0
        return float(data[-1].get("longAccount", 0.5)) / max(
            float(data[-1].get("shortAccount", 0.5)), 0.01
        )

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        symbol = ticker.split("/")[0]
        try:
            funding = self._fetch_funding(symbol)
            ls_ratio = self._fetch_ls_ratio(symbol)
        except Exception:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="On-chain fetch failed",
            )

        # Negative funding = longs being paid = bullish
        # LS ratio < 0.9 = more shorts = squeeze potential = bullish
        funding_bullish = funding < self._funding_threshold
        ls_bullish = ls_ratio < self._ls_ratio_threshold

        if funding_bullish and ls_bullish:
            direction, confidence = Direction.LONG, 0.75
        elif not funding_bullish and not ls_bullish:
            direction, confidence = Direction.SHORT, 0.65
        else:
            direction, confidence = Direction.HOLD, 0.0

        reasoning = f"Funding: {funding:.4f} | L/S ratio: {ls_ratio:.2f}"
        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
        )
