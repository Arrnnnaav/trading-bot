import json
import httpx
from pathlib import Path
from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

_SENTIMENT_MAP = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}


class NewsSentimentAgent(BaseAgent):
    name = "NewsSentiment"
    BASE_URL = "https://data-api.cryptocompare.com/news/v1/article/list"

    def __init__(self):
        thresholds = self._load_thresholds()
        self._sentiment_threshold = thresholds.get("news_sentiment_threshold", 0.2)

    def _load_thresholds(self) -> dict:
        try:
            return json.loads(Path("data/calibrated_thresholds.json").read_text())
        except Exception:
            return {}

    def _fetch_news(self, ticker: str) -> list[dict]:
        currency = ticker.split("/")[0].upper()
        resp = httpx.get(
            self.BASE_URL,
            params={"limit": 20, "lang": "EN", "categories": currency},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("Data", [])

    def _score_articles(self, articles: list[dict]) -> float:
        if not articles:
            return 0.0
        scores = [
            _SENTIMENT_MAP.get(a.get("SENTIMENT", "NEUTRAL"), 0.0)
            for a in articles[:10]
        ]
        return sum(scores) / len(scores)

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        try:
            articles = self._fetch_news(ticker)
            score = self._score_articles(articles)
        except Exception:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="News fetch failed",
            )

        if score > self._sentiment_threshold:
            direction, confidence = Direction.LONG, min(0.5 + score, 0.85)
        elif score < -self._sentiment_threshold:
            direction, confidence = Direction.SHORT, min(0.5 + abs(score), 0.85)
        else:
            direction, confidence = Direction.HOLD, 0.0

        headlines = [a.get("TITLE", "")[:60] for a in articles[:3]]
        reasoning = f"Sentiment score: {score:.2f}. Top: {' | '.join(headlines)}"
        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
        )
