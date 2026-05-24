import json
import httpx
from pathlib import Path
from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market


class NewsSentimentAgent(BaseAgent):
    name = "NewsSentiment"
    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, api_key: str):
        self.api_key = api_key
        thresholds = self._load_thresholds()
        self._sentiment_threshold = thresholds.get("news_sentiment_threshold", 0.2)

    def _load_thresholds(self) -> dict:
        try:
            return json.loads(Path("data/calibrated_thresholds.json").read_text())
        except (FileNotFoundError, Exception):
            return {}

    def _fetch_news(self, ticker: str) -> list[dict]:
        currency = ticker.split("/")[0]
        resp = httpx.get(
            self.BASE_URL,
            params={
                "auth_token": self.api_key,
                "currencies": currency,
                "filter": "hot",
                "public": "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    def _score_posts(self, posts: list[dict]) -> float:
        if not posts:
            return 0.0
        score = 0.0
        for post in posts[:10]:
            votes = post.get("votes", {})
            bullish = votes.get("liked", 0)
            bearish = votes.get("disliked", 0)
            total = bullish + bearish
            if total > 0:
                score += (bullish - bearish) / total
        return score / min(len(posts), 10)

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        try:
            posts = self._fetch_news(ticker)
            score = self._score_posts(posts)
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

        headlines = [p.get("title", "")[:60] for p in posts[:3]]
        reasoning = f"Sentiment score: {score:.2f}. Top: {' | '.join(headlines)}"
        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
        )
