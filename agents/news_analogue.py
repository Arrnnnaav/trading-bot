"""NewsAnalogueAgent — ChromaDB semantic analogue matching for regime votes."""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import httpx

from agents.base import BaseAgent
from config import config
from core.models import AgentVote, Direction, Market

_LOG = logging.getLogger(__name__)
_SIMILARITY_THRESHOLD = 0.75
_RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
]


class NewsAnalogueAgent(BaseAgent):
    name = "news_analogue"

    def __init__(
        self,
        chroma_path: str = "data/chroma",
        news_db_path: str = "data/news.db",
    ):
        self._chroma_path = chroma_path
        self._news_db_path = news_db_path
        self._collection = None
        self._embedder = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self._chroma_path)
            self._collection = client.get_collection("news_articles")
            # Collection must be created with hnsw:space=cosine (see scripts/backfill_news.py)
            # so that distances map to [0, 1] and 1.0 - distance == cosine similarity
            return self._collection
        except Exception:
            return None

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            return self._embedder
        except Exception:
            return None

    def _fetch_headlines(self) -> list[str]:
        headlines: list[str] = []
        # Marketaux
        if config.marketaux_api_key:
            try:
                resp = httpx.get(
                    "https://api.marketaux.com/v1/news/all",
                    params={
                        "symbols": "NSEI,NSEBANK",
                        "filter_entities": "true",
                        "language": "en",
                        "api_token": config.marketaux_api_key,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                for article in resp.json().get("data", [])[:5]:
                    headlines.append(article.get("title", ""))
            except Exception as exc:
                _LOG.debug("Marketaux fetch failed: %s", exc)
        # RSS fallback
        if len(headlines) < 3:
            try:
                import feedparser

                for url in _RSS_FEEDS:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:3]:
                        headlines.append(entry.get("title", ""))
                    if len(headlines) >= 5:
                        break
            except Exception as exc:
                _LOG.debug("RSS fetch failed: %s", exc)
        return [h for h in headlines if h][:5]

    def _query_analogues(self, headlines: list[str]) -> list[dict]:
        collection = self._get_collection()
        embedder = self._get_embedder()
        if collection is None or embedder is None or not headlines:
            return []
        try:
            query_text = " ".join(headlines)
            embedding = embedder.encode([query_text])[0].tolist()
            results = collection.query(
                query_embeddings=[embedding],
                n_results=3,
                include=["metadatas", "distances"],
            )
            analogues = []
            for metadata, distance in zip(
                results["metadatas"][0], results["distances"][0]
            ):
                similarity = 1.0 - distance  # cosine distance → similarity
                event_id = metadata.get("event_id")
                pct_change = self._lookup_outcome(event_id)
                analogues.append(
                    {
                        "similarity": similarity,
                        "nifty_pct_change_5d": pct_change or 0.0,
                        "event_id": event_id,
                    }
                )
            return analogues
        except Exception as exc:
            _LOG.warning("ChromaDB query failed: %s", exc)
            return []

    def _lookup_outcome(self, event_id: Optional[str]) -> Optional[float]:
        if not event_id:
            return None
        db_path = Path(self._news_db_path)
        if not db_path.exists():
            return None
        try:
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "SELECT nifty_pct_change_5d FROM market_events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        try:
            headlines = self._fetch_headlines()
            if not headlines:
                return AgentVote(
                    agent_name=self.name,
                    direction=Direction.HOLD,
                    confidence=0.0,
                    reasoning="no headlines fetched",
                )
            analogues = self._query_analogues(headlines)
            return self._vote(analogues)
        except Exception as exc:
            _LOG.warning("NewsAnalogueAgent failed: %s", exc)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"error: {exc}",
            )

    def _vote(self, analogues: list[dict]) -> AgentVote:
        filtered = [a for a in analogues if a["similarity"] >= _SIMILARITY_THRESHOLD]
        if not filtered:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="no analogues above similarity threshold",
            )
        avg_similarity = sum(a["similarity"] for a in filtered) / len(filtered)
        avg_outcome = sum(a["nifty_pct_change_5d"] for a in filtered) / len(filtered)
        confidence = round(min(avg_similarity, 0.85), 2)
        event_ref = filtered[0].get("event_id", "unknown")
        if avg_outcome > 0:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.LONG,
                confidence=confidence,
                reasoning=f"analogue={event_ref} avg_5d=+{avg_outcome:.1f}%",
            )
        elif avg_outcome < 0:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.SHORT,
                confidence=confidence,
                reasoning=f"analogue={event_ref} avg_5d={avg_outcome:.1f}%",
            )
        return AgentVote(
            agent_name=self.name,
            direction=Direction.HOLD,
            confidence=0.0,
            reasoning="neutral historical outcome",
        )
