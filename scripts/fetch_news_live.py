"""Live news ingestion — Marketaux + RSS fallback.

Sources:
  1. Marketaux API (primary) — 100 req/day free tier
  2. feedparser RSS (fallback) — Economic Times, Moneycontrol, Mint

Output:
  - data/news/YYYY-MM-DD.jsonl (append-only, one JSON object per line)
  - ChromaDB news_articles collection (upsert)

Format per line:
    {"headline": "...", "body_snippet": "...", "date": "2026-06-18",
     "source": "marketaux", "url": "..."}

Run:
    python -m scripts.fetch_news_live
or call fetch_and_ingest(api_key, chroma_path, output_dir, tickers) from other modules.

Scheduled: every 30min during market hours 9:00–15:30 IST (see main.py Task 6).
"""

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import httpx

_LOG = logging.getLogger(__name__)

_RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.livemint.com/rss/markets",
]


# ---------------------------------------------------------------------------
# Headline fetching
# ---------------------------------------------------------------------------


def fetch_headlines(
    api_key: str,
    tickers: list[str] | None = None,
) -> list[dict]:
    """Fetch up to 10 recent market headlines.

    Returns list of dicts with keys: headline, body_snippet, date, source, url.
    Tries Marketaux first; falls back to RSS feeds if fewer than 3 articles returned.
    """
    if tickers is None:
        tickers = ["NIFTY", "SENSEX"]

    articles: list[dict] = []
    today = date.today().strftime("%Y-%m-%d")

    # --- Marketaux ---
    if api_key:
        try:
            resp = httpx.get(
                "https://api.marketaux.com/v1/news/all",
                params={
                    "symbols": ",".join(tickers),
                    "filter_entities": "true",
                    "language": "en",
                    "api_token": api_key,
                    "limit": 10,
                },
                timeout=10,
            )
            resp.raise_for_status()
            for item in resp.json().get("data", [])[:10]:
                articles.append(
                    {
                        "headline": item.get("title", "").strip(),
                        "body_snippet": (
                            item.get("description") or item.get("title", "")
                        )[:300],
                        "date": today,
                        "source": "marketaux",
                        "url": item.get("url", ""),
                    }
                )
        except Exception as exc:
            _LOG.warning("Marketaux fetch failed: %s", exc)

    # --- RSS fallback ---
    if len(articles) < 3:
        try:
            import feedparser

            for feed_url in _RSS_FEEDS:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:
                        title = entry.get("title", "").strip()
                        if title:
                            articles.append(
                                {
                                    "headline": title,
                                    "body_snippet": entry.get("summary", title)[:300],
                                    "date": today,
                                    "source": "rss",
                                    "url": entry.get("link", ""),
                                }
                            )
                    if len(articles) >= 10:
                        break
                except Exception as feed_exc:
                    _LOG.debug("RSS feed %s failed: %s", feed_url, feed_exc)
        except ImportError:
            _LOG.warning("feedparser not installed — RSS fallback unavailable")

    return [a for a in articles if a["headline"]][:10]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _article_id(article: dict) -> str:
    """Deterministic ID based on headline + date to avoid duplicates."""
    raw = f"{article['date']}_{article['headline']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _save_jsonl(articles: list[dict], output_dir: str) -> Path:
    """Append articles to data/news/YYYY-MM-DD.jsonl (one JSON per line)."""
    today = date.today().strftime("%Y-%m-%d")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today}.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
    return out_path


def _upsert_to_chroma(articles: list[dict], chroma_path: str) -> int:
    """Embed articles and upsert into ChromaDB news_articles collection."""
    if not articles:
        return 0
    try:
        from sentence_transformers import SentenceTransformer
        from scripts.setup_chroma import get_or_create_collection

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        collection = get_or_create_collection(chroma_path)

        ids, embeddings, metadatas = [], [], []
        for article in articles:
            doc_id = _article_id(article)
            text = article["headline"] + " " + article.get("body_snippet", "")
            embedding = embedder.encode([text])[0].tolist()
            ids.append(doc_id)
            embeddings.append(embedding)
            metadatas.append(
                {
                    "id": doc_id,
                    "headline": article["headline"],
                    "body_snippet": article.get("body_snippet", ""),
                    "date": article["date"],
                    "source": article["source"],
                    "event_id": "",  # live articles not yet linked to historical events
                    "tags": "live",
                }
            )

        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        return len(ids)
    except Exception as exc:
        _LOG.warning("ChromaDB upsert failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fetch_and_ingest(
    api_key: str,
    chroma_path: str = "data/chroma",
    output_dir: str = "data/news",
    tickers: list[str] | None = None,
) -> dict:
    """Fetch headlines, save to JSONL, upsert to ChromaDB.

    Returns dict with keys: articles_fetched, articles_saved, chroma_upserted.
    Never raises — all errors are logged and counted.
    """
    articles = fetch_headlines(api_key=api_key, tickers=tickers)
    if not articles:
        _LOG.warning("No articles fetched — skipping save and embed")
        return {"articles_fetched": 0, "articles_saved": 0, "chroma_upserted": 0}

    saved_path = _save_jsonl(articles, output_dir)
    _LOG.info("Saved %d articles to %s", len(articles), saved_path)

    n_upserted = _upsert_to_chroma(articles, chroma_path)
    _LOG.info("Upserted %d articles to ChromaDB", n_upserted)

    return {
        "articles_fetched": len(articles),
        "articles_saved": len(articles),
        "chroma_upserted": n_upserted,
    }


if __name__ == "__main__":
    import sys
    from config import config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = fetch_and_ingest(
        api_key=config.marketaux_api_key,
        chroma_path=config.chroma_db_path,
        output_dir="data/news",
    )
    print(f"Done: {result}")
    sys.exit(0 if result["articles_fetched"] > 0 else 1)
