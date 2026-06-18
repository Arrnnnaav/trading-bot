"""ChromaDB collection setup for news intelligence.

Run:
    python -m scripts.setup_chroma
or import and call get_or_create_collection(chroma_path) from other scripts.

Collection: news_articles
Distance space: hnsw:space=cosine
  → 1.0 - distance == cosine similarity (used by NewsAnalogueAgent._query_analogues)

Document metadata fields:
    id, headline, body_snippet, date, source, event_id, tags
"""

from pathlib import Path


def get_or_create_collection(chroma_path: str = "data/chroma"):
    """Return the news_articles ChromaDB collection, creating it if absent.

    Must be called before backfill_news.py or fetch_news_live.py.
    """
    import chromadb

    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name="news_articles",
        metadata={"hnsw:space": "cosine"},
    )
    print(
        f"[setup_chroma] Collection 'news_articles' ready at {chroma_path} "
        f"(count={collection.count()})"
    )
    return collection


if __name__ == "__main__":
    get_or_create_collection()
