"""Tests for scripts/backfill_news.py"""

import sqlite3

import pytest

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from scripts.backfill_news import backfill, MARKET_EVENTS, HEADLINES_BY_EVENT


def test_event_count_in_sqlite(tmp_path):
    db = str(tmp_path / "news.db")
    backfill(db_path=db, chroma_path=str(tmp_path / "chroma"))
    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM market_events").fetchone()[0]
    assert count == 15


def test_chroma_doc_count_at_least_45(tmp_path):
    db = str(tmp_path / "news.db")
    chroma = str(tmp_path / "chroma")
    backfill(db_path=db, chroma_path=chroma)
    import chromadb as _chromadb

    client = _chromadb.PersistentClient(path=chroma)
    col = client.get_collection("news_articles")
    assert col.count() >= 45


def test_nifty_pct_change_5d_column_name(tmp_path):
    """NewsAnalogueAgent queries this exact column name — must exist."""
    db = str(tmp_path / "news.db")
    backfill(db_path=db, chroma_path=str(tmp_path / "chroma"))
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT nifty_pct_change_5d FROM market_events WHERE event_id = ?",
            ("covid_crash_2020",),
        ).fetchone()
    assert row is not None
    assert row[0] == -25.0


def test_idempotent_second_run(tmp_path):
    db = str(tmp_path / "news.db")
    chroma = str(tmp_path / "chroma")
    backfill(db_path=db, chroma_path=chroma)
    backfill(db_path=db, chroma_path=chroma)  # must not raise or duplicate
    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM market_events").fetchone()[0]
    assert count == 15


def test_all_events_have_headlines():
    for ev in MARKET_EVENTS:
        headlines = HEADLINES_BY_EVENT.get(ev["event_id"], [])
        assert len(headlines) >= 3, (
            f"Event {ev['event_id']} needs >=3 headlines, has {len(headlines)}"
        )


def test_chroma_metadata_fields(tmp_path):
    """All docs must have the required metadata fields NewsAnalogueAgent expects."""
    db = str(tmp_path / "news.db")
    chroma = str(tmp_path / "chroma")
    backfill(db_path=db, chroma_path=chroma)
    import chromadb as _chromadb

    client = _chromadb.PersistentClient(path=chroma)
    col = client.get_collection("news_articles")
    results = col.get(limit=5, include=["metadatas"])
    required_fields = {
        "id",
        "headline",
        "body_snippet",
        "date",
        "source",
        "event_id",
        "tags",
    }
    for meta in results["metadatas"]:
        missing = required_fields - set(meta.keys())
        assert not missing, f"Metadata missing fields: {missing}"
