"""Tests for scripts/fetch_news_live.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from scripts.fetch_news_live import (
    fetch_headlines,
    fetch_and_ingest,
    _article_id,
    _save_jsonl,
)


_MOCK_MARKETAUX_RESPONSE = {
    "data": [
        {
            "title": "Nifty rises 200 points on strong FII buying",
            "description": "Indian markets rallied sharply as FIIs bought heavily",
            "url": "https://example.com/article1",
        },
        {
            "title": "RBI holds rates steady at MPC meeting",
            "description": "Reserve Bank keeps repo rate unchanged at 6.5%",
            "url": "https://example.com/article2",
        },
        {
            "title": "Bank Nifty surges 500 points on banking sector optimism",
            "description": "Banking stocks lead the rally as credit growth picks up",
            "url": "https://example.com/article3",
        },
    ]
}

_MOCK_RSS_FEED = MagicMock()
_MOCK_RSS_FEED.entries = [
    MagicMock(
        title="Sensex gains 400 points", summary="Markets up", link="http://et.com/1"
    ),
    MagicMock(
        title="Midcap stocks outperform", summary="Midcap rally", link="http://et.com/2"
    ),
    MagicMock(
        title="India VIX drops to 12 — low fear",
        summary="Calm markets",
        link="http://et.com/3",
    ),
]


def test_fetch_headlines_marketaux_primary():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _MOCK_MARKETAUX_RESPONSE
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        results = fetch_headlines(api_key="test_key", tickers=["NIFTY"])

    assert len(results) == 3
    assert results[0]["headline"] == "Nifty rises 200 points on strong FII buying"
    assert results[0]["source"] == "marketaux"
    assert results[0]["date"] is not None


def test_fetch_headlines_rss_fallback_when_no_api_key():
    mock_feedparser = MagicMock()
    mock_feedparser.parse.return_value = _MOCK_RSS_FEED

    with patch.dict("sys.modules", {"feedparser": mock_feedparser}):
        results = fetch_headlines(api_key="", tickers=["NIFTY"])

    assert len(results) > 0
    for r in results:
        assert r["source"] == "rss"


def test_save_jsonl_creates_file():
    articles = [
        {
            "headline": "Test headline",
            "body_snippet": "snippet",
            "date": "2026-06-18",
            "source": "test",
            "url": "http://x.com",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        _save_jsonl(articles, tmp)
        files = list(Path(tmp).glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["headline"] == "Test headline"
        assert obj["source"] == "test"


def test_fetch_and_ingest_writes_jsonl_and_upserts_chroma():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _MOCK_MARKETAUX_RESPONSE
    mock_resp.raise_for_status = MagicMock()

    # ignore_cleanup_errors=True: ChromaDB holds file locks on Windows; the test
    # assertions complete before cleanup, so a PermissionError on teardown is safe.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chroma_path = str(Path(tmp) / "chroma")
        news_path = str(Path(tmp) / "news")

        with patch("httpx.get", return_value=mock_resp):
            result = fetch_and_ingest(
                api_key="test_key",
                chroma_path=chroma_path,
                output_dir=news_path,
                tickers=["NIFTY"],
            )

        assert result["articles_fetched"] == 3
        assert result["articles_saved"] == 3
        assert result["chroma_upserted"] == 3

        # Verify JSONL file
        jsonl_files = list(Path(news_path).glob("*.jsonl"))
        assert len(jsonl_files) == 1

        # Verify ChromaDB
        import chromadb as _chromadb

        client = _chromadb.PersistentClient(path=chroma_path)
        col = client.get_collection("news_articles")
        assert col.count() == 3


def test_fetch_and_ingest_returns_zeros_on_empty():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status = MagicMock()

    with tempfile.TemporaryDirectory() as tmp:
        with patch("httpx.get", return_value=mock_resp):
            result = fetch_and_ingest(
                api_key="test_key",
                chroma_path=str(Path(tmp) / "chroma"),
                output_dir=str(Path(tmp) / "news"),
            )
        assert result["articles_fetched"] == 0
        assert result["chroma_upserted"] == 0


def test_article_id_is_deterministic():
    a = {"headline": "Nifty up 200 points", "date": "2026-06-18", "source": "test"}
    id1 = _article_id(a)
    id2 = _article_id(a)
    assert id1 == id2
    assert len(id1) == 16
