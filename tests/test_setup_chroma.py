"""Tests for scripts/setup_chroma.py"""


import pytest

chromadb = pytest.importorskip("chromadb")

from scripts.setup_chroma import get_or_create_collection


def test_collection_exists(tmp_path):
    col = get_or_create_collection(str(tmp_path))
    assert col is not None
    assert col.name == "news_articles"


def test_distance_metric_is_cosine(tmp_path):
    col = get_or_create_collection(str(tmp_path))
    meta = col.metadata or {}
    assert meta.get("hnsw:space") == "cosine", f"Expected hnsw:space=cosine, got {meta}"


def test_idempotent_create_twice(tmp_path):
    col1 = get_or_create_collection(str(tmp_path))
    col2 = get_or_create_collection(str(tmp_path))
    assert col1.name == col2.name == "news_articles"


def test_agent_can_get_collection(tmp_path):
    """Simulate the exact call NewsAnalogueAgent._get_collection() makes."""
    import chromadb as _chromadb

    get_or_create_collection(str(tmp_path))  # must create first
    client = _chromadb.PersistentClient(path=str(tmp_path))
    col = client.get_collection("news_articles")  # must not raise
    assert col is not None
