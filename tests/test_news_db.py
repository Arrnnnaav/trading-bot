"""Tests for scripts/setup_news_db.py"""

import sqlite3
import tempfile
from pathlib import Path


from scripts.setup_news_db import setup_db


def test_creates_market_events_table():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test_news.db")
        setup_db(db_path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='market_events'"
            ).fetchall()
            assert len(rows) == 1, "market_events table must exist"
        finally:
            conn.close()


def test_market_events_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test_news.db")
        setup_db(db_path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(market_events)")}
            required = {
                "event_id",
                "name",
                "category",
                "start_date",
                "end_date",
                "nifty_pct_change_5d",
                "recovery_days",
                "key_characteristics",
            }
            assert required.issubset(cols), f"Missing columns: {required - cols}"
        finally:
            conn.close()


def test_creates_news_articles_meta_table():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test_news.db")
        setup_db(db_path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='news_articles_meta'"
            ).fetchall()
            assert len(rows) == 1
        finally:
            conn.close()


def test_news_articles_meta_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test_news.db")
        setup_db(db_path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            cols = {
                row[1] for row in conn.execute("PRAGMA table_info(news_articles_meta)")
            }
            expected = {
                "id",
                "headline",
                "body_snippet",
                "date",
                "source",
                "url",
                "event_id",
                "tags",
            }
            assert cols == expected, f"Column mismatch. Expected {expected}, got {cols}"
        finally:
            conn.close()


def test_setup_db_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test_news.db")
        setup_db(db_path)
        # Verify schema before second run
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            cols_before = {
                row[1] for row in conn.execute("PRAGMA table_info(news_articles_meta)")
            }
        finally:
            conn.close()

        # Run setup again
        setup_db(db_path)

        # Verify schema is still intact
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            cols_after = {
                row[1] for row in conn.execute("PRAGMA table_info(news_articles_meta)")
            }
            assert cols_before == cols_after, "Schema changed after second run"
            expected = {
                "id",
                "headline",
                "body_snippet",
                "date",
                "source",
                "url",
                "event_id",
                "tags",
            }
            assert cols_after == expected, (
                f"Column mismatch. Expected {expected}, got {cols_after}"
            )
        finally:
            conn.close()
