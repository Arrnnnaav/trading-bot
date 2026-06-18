"""One-time SQLite setup for news intelligence.

Run:
    python -m scripts.setup_news_db
or import and call setup_db(db_path) from other scripts.
"""

import sqlite3
from pathlib import Path


def setup_db(db_path: str = "data/news.db") -> None:
    """Create data/news.db with market_events and news_articles_meta tables.

    Idempotent: safe to run multiple times (uses CREATE TABLE IF NOT EXISTS).
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_events (
                event_id            TEXT PRIMARY KEY,
                name                TEXT NOT NULL,
                category            TEXT NOT NULL,
                start_date          DATE NOT NULL,
                end_date            DATE NOT NULL,
                nifty_pct_change_5d REAL NOT NULL,
                recovery_days       INTEGER,
                key_characteristics JSON
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_articles_meta (
                article_id   TEXT PRIMARY KEY,
                headline     TEXT NOT NULL,
                body_snippet TEXT,
                date         DATE NOT NULL,
                source       TEXT NOT NULL,
                url          TEXT,
                event_id     TEXT REFERENCES market_events(event_id)
            )
        """)
        conn.commit()
    finally:
        conn.close()
    print(f"[setup_news_db] DB ready at {db_path}")


if __name__ == "__main__":
    setup_db()
