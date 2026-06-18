"""One-time historical news backfill.

Populates:
  - data/news.db market_events  (15 seed events)
  - data/chroma news_articles   (3-5 headlines per event, embedded)

Idempotent: skips events already in DB; skips ChromaDB docs already upserted.

Run:
    python -m scripts.backfill_news
"""

import json
import logging
import sqlite3

from scripts.setup_news_db import setup_db
from scripts.setup_chroma import get_or_create_collection

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

MARKET_EVENTS = [
    {
        "event_id": "dotcom_rise_1999",
        "name": "Dot-com bubble rise",
        "category": "bubble",
        "start_date": "1999-01-01",
        "end_date": "2000-03-10",
        "nifty_pct_change_5d": 8.5,
        "recovery_days": None,
        "key_characteristics": [
            "tech euphoria",
            "high PE",
            "retail frenzy",
            "IPO boom",
        ],
    },
    {
        "event_id": "dotcom_crash_2000",
        "name": "Dot-com crash",
        "category": "crash",
        "start_date": "2000-03-11",
        "end_date": "2002-10-09",
        "nifty_pct_change_5d": -15.2,
        "recovery_days": 730,
        "key_characteristics": [
            "tech selloff",
            "margin calls",
            "corporate fraud",
            "recession fears",
        ],
    },
    {
        "event_id": "sept11_2001",
        "name": "9/11 terrorist attacks",
        "category": "geopolitical",
        "start_date": "2001-09-11",
        "end_date": "2001-09-21",
        "nifty_pct_change_5d": -9.3,
        "recovery_days": 30,
        "key_characteristics": [
            "terror attack",
            "market closure",
            "panic selling",
            "US crisis",
        ],
    },
    {
        "event_id": "gfc_crash_2008",
        "name": "Global Financial Crisis — crash phase",
        "category": "crash",
        "start_date": "2008-09-15",
        "end_date": "2009-03-09",
        "nifty_pct_change_5d": -20.1,
        "recovery_days": 540,
        "key_characteristics": [
            "Lehman collapse",
            "credit freeze",
            "bank bailouts",
            "global recession",
        ],
    },
    {
        "event_id": "gfc_recovery_2009",
        "name": "Global Financial Crisis — recovery",
        "category": "rally",
        "start_date": "2009-03-10",
        "end_date": "2009-12-31",
        "nifty_pct_change_5d": 5.2,
        "recovery_days": None,
        "key_characteristics": [
            "Fed stimulus",
            "quantitative easing",
            "V-recovery",
            "risk-on",
        ],
    },
    {
        "event_id": "india_demonetization_2016",
        "name": "India demonetization",
        "category": "policy",
        "start_date": "2016-11-08",
        "end_date": "2016-12-31",
        "nifty_pct_change_5d": -7.8,
        "recovery_days": 60,
        "key_characteristics": [
            "Rs 500/1000 ban",
            "cash crunch",
            "GDP shock",
            "RBI policy",
        ],
    },
    {
        "event_id": "covid_crash_2020",
        "name": "COVID-19 crash",
        "category": "crash",
        "start_date": "2020-02-20",
        "end_date": "2020-03-23",
        "nifty_pct_change_5d": -25.0,
        "recovery_days": 180,
        "key_characteristics": [
            "pandemic",
            "lockdown",
            "supply chain collapse",
            "demand destruction",
        ],
    },
    {
        "event_id": "covid_recovery_2020",
        "name": "COVID-19 V-recovery",
        "category": "rally",
        "start_date": "2020-03-24",
        "end_date": "2021-02-15",
        "nifty_pct_change_5d": 12.0,
        "recovery_days": None,
        "key_characteristics": [
            "vaccine optimism",
            "fiscal stimulus",
            "FOMO rally",
            "FII buying",
        ],
    },
    {
        "event_id": "rate_hike_cycle_2022",
        "name": "Global rate hike cycle start 2022",
        "category": "policy",
        "start_date": "2022-03-16",
        "end_date": "2022-12-31",
        "nifty_pct_change_5d": -5.5,
        "recovery_days": 300,
        "key_characteristics": [
            "Fed hawkish",
            "inflation surge",
            "bond sell-off",
            "FII outflows",
        ],
    },
    {
        "event_id": "india_budget_positive",
        "name": "India Union Budget — positive surprise",
        "category": "policy",
        "start_date": "2021-02-01",
        "end_date": "2021-02-05",
        "nifty_pct_change_5d": 4.2,
        "recovery_days": None,
        "key_characteristics": [
            "capex boost",
            "infra spending",
            "tax relief",
            "market rally",
        ],
    },
    {
        "event_id": "india_budget_negative",
        "name": "India Union Budget — negative surprise",
        "category": "policy",
        "start_date": "2019-07-05",
        "end_date": "2019-07-09",
        "nifty_pct_change_5d": -3.8,
        "recovery_days": 30,
        "key_characteristics": [
            "LTCG tax hike",
            "FPI surcharge",
            "selloff",
            "FII exit",
        ],
    },
    {
        "event_id": "taper_tantrum_2013",
        "name": "Taper Tantrum — Fed tapering fears",
        "category": "policy",
        "start_date": "2013-05-22",
        "end_date": "2013-08-30",
        "nifty_pct_change_5d": -8.0,
        "recovery_days": 120,
        "key_characteristics": [
            "Fed taper",
            "EM selloff",
            "rupee crisis",
            "FII outflows",
            "INR depreciation",
        ],
    },
    {
        "event_id": "ai_bubble_concerns_2024",
        "name": "AI valuation bubble concerns 2024",
        "category": "correction",
        "start_date": "2024-07-01",
        "end_date": "2024-08-15",
        "nifty_pct_change_5d": -2.5,
        "recovery_days": 20,
        "key_characteristics": [
            "AI valuation concerns",
            "tech correction",
            "Nvidia selloff",
            "risk-off",
        ],
    },
    {
        "event_id": "rbi_surprise_cut",
        "name": "RBI surprise rate cut",
        "category": "policy",
        "start_date": "2020-05-22",
        "end_date": "2020-05-26",
        "nifty_pct_change_5d": 2.8,
        "recovery_days": None,
        "key_characteristics": [
            "RBI cut",
            "liquidity injection",
            "rate relief",
            "bond rally",
        ],
    },
    {
        "event_id": "rbi_surprise_hike",
        "name": "RBI surprise rate hike",
        "category": "policy",
        "start_date": "2022-05-04",
        "end_date": "2022-05-08",
        "nifty_pct_change_5d": -3.5,
        "recovery_days": 14,
        "key_characteristics": [
            "RBI hike",
            "off-cycle tightening",
            "inflation control",
            "bond selloff",
        ],
    },
]

# Representative headlines per event (3-5 each; hardcoded, no scraping)
HEADLINES_BY_EVENT: dict[str, list[str]] = {
    "dotcom_rise_1999": [
        "Tech stocks surge as internet companies defy gravity with triple-digit gains",
        "Sensex hits new highs as dot-com fever grips Indian markets",
        "IPO frenzy: tech startups list at 300% premium on opening day",
        "Retail investors pile into IT stocks amid global technology euphoria",
    ],
    "dotcom_crash_2000": [
        "Nasdaq crashes 34% in worst week as dot-com bubble bursts",
        "Nifty tanks as global tech selloff spreads to Indian markets",
        "IT stocks in freefall: Infosys and Wipro lose 40% in a month",
        "Margin calls trigger cascading selloff in technology sector",
        "Indian IT sector faces reckoning as US clients slash budgets",
    ],
    "sept11_2001": [
        "Markets shut as terrorist attacks rock New York and Washington",
        "Sensex crashes 5% on reopening after 9/11 attacks",
        "FII selling intensifies as global risk-off sentiment grips markets",
        "Aviation and hotel stocks collapse after terror attacks",
    ],
    "gfc_crash_2008": [
        "Lehman Brothers files for bankruptcy in largest collapse in history",
        "Nifty plunges 8% as global credit markets freeze after Lehman",
        "FIIs pull out Rs 50,000 crore from Indian markets in October",
        "RBI emergency rate cut fails to halt market rout",
        "Global recession fears trigger panic selling across all sectors",
    ],
    "gfc_recovery_2009": [
        "Markets rally as Fed launches quantitative easing programme",
        "Nifty up 15% in March as global risk appetite returns",
        "FIIs return to India with record buying after months of selloff",
        "Sensex posts best quarter in decade on stimulus optimism",
    ],
    "india_demonetization_2016": [
        "PM Modi announces ban on Rs 500 and Rs 1000 currency notes",
        "Nifty falls 6% as demonetization shock rocks Indian markets",
        "Consumer and real estate stocks crash on cash crunch fears",
        "Bank queues grow as demonetization impact spreads across economy",
        "GDP growth forecast slashed as demonetization hits demand",
    ],
    "covid_crash_2020": [
        "WHO declares COVID-19 a global pandemic as cases surge worldwide",
        "Nifty circuit breaker triggered as markets crash 10% in single session",
        "India announces nationwide lockdown to contain coronavirus spread",
        "FIIs sell record Rs 65,000 crore as global funds flee emerging markets",
        "Sensex loses 10,000 points in worst crash since 2008 financial crisis",
    ],
    "covid_recovery_2020": [
        "Pfizer announces 90% effective COVID vaccine sending markets soaring",
        "Nifty recovers all COVID losses hitting pre-pandemic highs",
        "FOMO drives retail investors to pour Rs 10,000 crore into equities",
        "India vaccination drive boosts economic recovery hopes",
    ],
    "rate_hike_cycle_2022": [
        "Fed raises rates 50 bps in fastest tightening cycle in four decades",
        "US inflation hits 8.5%, highest since 1981, triggering global selloff",
        "FIIs dump Rs 1 lakh crore in Indian equities over rate hike fears",
        "Nifty falls 10% in April as global rate fears hit valuations",
        "Bond yields spike to 7.5% as RBI signals hawkish pivot",
    ],
    "india_budget_positive": [
        "Budget 2021 shocks street with massive infrastructure spending boost",
        "Finance minister announces Rs 5.5 lakh crore capex in historic budget",
        "Nifty surges 5% on budget day as fiscal prudence surprises markets",
        "Capex-heavy budget triggers rally in infrastructure and cement stocks",
    ],
    "india_budget_negative": [
        "Budget 2019 hikes surcharge on FPIs triggering sharp market selloff",
        "Nifty falls 3% as LTCG tax increase disappoints Street",
        "FIIs pull back after Finance Minister raises taxes on foreign investors",
        "Markets rattled by wealth tax proposals in Union Budget",
    ],
    "taper_tantrum_2013": [
        "Bernanke hints Fed may taper QE, triggers global bond and equity selloff",
        "Indian rupee crashes to record 68 per dollar on taper fears",
        "Nifty loses 8% in two months as FIIs exit emerging markets en masse",
        "RBI intervenes to defend rupee as current account deficit balloons",
        "EM currencies from India to Indonesia collapse on Fed tapering fears",
    ],
    "ai_bubble_concerns_2024": [
        "Nvidia stock falls 15% on concerns AI capex spending may disappoint",
        "Tech analysts warn of AI valuation bubble as Nasdaq drops sharply",
        "Indian IT stocks slide as global technology sentiment sours",
        "Risk-off mood grips markets after AI spending growth questioned",
    ],
    "rbi_surprise_cut": [
        "RBI cuts repo rate by 40 bps in unscheduled MPC meeting",
        "Indian markets rally 3% after surprise RBI rate reduction",
        "RBI injects Rs 4 lakh crore liquidity to support pandemic-hit economy",
        "Bond yields crash as RBI signals accommodative stance for extended period",
    ],
    "rbi_surprise_hike": [
        "RBI shocks market with off-cycle 40 bps rate hike to fight inflation",
        "Nifty falls 2.5% after surprise RBI tightening rattles bond markets",
        "Governor Das signals inflation priority over growth in hawkish statement",
        "Bank stocks tumble as RBI's surprise hike signals end of easy money era",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _insert_events(db_path: str) -> int:
    """Insert seed events; skip existing event_ids. Returns count inserted."""
    inserted = 0
    with sqlite3.connect(db_path) as conn:
        for ev in MARKET_EVENTS:
            existing = conn.execute(
                "SELECT 1 FROM market_events WHERE event_id = ?", (ev["event_id"],)
            ).fetchone()
            if existing:
                _LOG.info("Skipping existing event: %s", ev["event_id"])
                continue
            conn.execute(
                """INSERT INTO market_events
                   (event_id, name, category, start_date, end_date,
                    nifty_pct_change_5d, recovery_days, key_characteristics)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ev["event_id"],
                    ev["name"],
                    ev["category"],
                    ev["start_date"],
                    ev["end_date"],
                    ev["nifty_pct_change_5d"],
                    ev["recovery_days"],
                    json.dumps(ev["key_characteristics"]),
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def _embed_headlines(collection, embedder) -> int:
    """Embed headlines into ChromaDB; skip if doc_id already present."""
    upserted = 0
    for ev in MARKET_EVENTS:
        headlines = HEADLINES_BY_EVENT.get(ev["event_id"], [])
        for idx, headline in enumerate(headlines):
            doc_id = f"{ev['event_id']}_h{idx}"
            # Check if already in ChromaDB
            existing = collection.get(ids=[doc_id])
            if existing["ids"]:
                _LOG.info("Skipping existing ChromaDB doc: %s", doc_id)
                continue
            embedding = embedder.encode([headline])[0].tolist()
            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[
                    {
                        "id": doc_id,
                        "headline": headline,
                        "body_snippet": headline,
                        "date": ev["start_date"],
                        "source": "seed",
                        "event_id": ev["event_id"],
                        "tags": ev["category"],
                    }
                ],
            )
            upserted += 1
    return upserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def backfill(
    db_path: str = "data/news.db",
    chroma_path: str = "data/chroma",
) -> None:
    _LOG.info("=== Phase 4: Historical news backfill ===")

    setup_db(db_path)
    collection = get_or_create_collection(chroma_path)

    _LOG.info("Inserting market events into SQLite...")
    n_events = _insert_events(db_path)
    _LOG.info("Inserted %d new events (skipped %d existing)", n_events, 15 - n_events)

    _LOG.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
    embedder = _get_embedder()

    _LOG.info("Embedding headlines into ChromaDB...")
    n_headlines = _embed_headlines(collection, embedder)
    _LOG.info("Upserted %d new headline documents", n_headlines)

    # Final counts
    with sqlite3.connect(db_path) as conn:
        event_count = conn.execute("SELECT COUNT(*) FROM market_events").fetchone()[0]
    chroma_count = collection.count()
    _LOG.info(
        "=== Backfill complete: %d events in SQLite, %d docs in ChromaDB ===",
        event_count,
        chroma_count,
    )
    if event_count != 15:
        raise RuntimeError(
            f"Backfill incomplete: expected 15 market events in SQLite, got {event_count}"
        )
    if chroma_count < 45:
        raise RuntimeError(
            f"Backfill incomplete: expected ≥45 ChromaDB docs, got {chroma_count}"
        )


if __name__ == "__main__":
    backfill()
