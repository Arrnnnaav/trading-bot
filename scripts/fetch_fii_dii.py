"""Daily FII/DII data fetcher.

Calls nsefin to get net institutional flows and saves to:
    data/fii_dii/YYYY-MM-DD.json

Format:
    {"date": "2026-06-18", "fii_net_cr": 1234.5, "dii_net_cr": -567.8}

Run:
    python -m scripts.fetch_fii_dii
or call fetch_and_save(output_dir, date) from other modules.

Scheduled: 19:00 IST on weekdays (see main.py Task 6).
NSE publishes FII/DII data after market close (~18:30 IST).
"""

import json
import logging
from datetime import date as DateType
from pathlib import Path

_LOG = logging.getLogger(__name__)


def fetch_and_save(
    output_dir: str = "data/fii_dii",
    fetch_date: DateType | None = None,
) -> Path | None:
    """Fetch FII/DII data for fetch_date (defaults to today) and save as JSON.

    Returns the path written, or None if fetch failed (graceful — does not crash).
    """
    from datetime import date as _date

    if fetch_date is None:
        fetch_date = _date.today()

    date_str = fetch_date.strftime("%Y-%m-%d")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.json"

    if out_path.exists():
        _LOG.info("FII/DII data for %s already exists at %s", date_str, out_path)
        return out_path

    try:
        import nsefin

        raw = nsefin.nse.get_fii_dii_activity()
        # nsefin returns a list of dicts; most recent entry is first
        # Each row has keys: date, fii_net_value (crore), dii_net_value (crore)
        if not raw:
            _LOG.warning("nsefin returned empty data for %s", date_str)
            return None

        # Find today's row or use the most recent
        entry = None
        for row in raw:
            row_date = str(row.get("date", "")).strip()
            if row_date == date_str or row_date.startswith(date_str):
                entry = row
                break
        if entry is None:
            entry = raw[0]  # fallback: most recent

        payload = {
            "date": date_str,
            "fii_net_cr": float(entry.get("fii_net_value", 0.0)),
            "dii_net_cr": float(entry.get("dii_net_value", 0.0)),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _LOG.info(
            "Saved FII/DII data for %s: FII=%.1f Cr, DII=%.1f Cr",
            date_str,
            payload["fii_net_cr"],
            payload["dii_net_cr"],
        )
        return out_path

    except ImportError:
        _LOG.warning(
            "nsefin not installed — skipping FII/DII fetch. pip install nsefin"
        )
        return None
    except Exception as exc:
        _LOG.warning("FII/DII fetch failed for %s: %s", date_str, exc)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = fetch_and_save()
    if result:
        print(f"Saved: {result}")
    else:
        print("Fetch failed or skipped — check logs")
