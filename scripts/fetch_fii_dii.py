"""Fetch today's FII/DII activity from NSE via nsefin and cache to data/fii_dii/.

Usage:
    python -m scripts.fetch_fii_dii           # fetch today
    python -m scripts.fetch_fii_dii --date 2026-06-18

Run at 19:00 IST after NSE publishes daily data.
Output: data/fii_dii/YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import nsefin

_LOG = logging.getLogger(__name__)


def _parse_date(raw_date: str) -> str:
    """Convert '18-Jun-2026' or '2026-06-18' to 'YYYY-MM-DD'."""
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw_date.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {raw_date!r}")


def _parse_cr(value: str | float | int) -> float:
    """Parse ₹ crore value from string like '2,500.50' or '-800.25'."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(re.sub(r"[,\s]", "", str(value)))


def fetch_fii_dii(data_dir: Path | str = "data/fii_dii") -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw = nsefin.nse.get_fii_dii_activity()
    date_str = _parse_date(raw["date"])
    out_path = data_dir / f"{date_str}.json"

    record = {
        "date": date_str,
        "fii_net_cr": _parse_cr(raw["fii_net_purchase_sales"]),
        "dii_net_cr": _parse_cr(raw["dii_net_purchase_sales"]),
        "source": "nsefin",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(record, indent=2))
    _LOG.info(
        "FII/DII saved to %s: FII=%+.0f DII=%+.0f Cr",
        out_path,
        record["fii_net_cr"],
        record["dii_net_cr"],
    )
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/fii_dii")
    args = parser.parse_args()
    fetch_fii_dii(data_dir=args.data_dir)
