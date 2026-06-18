"""Download 25yr daily OHLCV for Indian indices from yfinance.

Usage:
    python -m scripts.fetch_historical          # full download
    python -m scripts.fetch_historical --update # append since last date

Output: data/historical/{NSEI,NSEBANK,BSESN,CNXIT}_daily.parquet
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = {
    "NSEI": "^NSEI",
    "NSEBANK": "^NSEBANK",
    "BSESN": "^BSESN",
    "CNXIT": "^CNXIT",
}
DEFAULT_START = "2000-01-01"


def fetch_all(
    data_dir: Path | str = "data/historical",
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    end_date = end_date or date.today().isoformat()

    for name, ticker in TICKERS.items():
        out_path = data_dir / f"{name}.parquet"

        # Incremental: if file exists, start from day after last stored date
        fetch_start = start_date
        existing_df: pd.DataFrame | None = None
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            if not existing_df.empty:
                last_date = existing_df.index.max()
                fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"Fetching {name} ({ticker}) from {fetch_start} to {end_date}...")
        raw = yf.download(
            ticker,
            start=fetch_start,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            print(f"  No new data for {name}.")
            continue

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"

        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

        df.to_parquet(out_path)
        print(f"  Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--data-dir", default="data/historical")
    args = parser.parse_args()
    fetch_all(data_dir=args.data_dir, start_date=args.start, end_date=args.end)
