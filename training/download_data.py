import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BINANCE_URL = "https://api.binance.com/api/v3/klines"
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "15m"
CANDLE_MS = 15 * 60 * 1000
LIMIT = 1000


def fetch_klines(
    symbol: str, interval: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """Fetch up to 1000 candles from Binance. Returns empty DataFrame if no data."""
    for attempt in range(3):
        try:
            resp = requests.get(
                BINANCE_URL,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": LIMIT,
                },
                timeout=10,
            )
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                return pd.DataFrame(
                    columns=["open_time", "open", "high", "low", "close", "volume"]
                )
            df = pd.DataFrame(
                rows,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_vol",
                    "num_trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
            df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["open_time"] = df["open_time"].astype("int64")
            return df
        except requests.HTTPError:
            time.sleep(2**attempt)
    return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])


def download_ticker(symbol: str, years: int = 5) -> pd.DataFrame:
    """Download `years` of 15min candles for `symbol`."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - years * 365 * 24 * 60 * 60 * 1000
    all_dfs = []
    current = start_ms
    with tqdm(desc=symbol, unit="batch") as pbar:
        while current < end_ms:
            batch_end = min(current + LIMIT * CANDLE_MS, end_ms)
            df = fetch_klines(symbol, INTERVAL, current, batch_end)
            if df.empty:
                current = batch_end + CANDLE_MS
                continue
            all_dfs.append(df)
            current = int(df["open_time"].iloc[-1]) + CANDLE_MS
            pbar.update(1)
            time.sleep(0.05)  # stay under rate limit
    if not all_dfs:
        return pd.DataFrame()
    result = pd.concat(all_dfs, ignore_index=True)
    result = (
        result.drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return result


def main():
    out_dir = Path("data/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in TICKERS:
        out_path = out_dir / f"{symbol}_15m.parquet"
        if out_path.exists():
            print(f"{symbol}: already exists, skipping")
            continue
        print(f"Downloading {symbol}...")
        df = download_ticker(symbol, years=5)
        df.to_parquet(out_path, index=False)
        print(f"{symbol}: {len(df):,} candles saved to {out_path}")


if __name__ == "__main__":
    main()
