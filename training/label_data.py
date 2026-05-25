import pandas as pd
import numpy as np
from pathlib import Path

TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TARGET_PCT = 0.010  # +1.0% for LONG, -1.0% for SHORT
STOP_PCT = 0.005  # -0.5% stop for LONG,  +0.5% stop for SHORT
WINDOW = 4  # look-ahead candles


def label_candles(
    df: pd.DataFrame,
    target_pct: float = TARGET_PCT,
    stop_pct: float = STOP_PCT,
    window: int = WINDOW,
) -> pd.DataFrame:
    """
    Add 'label' column: LONG / SHORT / HOLD.
    LONG  — close rises >= target_pct before falling <= stop_pct within window.
    SHORT — close falls >= target_pct before rising <= stop_pct within window.
    HOLD  — neither threshold reached.
    Last `window` rows always HOLD (no complete look-ahead).
    """
    closes = df["close"].to_numpy(dtype=np.float64)
    n = len(closes)
    labels = ["HOLD"] * n

    for i in range(n - window):
        entry = closes[i]
        long_target = entry * (1 + target_pct)
        long_stop = entry * (1 - stop_pct)
        short_target = entry * (1 - target_pct)
        short_stop = entry * (1 + stop_pct)

        long_label = short_label = False
        for j in range(i + 1, i + 1 + window):
            c = closes[j]
            if c <= long_stop:
                break
            if c >= long_target:
                long_label = True
                break

        for j in range(i + 1, i + 1 + window):
            c = closes[j]
            if c >= short_stop:
                break
            if c <= short_target:
                short_label = True
                break

        if long_label and not short_label:
            labels[i] = "LONG"
        elif short_label and not long_label:
            labels[i] = "SHORT"
        # else HOLD (ambiguous or neither)

    df = df.copy()
    df["label"] = labels
    return df


def main():
    data_dir = Path("data/training")
    for symbol in TICKERS:
        src = data_dir / f"{symbol}_15m.parquet"
        dst = data_dir / f"{symbol}_15m_labeled.parquet"
        if dst.exists():
            print(f"{symbol}: labeled file exists, skipping")
            continue
        print(f"Labeling {symbol}...")
        df = pd.read_parquet(src)
        labeled = label_candles(df)
        counts = labeled["label"].value_counts()
        print(
            f"  {len(labeled):,} rows | LONG={counts.get('LONG', 0):,} SHORT={counts.get('SHORT', 0):,} HOLD={counts.get('HOLD', 0):,}"
        )
        labeled.to_parquet(dst, index=False)
        print(f"  Saved to {dst}")


if __name__ == "__main__":
    main()
