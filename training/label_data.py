import pandas as pd
import numpy as np
import os
from pathlib import Path

TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TARGET_PCT = 0.010  # +1.0% for LONG, -1.0% for SHORT
STOP_PCT = 0.005  # -0.5% stop for LONG,  +0.5% stop for SHORT
WINDOW = 4  # look-ahead candles
ATR_PERIOD = 14
ATR_TARGET_MULT = 1.5
ATR_STOP_MULT = 1.5
ATR_WINDOW = 24  # 6 hours of 15m candles


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


def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def label_candles_triple_barrier(
    df: pd.DataFrame,
    target_atr_mult: float = ATR_TARGET_MULT,
    stop_atr_mult: float = ATR_STOP_MULT,
    window: int = ATR_WINDOW,
    atr_period: int = ATR_PERIOD,
) -> pd.DataFrame:
    """
    ATR-based triple-barrier labels.

    LONG if the upside barrier is touched before the downside barrier.
    SHORT if the downside barrier is touched before the upside barrier.
    HOLD if neither barrier is touched inside the time window.
    """
    df = df.copy()
    atr = _compute_atr(df, atr_period)
    labels = ["HOLD"] * len(df)

    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)

    for i in range(len(df) - window):
        atr_val = atr.iloc[i]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        entry = closes[i]
        upper = entry + target_atr_mult * atr_val
        lower = entry - stop_atr_mult * atr_val

        for j in range(i + 1, i + 1 + window):
            hit_upper = highs[j] >= upper
            hit_lower = lows[j] <= lower
            if hit_upper and not hit_lower:
                labels[i] = "LONG"
                break
            if hit_lower and not hit_upper:
                labels[i] = "SHORT"
                break
            if hit_upper and hit_lower:
                labels[i] = "HOLD"
                break

    df["label"] = labels
    df["label_method"] = "atr_triple_barrier"
    df["label_target_atr_mult"] = target_atr_mult
    df["label_stop_atr_mult"] = stop_atr_mult
    df["label_window"] = window
    return df


def main():
    data_dir = Path("data/training")
    method = os.environ.get("LABEL_METHOD", "atr_triple_barrier").lower()
    force = os.environ.get("FORCE_RELABEL", "false").lower() == "true"
    for symbol in TICKERS:
        src = data_dir / f"{symbol}_15m.parquet"
        dst = data_dir / f"{symbol}_15m_labeled.parquet"
        if dst.exists() and not force:
            print(f"{symbol}: labeled file exists, skipping")
            continue
        print(f"Labeling {symbol}...")
        df = pd.read_parquet(src)
        if method == "fixed_pct":
            labeled = label_candles(df)
        else:
            labeled = label_candles_triple_barrier(df)
        counts = labeled["label"].value_counts()
        print(
            f"  {len(labeled):,} rows | LONG={counts.get('LONG', 0):,} SHORT={counts.get('SHORT', 0):,} HOLD={counts.get('HOLD', 0):,}"
        )
        labeled.to_parquet(dst, index=False)
        print(f"  Saved to {dst}")


if __name__ == "__main__":
    main()
