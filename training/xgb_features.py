"""
xgb_features.py — compute technical indicator features from OHLCV parquet files.

Features computed per row (no look-ahead — all indicators use only past data):
  RSI(14), MACD(12,26,9), BB %B + bandwidth, ATR(14)/close,
  EMA(9/21/50) ratio to close, SMA(20/50) ratio to close,
  Volume ratio (vol / SMA_vol_20), Stoch %K/%D(14,3),
  Williams %R(14), CCI(20), ADX(14), ROC(10), Momentum(10),
  HL ratio, OBV normalized, candle body/wick ratios

Output: DataFrame with feature columns + label (no NaN rows).
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

LABEL_MAP = {"LONG": 0, "SHORT": 1, "HOLD": 2}
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to df in-place, return cleaned copy."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── momentum ──────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.rsi(c, length=14)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    df["macd_line"] = macd["MACD_12_26_9"] / c
    df["macd_signal"] = macd["MACDs_12_26_9"] / c
    df["macd_hist"] = macd["MACDh_12_26_9"] / c

    df["roc_10"] = ta.roc(c, length=10)
    df["mom_10"] = ta.mom(c, length=10) / c

    stoch = ta.stoch(h, l, c, k=14, d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]

    df["willr_14"] = ta.willr(h, l, c, length=14)
    df["cci_20"] = ta.cci(h, l, c, length=20)

    # ── trend ─────────────────────────────────────────────────────────────────
    adx = ta.adx(h, l, c, length=14)
    df["adx_14"] = adx["ADX_14"]
    df["dmp_14"] = adx["DMP_14"]
    df["dmn_14"] = adx["DMN_14"]

    df["ema9_ratio"] = ta.ema(c, length=9) / c - 1
    df["ema21_ratio"] = ta.ema(c, length=21) / c - 1
    df["ema50_ratio"] = ta.ema(c, length=50) / c - 1
    df["sma20_ratio"] = ta.sma(c, length=20) / c - 1
    df["sma50_ratio"] = ta.sma(c, length=50) / c - 1

    # ── volatility ────────────────────────────────────────────────────────────
    atr = ta.atr(h, l, c, length=14)
    df["atr_ratio"] = atr / c

    bb = ta.bbands(c, length=20, std=2)
    df["bb_pct"] = bb["BBP_20_2.0_2.0"]
    df["bb_bw"] = bb["BBB_20_2.0_2.0"]

    # ── volume ────────────────────────────────────────────────────────────────
    vol_sma = ta.sma(v, length=20)
    df["vol_ratio"] = v / (vol_sma + 1e-8)
    obv = ta.obv(c, v)
    df["obv_norm"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-8)

    # ── candle structure ──────────────────────────────────────────────────────
    body = (c - o).abs()
    hl_range = (h - l).clip(lower=1e-8)
    df["body_ratio"] = body / hl_range
    df["upper_wick"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / hl_range
    df["lower_wick"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / hl_range
    df["hl_ratio"] = hl_range / c

    return df.dropna()


def load_features(data_dir: str = "data/training") -> pd.DataFrame:
    dfs = []
    for symbol in TICKERS:
        path = Path(data_dir) / f"{symbol}_15m_labeled.parquet"
        df = pd.read_parquet(path).copy()
        df["symbol"] = symbol
        df = _compute_features(df)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("open_time").reset_index(drop=True)
    return combined


FEATURE_COLS = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "roc_10",
    "mom_10",
    "stoch_k",
    "stoch_d",
    "willr_14",
    "cci_20",
    "adx_14",
    "dmp_14",
    "dmn_14",
    "ema9_ratio",
    "ema21_ratio",
    "ema50_ratio",
    "sma20_ratio",
    "sma50_ratio",
    "atr_ratio",
    "bb_pct",
    "bb_bw",
    "vol_ratio",
    "obv_norm",
    "body_ratio",
    "upper_wick",
    "lower_wick",
    "hl_ratio",
]


def get_splits(df: pd.DataFrame, train_frac=0.80, val_frac=0.10):
    """Time-ordered split. Returns (X_train, y_train, X_val, y_val, X_test, y_test)."""
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))

    y = df["label"].map(LABEL_MAP).to_numpy()
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)

    return (
        X[:t1],
        y[:t1],
        X[t1:t2],
        y[t1:t2],
        X[t2:],
        y[t2:],
    )
