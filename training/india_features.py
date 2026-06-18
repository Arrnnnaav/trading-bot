"""Pandas-based feature engineering for India daily OHLCV data."""

import json
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal indicator helpers (pandas-native, matching agents/_indicators.py)
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI via EWM (alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    line = fast - slow
    signal = _ema(line, 9)
    return line, signal


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + num_std * std, mid, mid - num_std * std


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _obv_slope(close: pd.Series, volume: pd.Series, window: int = 5) -> pd.Series:
    direction = close.diff().map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    return obv.diff(window) / window


def _fii_5d_net(df: pd.DataFrame, fii_dir: str) -> pd.Series:
    """Join 5-day rolling FII net from JSON files onto df's DatetimeIndex."""
    fii_path = Path(fii_dir)
    records = []
    for f in sorted(fii_path.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            records.append(
                {
                    "date": pd.Timestamp(data["date"]),
                    "fii_net_cr": float(data["fii_net_cr"]),
                }
            )
        except Exception:
            pass
    if not records:
        return pd.Series(0.0, index=df.index)
    fii_df = pd.DataFrame(records).set_index("date").sort_index()
    fii_df["fii_5d_net"] = fii_df["fii_net_cr"].rolling(5, min_periods=1).sum()
    # Map onto df's date index; fill missing dates with 0
    return df.index.map(fii_df["fii_5d_net"]).fillna(0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_features(df: pd.DataFrame, fii_dir: str = "data/fii_dii") -> pd.DataFrame:
    """
    Compute all training features from a daily OHLCV DataFrame.

    Input:  DataFrame with DatetimeIndex (name='date') or 'date' column,
            columns: open, high, low, close, volume (volume may be 0 for index tickers).
    Output: Same DataFrame with feature columns appended.
            Call .dropna() after this to remove warm-up rows.
    """
    df = df.copy()
    close = df["close"]
    volume = df.get("volume", pd.Series(0.0, index=df.index)).replace(0, np.nan)

    # --- Momentum ---
    df["rsi_14"] = _rsi(close, 14)

    macd_line, macd_signal = _macd(close)
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_line - macd_signal

    # --- Bollinger Bands ---
    bb_upper, bb_mid, bb_lower = _bollinger(close)
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_pct"] = (close - bb_lower) / bb_range  # %B: 0=lower, 1=upper
    df["bb_width"] = bb_range / bb_mid.replace(0, np.nan)

    # --- EMA ratios ---
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    df["ema9_ratio"] = close / ema9.replace(0, np.nan)
    df["ema21_ratio"] = close / ema21.replace(0, np.nan)
    df["ema50_ratio"] = close / ema50.replace(0, np.nan)
    df["ema9_21_cross"] = ema9 / ema21.replace(0, np.nan)

    # --- Volatility ---
    df["atr_pct"] = _atr(df) / close.replace(0, np.nan)

    # --- Volume (0 for index tickers → NaN → volume_ratio=NaN → filled with 1.0) ---
    vol_ma20 = volume.rolling(20).mean()
    df["volume_ratio"] = (volume / vol_ma20.replace(0, np.nan)).fillna(1.0)
    df["obv_slope"] = _obv_slope(close, volume.fillna(0))

    # --- Returns ---
    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)

    # --- Days since 52-week high ---
    rolling_max_252 = close.rolling(252, min_periods=1).max()
    is_high = close >= rolling_max_252
    days_since = []
    count = 0
    for flag in is_high:
        count = 0 if flag else count + 1
        days_since.append(count)
    df["days_since_52w_high"] = days_since

    # --- FII 5-day rolling net ---
    df["fii_5d_net"] = _fii_5d_net(df, fii_dir)

    return df
