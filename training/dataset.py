import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple

LABEL_MAP = {"LONG": 0, "SHORT": 1, "HOLD": 2}
LABEL_NAMES = ["LONG", "SHORT", "HOLD"]
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
WINDOW = 96  # candles fed to model (24 hours of 15min data)
N_STAT_FEATURES = 7  # 6 momentum/vol features + 1 volume surge feature


def compute_stat_features(closes: np.ndarray, volumes: np.ndarray = None) -> np.ndarray:
    """
    7 hand-crafted features fused at head (6 price momentum/vol + 1 volume surge).
    All values are raw ratios — no additional normalization needed.
    Requires len(closes) >= 21.

    Features:
      ret_1   — 1-period return
      ret_5   — 5-period return
      ret_20  — 20-period return
      pvol_5  — 5-period price volatility (std of returns)
      pvol_20 — 20-period price volatility
      pvol_ratio — pvol_5 / pvol_20 (vol expansion)
      vol_surge  — current volume / 20-period mean volume (0.0 if volumes not provided)
    """
    rets = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    ret_1 = float(rets[-1])
    ret_5 = float((closes[-1] - closes[-6]) / (np.abs(closes[-6]) + 1e-8))
    ret_20 = float((closes[-1] - closes[-21]) / (np.abs(closes[-21]) + 1e-8))
    pvol_5 = float(rets[-5:].std() + 1e-8)
    pvol_20 = float(rets[-20:].std() + 1e-8)
    pvol_ratio = float(pvol_5 / (pvol_20 + 1e-8))

    if volumes is not None and len(volumes) >= 20:
        vol_mean_20 = float(volumes[-20:].mean() + 1e-8)
        vol_surge = float(volumes[-1] / vol_mean_20)
    else:
        vol_surge = 1.0  # neutral if volume not available

    return np.array(
        [ret_1, ret_5, ret_20, pvol_5, pvol_20, pvol_ratio, vol_surge],
        dtype=np.float32,
    )


class CryptoDataset(Dataset):
    """
    Each item: (close_window [WINDOW], stat_features [N_STAT_FEATURES], label_int)
    close_window is z-score normalized per sample to prevent look-ahead bias.
    stat_features: 6 price momentum/vol ratios + 1 volume surge (all from raw data).
    """

    def __init__(self, df: pd.DataFrame, window: int = WINDOW):
        self.closes = df["close"].to_numpy(dtype=np.float32)
        self.volumes = (
            df["volume"].to_numpy(dtype=np.float32) if "volume" in df.columns else None
        )
        self.labels = df["label"].map(LABEL_MAP).to_numpy(dtype=np.int64)
        self.window = window
        self.indices = list(range(window, len(df)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        end = self.indices[idx]
        start = end - self.window
        raw_window = self.closes[start:end].copy()
        vol_window = (
            self.volumes[start:end].copy() if self.volumes is not None else None
        )

        stat = compute_stat_features(raw_window, vol_window)

        mean = raw_window.mean()
        std = raw_window.std() + 1e-8
        normed = (raw_window - mean) / std

        return (
            torch.tensor(normed, dtype=torch.float32),
            torch.tensor(stat, dtype=torch.float32),
            int(self.labels[end]),
        )


def load_all_tickers(data_dir: str = "data/training") -> pd.DataFrame:
    """Concatenate labeled parquet files for all tickers."""
    dfs = []
    for symbol in TICKERS:
        path = Path(data_dir) / f"{symbol}_15m_labeled.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}. Run training/label_data.py first."
            )
        df = pd.read_parquet(path)
        df["symbol"] = symbol
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("open_time").reset_index(drop=True)
    return combined


def split_datasets(
    df: pd.DataFrame,
    window: int = WINDOW,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> Tuple[CryptoDataset, CryptoDataset, CryptoDataset]:
    """
    Time-ordered split. Never shuffled — prevents future data leakage.
    Returns (train_dataset, val_dataset, test_dataset).
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return (
        CryptoDataset(train_df, window),
        CryptoDataset(val_df, window),
        CryptoDataset(test_df, window),
    )


def compute_class_weights(dataset: CryptoDataset) -> torch.Tensor:
    """Returns [3] weight tensor for CrossEntropyLoss — upweights LONG/SHORT."""
    labels = [dataset[i][2] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    total = counts.sum()
    # Inverse frequency, clipped to [1, 5]
    weights = np.clip(total / (3 * counts + 1e-8), 1.0, 5.0)
    return torch.tensor(weights, dtype=torch.float32)
