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


class CryptoDataset(Dataset):
    """
    Each item: (close_window [WINDOW], label_int)
    close_window is z-score normalized per sample to prevent look-ahead bias.
    """

    def __init__(self, df: pd.DataFrame, window: int = WINDOW):
        self.closes = df["close"].to_numpy(dtype=np.float32)
        self.labels = df["label"].map(LABEL_MAP).to_numpy(dtype=np.int64)
        self.window = window
        # Valid start indices: window .. len-1
        self.indices = list(range(window, len(df)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        end = self.indices[idx]
        start = end - self.window
        window = self.closes[start:end].copy()
        # Z-score normalize per sample
        mean = window.mean()
        std = window.std() + 1e-8
        window = (window - mean) / std
        return torch.tensor(window, dtype=torch.float32), int(self.labels[end])


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
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    total = counts.sum()
    # Inverse frequency, clipped to [1, 5]
    weights = np.clip(total / (3 * counts + 1e-8), 1.0, 5.0)
    return torch.tensor(weights, dtype=torch.float32)
