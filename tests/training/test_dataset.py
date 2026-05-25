import pandas as pd
import numpy as np
import torch


def make_labeled_df(n=200):
    closes = np.linspace(100, 110, n).tolist()
    return pd.DataFrame(
        {
            "open_time": range(n),
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "label": ["LONG", "SHORT", "HOLD"] * (n // 3) + ["HOLD"] * (n % 3),
        }
    )


def test_dataset_length():
    from training.dataset import CryptoDataset

    df = make_labeled_df(200)
    ds = CryptoDataset(df, window=96)
    # Valid samples start at index 96 (window), end at n-1
    assert len(ds) == 200 - 96


def test_dataset_item_shapes():
    from training.dataset import CryptoDataset

    df = make_labeled_df(200)
    ds = CryptoDataset(df, window=96)
    close_window, label = ds[0]
    assert close_window.shape == torch.Size([96])
    assert isinstance(label, int)
    assert label in (0, 1, 2)


def test_dataset_close_is_normalized():
    from training.dataset import CryptoDataset

    df = make_labeled_df(200)
    ds = CryptoDataset(df, window=96)
    close_window, _ = ds[0]
    # z-score: mean ~0, std ~1
    assert abs(close_window.mean().item()) < 0.5
    assert 0.1 < close_window.std().item() < 10.0


def test_split_no_leakage():
    from training.dataset import split_datasets

    df = make_labeled_df(1000)
    train, val, test = split_datasets(df, window=96)
    # Each split needs its own warmup window — train is 80%, val/test 10% each
    # n=1000: train=800 rows, val=100 rows, test=100 rows
    assert len(train) == 800 - 96
    assert len(val) == 100 - 96
    assert len(test) == 100 - 96
    # Train is largest (time-ordered, no shuffle)
    assert len(train) > len(val)
