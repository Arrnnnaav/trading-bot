import pytest
import numpy as np
import pandas as pd
import torch


def _make_labeled_india_df(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = np.linspace(10000, 12000, n)
    labels = (["LONG", "SHORT", "HOLD"] * (n // 3 + 1))[:n]
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": 1000.0,
            # Simulate a few feature columns
            "rsi_14": np.random.uniform(30, 70, n),
            "macd_line": np.random.randn(n),
            "macd_signal": np.random.randn(n),
            "macd_hist": np.random.randn(n),
            "bb_pct": np.random.uniform(0, 1, n),
            "bb_width": np.random.uniform(0.01, 0.05, n),
            "ema9_ratio": np.ones(n),
            "ema21_ratio": np.ones(n),
            "ema50_ratio": np.ones(n),
            "ema9_21_cross": np.ones(n),
            "atr_pct": np.random.uniform(0.005, 0.02, n),
            "volume_ratio": np.ones(n),
            "obv_slope": np.random.randn(n),
            "ret_1d": np.random.randn(n) * 0.01,
            "ret_5d": np.random.randn(n) * 0.02,
            "ret_20d": np.random.randn(n) * 0.04,
            "days_since_52w_high": np.arange(n),
            "fii_5d_net": np.random.randn(n) * 500,
            "label": labels,
        },
        index=pd.Index(idx, name="date"),
    )


def _make_labeled_ohlcv_df(n=200):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = np.linspace(10000, 12000, n)
    labels = (["LONG", "SHORT", "HOLD"] * (n // 3 + 1))[:n]
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": 1000.0,
            "label": labels,
        },
        index=pd.Index(idx, name="date"),
    )


def test_india_dataset_length():
    from training.dataset import IndiaDataset

    df = _make_labeled_india_df(100)
    ds = IndiaDataset(df)
    assert len(ds) == 100


def test_india_dataset_item_shapes():
    from training.dataset import IndiaDataset

    df = _make_labeled_india_df(100)
    ds = IndiaDataset(df)
    x, label = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 1
    assert x.shape[0] == len(ds.feature_names)
    assert isinstance(label, int)
    assert label in (0, 1, 2)


def test_india_dataset_excludes_ohlcv_from_features():
    from training.dataset import IndiaDataset

    df = _make_labeled_india_df(100)
    ds = IndiaDataset(df)
    for excluded in ("open", "high", "low", "close", "volume", "label"):
        assert excluded not in ds.feature_names


def test_kronos_dataset_length():
    from training.dataset import KronosDataset

    df = _make_labeled_ohlcv_df(200)
    ds = KronosDataset(df)
    assert len(ds) == 200 - KronosDataset.SEQ_LEN


def test_kronos_dataset_item_shape():
    from training.dataset import KronosDataset

    df = _make_labeled_ohlcv_df(200)
    ds = KronosDataset(df)
    x, label = ds[0]
    assert x.shape == torch.Size([KronosDataset.SEQ_LEN - 1, 5])
    assert isinstance(label, int)
    assert label in (0, 1, 2)


def test_split_india_datasets_no_leakage():
    from training.dataset import split_india_datasets

    df = _make_labeled_india_df(1000)
    train, val, test = split_india_datasets(df, train_frac=0.80, val_frac=0.10)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    assert len(train) > len(val) > 0


def test_load_india_data_missing_dir_raises(tmp_path):
    from training.dataset import load_india_data

    with pytest.raises(FileNotFoundError):
        load_india_data(historical_dir=str(tmp_path / "nonexistent"))
