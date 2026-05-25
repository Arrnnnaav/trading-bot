import torch
import pandas as pd
import numpy as np


def make_tiny_df(n=200):
    closes = np.random.randn(n).cumsum() + 100
    return pd.DataFrame(
        {
            "open_time": range(n),
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": [1000.0] * n,
            "label": (["LONG", "SHORT", "HOLD"] * (n // 3 + 1))[:n],
        }
    )


def test_train_one_epoch_runs():
    from training.train import train_epoch
    from training.model import ChronosClassifier
    from training.dataset import CryptoDataset
    from torch.utils.data import DataLoader

    df = make_tiny_df(200)
    ds = CryptoDataset(df, window=96)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    model = ChronosClassifier()
    model.freeze_encoder()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    loss = train_epoch(model, loader, optimizer, device=torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss > 0


def test_eval_returns_metrics():
    from training.train import eval_epoch
    from training.model import ChronosClassifier
    from training.dataset import CryptoDataset
    from torch.utils.data import DataLoader

    df = make_tiny_df(200)
    ds = CryptoDataset(df, window=96)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    model = ChronosClassifier()
    metrics = eval_epoch(model, loader, device=torch.device("cpu"))
    assert "loss" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0
