import numpy as np
import pandas as pd


def _synthetic_df(n=800):
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


def test_train_smoke(tmp_path, monkeypatch):
    from training import train_kronos_india as tk

    monkeypatch.setattr(tk, "load_india_data", lambda **kw: _synthetic_df(800))
    monkeypatch.setattr(tk, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(tk, "PHASE1_EPOCHS", 1)
    monkeypatch.setattr(tk, "PHASE2_EPOCHS", 1)

    model, best_f1 = tk.train(device_str="cpu")
    assert (tmp_path / "best.pt").exists()
    assert 0.0 <= best_f1 <= 1.0


def test_train_saves_loadable_model(tmp_path, monkeypatch):
    from training import train_kronos_india as tk
    from training.kronos_model import KronosClassifier

    monkeypatch.setattr(tk, "load_india_data", lambda **kw: _synthetic_df(800))
    monkeypatch.setattr(tk, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(tk, "PHASE1_EPOCHS", 1)
    monkeypatch.setattr(tk, "PHASE2_EPOCHS", 1)

    tk.train(device_str="cpu")

    loaded = KronosClassifier.load(str(tmp_path / "best.pt"))
    x = [[0.001, 0.002, -0.001, 0.0015, 0.5]] * 63
    result = loaded.predict(x)
    assert "long_prob" in result
    assert abs(sum(result.values()) - 1.0) < 1e-5
