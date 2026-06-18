import numpy as np


def _make_xy(n=500, n_features=18):
    X = np.random.randn(n, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n)
    return X, y


def test_train_lgb_returns_fitted_model():
    from training.train_xgb_india import _train_lgb

    X, y = _make_xy(200)
    model = _train_lgb(X, y, n_estimators=5)
    preds = model.predict(X)
    assert len(preds) == 200
    assert set(preds).issubset({0, 1, 2})


def test_train_lgb_with_validation():
    from training.train_xgb_india import _train_lgb

    X_tr, y_tr = _make_xy(300)
    X_val, y_val = _make_xy(100)
    model = _train_lgb(X_tr, y_tr, X_val=X_val, y_val=y_val, n_estimators=10)
    assert model.predict(X_val).shape == (100,)


def test_train_saves_artifacts(tmp_path, monkeypatch):
    from training import train_xgb_india

    # Patch load_india_data to return synthetic data
    import pandas as pd
    import numpy as np

    def _fake_load(historical_dir, fii_dir):
        n = 300
        closes = np.linspace(10000, 12000, n)
        labels = (["LONG", "SHORT", "HOLD"] * (n // 3 + 1))[:n]
        return pd.DataFrame(
            {
                "open": closes,
                "high": closes * 1.005,
                "low": closes * 0.995,
                "close": closes,
                "volume": 1000.0,
                "rsi_14": 50.0,
                "macd_line": 0.0,
                "macd_signal": 0.0,
                "macd_hist": 0.0,
                "bb_pct": 0.5,
                "bb_width": 0.02,
                "ema9_ratio": 1.0,
                "ema21_ratio": 1.0,
                "ema50_ratio": 1.0,
                "ema9_21_cross": 1.0,
                "atr_pct": 0.01,
                "volume_ratio": 1.0,
                "obv_slope": 0.0,
                "ret_1d": 0.001,
                "ret_5d": 0.005,
                "ret_20d": 0.01,
                "days_since_52w_high": range(n),
                "fii_5d_net": 0.0,
                "label": labels,
            }
        )

    monkeypatch.setattr(train_xgb_india, "load_india_data", _fake_load)
    monkeypatch.setattr(train_xgb_india, "MODEL_DIR", tmp_path)

    model, f1 = train_xgb_india.train(n_trials=2)
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "feature_importance.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert 0.0 <= f1 <= 1.0
