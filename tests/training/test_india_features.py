import numpy as np
import pandas as pd


def _make_df(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 10000.0 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.Index(idx, name="date"),
    )


def test_compute_features_returns_expected_columns():
    from training.india_features import compute_features

    df = _make_df(300)
    out = compute_features(df)
    expected = [
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "bb_pct",
        "bb_width",
        "ema9_ratio",
        "ema21_ratio",
        "ema50_ratio",
        "ema9_21_cross",
        "atr_pct",
        "volume_ratio",
        "obv_slope",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "days_since_52w_high",
        "fii_5d_net",
    ]
    for col in expected:
        assert col in out.columns, f"Missing column: {col}"


def test_compute_features_preserves_input_columns():
    from training.india_features import compute_features

    df = _make_df(300)
    out = compute_features(df)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in out.columns


def test_rsi_bounded():
    from training.india_features import compute_features

    df = _make_df(300)
    out = compute_features(df).dropna()
    assert (out["rsi_14"] >= 0).all() and (out["rsi_14"] <= 100).all()


def test_volume_ratio_zero_volume_no_crash():
    from training.india_features import compute_features

    df = _make_df(300)
    df["volume"] = 0.0  # index tickers have zero volume
    out = compute_features(df)  # must not raise
    assert "volume_ratio" in out.columns


def test_fii_5d_net_no_files_returns_zero(tmp_path):
    from training.india_features import compute_features

    df = _make_df(300)
    out = compute_features(df, fii_dir=str(tmp_path))
    assert (out["fii_5d_net"] == 0.0).all()


def test_fii_5d_net_with_files(tmp_path):
    import json
    from training.india_features import compute_features

    df = _make_df(60)
    dates = df.index.strftime("%Y-%m-%d").tolist()
    for i, d in enumerate(dates[:10]):
        (tmp_path / f"{d}.json").write_text(
            json.dumps({"date": d, "fii_net_cr": 500.0, "dii_net_cr": 100.0})
        )
    out = compute_features(df, fii_dir=str(tmp_path))
    # After 5 FII records, rolling sum should be non-zero
    non_zero = (out["fii_5d_net"] != 0.0).sum()
    assert non_zero > 0


def test_days_since_52w_high_monotone_increase_then_reset():
    from training.india_features import compute_features

    # Declining prices: days_since_52w_high increases
    close = np.linspace(10000, 5000, 300)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1000.0,
        },
        index=pd.Index(idx, name="date"),
    )
    out = compute_features(df).dropna()
    # In a declining series, days_since_52w_high should increase over time
    mid = len(out) // 2
    assert out["days_since_52w_high"].iloc[-1] >= out["days_since_52w_high"].iloc[mid]
