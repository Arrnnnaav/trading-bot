import numpy as np


def test_sharpe_positive_returns():
    from training.walk_forward import _sharpe

    returns = np.array([0.001] * 252)  # +0.1% daily, 0 std → inf Sharpe clipped
    s = _sharpe(returns)
    assert s > 0


def test_sharpe_zero_std_returns_zero():
    from training.walk_forward import _sharpe

    returns = np.array([0.001] * 252)
    # All same → std=0 → should return 0 (not inf/nan)
    # Actually constant returns have std 0 for ddof=1 when n>1, but same sign means sharpe > 0
    # Test the degenerate case: all zeros
    s = _sharpe(np.zeros(10))
    assert s == 0.0


def test_sharpe_negative_returns():
    from training.walk_forward import _sharpe

    returns = np.array([-0.001] * 252)
    s = _sharpe(returns)
    assert s < 0


def test_passes_gate_all_above():
    from training.walk_forward import passes_gate

    results = {2019: 1.2, 2020: 1.5, 2021: 1.1}
    assert passes_gate(results, min_sharpe=1.0) is True


def test_passes_gate_one_below():
    from training.walk_forward import passes_gate

    results = {2019: 1.2, 2020: 0.8, 2021: 1.1}
    assert passes_gate(results, min_sharpe=1.0) is False


def test_passes_gate_empty():
    from training.walk_forward import passes_gate

    assert passes_gate({}) is False


def test_run_walk_forward_returns_dict(monkeypatch):
    import pandas as pd
    import numpy as np
    from training import walk_forward as wf

    # Patch load_india_data to return synthetic 5yr daily data
    n = 252 * 5  # 5 years
    idx = pd.date_range("2017-01-01", periods=n, freq="B")
    closes = 10000.0 + np.cumsum(np.random.randn(n) * 50)
    labels = (["LONG", "SHORT", "HOLD"] * (n // 3 + 1))[:n]
    synthetic = pd.DataFrame(
        {
            "date": idx,
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

    monkeypatch.setattr(wf, "load_india_data", lambda **kw: synthetic)

    results = wf.run_walk_forward(start_oos_year=2021, end_oos_year=2021)
    assert isinstance(results, dict)
    # 2021 should be in results (enough training data before 2021)
    assert 2021 in results
    assert isinstance(results[2021], float)
