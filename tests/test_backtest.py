import pandas as pd
import numpy as np
from unittest.mock import MagicMock


def make_test_df(n=300):
    closes = np.linspace(100, 120, n)
    return pd.DataFrame(
        {
            "open_time": range(n),
            "open": closes * 0.999,
            "high": closes * 1.015,
            "low": closes * 0.985,
            "close": closes,
            "volume": [1000.0] * n,
            "label": (["LONG"] * 5 + ["HOLD"] * 5) * (n // 10) + ["HOLD"] * (n % 10),
        }
    )


def test_backtest_produces_report():
    from backtesting.crypto_backtest import run_backtest
    from core.models import Direction

    mock_model = MagicMock()
    mock_model.predict.return_value = (Direction.LONG, 0.72)
    mock_model.eval = MagicMock()

    df = make_test_df(300)
    results = run_backtest(df, model=mock_model, window=96)

    assert "total_signals" in results
    assert "win_rate" in results
    assert "signal_rate_pct" in results
    assert isinstance(results["total_signals"], int)


def test_backtest_zero_signals_if_always_hold():
    from backtesting.crypto_backtest import run_backtest
    from core.models import Direction

    mock_model = MagicMock()
    mock_model.predict.return_value = (Direction.HOLD, 0.0)
    mock_model.eval = MagicMock()

    df = make_test_df(300)
    results = run_backtest(df, model=mock_model, window=96)
    assert results["total_signals"] == 0
