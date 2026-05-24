import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_fetch_klines_returns_dataframe():
    from training.download_data import fetch_klines

    mock_data = [
        [
            1609459200000,
            "29000.0",
            "29500.0",
            "28800.0",
            "29300.0",
            "100.5",
            1609460100000,
            "2945000.0",
            150,
            "60.0",
            "1767000.0",
            "0",
        ],
    ]
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        df = fetch_klines(
            "BTCUSDT", "15m", start_ms=1609459200000, end_ms=1609460100000
        )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open_time", "open", "high", "low", "close", "volume"]
    assert len(df) == 1
    assert df["close"].iloc[0] == pytest.approx(29300.0)


def test_fetch_klines_retries_on_rate_limit():
    from training.download_data import fetch_klines
    import requests

    with patch("requests.get") as mock_get:
        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = requests.HTTPError("429")
        ok_resp = MagicMock()
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = []
        mock_get.side_effect = [err_resp, ok_resp]
        df = fetch_klines("BTCUSDT", "15m", start_ms=0, end_ms=1000)
    assert isinstance(df, pd.DataFrame)
