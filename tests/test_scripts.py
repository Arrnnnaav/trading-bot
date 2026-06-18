import pandas as pd
from unittest.mock import patch


def _fake_yf_download(ticker, start, end, auto_adjust, progress):
    import pandas as pd

    idx = pd.date_range("2000-01-03", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [105.0] * 5,
            "Low": [98.0] * 5,
            "Close": [102.0] * 5,
            "Volume": [1_000_000] * 5,
        },
        index=idx,
    )


def test_fetch_historical_creates_parquet_files(tmp_path):
    with patch("yfinance.download", side_effect=_fake_yf_download):
        from scripts.fetch_historical import fetch_all

        fetch_all(data_dir=tmp_path, start_date="2000-01-01", end_date="2000-01-10")

    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 4
    names = {f.stem for f in files}
    assert names == {"NSEI", "NSEBANK", "BSESN", "CNXIT"}


def test_fetch_historical_parquet_has_expected_columns(tmp_path):
    with patch("yfinance.download", side_effect=_fake_yf_download):
        from scripts.fetch_historical import fetch_all

        fetch_all(data_dir=tmp_path, start_date="2000-01-01", end_date="2000-01-10")

    df = pd.read_parquet(tmp_path / "NSEI.parquet")
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
    assert len(df) == 5


def test_fetch_historical_incremental_append(tmp_path):
    with patch("yfinance.download", side_effect=_fake_yf_download):
        from scripts.fetch_historical import fetch_all

        fetch_all(data_dir=tmp_path, start_date="2000-01-01", end_date="2000-01-10")
        fetch_all(data_dir=tmp_path, start_date="2000-01-01", end_date="2000-01-10")

    df = pd.read_parquet(tmp_path / "NSEI.parquet")
    # Incremental append deduplicates by date — should not double rows
    assert len(df) == 5


def test_fetch_fii_dii_creates_json(tmp_path):
    from datetime import date
    from unittest.mock import MagicMock
    import json

    mock_nsefin = MagicMock()
    mock_nsefin.nse.get_fii_dii_activity.return_value = [
        {
            "date": "2026-06-18",
            "fii_net_value": 2500.50,
            "dii_net_value": -800.25,
        }
    ]
    with patch.dict("sys.modules", {"nsefin": mock_nsefin}):
        from scripts.fetch_fii_dii import fetch_and_save

        result = fetch_and_save(output_dir=str(tmp_path), fetch_date=date(2026, 6, 18))

    assert result is not None and result.exists()
    data = json.loads(result.read_text())
    assert data["fii_net_cr"] == 2500.50
    assert data["dii_net_cr"] == -800.25
