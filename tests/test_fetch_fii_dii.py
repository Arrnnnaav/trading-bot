"""Tests for scripts/fetch_fii_dii.py"""

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch


from scripts.fetch_fii_dii import fetch_and_save


def _mock_nsefin_data():
    return [
        {"date": "2026-06-18", "fii_net_value": 1234.5, "dii_net_value": -567.8},
        {"date": "2026-06-17", "fii_net_value": 900.0, "dii_net_value": -300.0},
    ]


def test_creates_json_file_with_correct_format():
    with tempfile.TemporaryDirectory() as tmp:
        mock_nsefin = MagicMock()
        mock_nsefin.get_fii_dii_activity.return_value = _mock_nsefin_data()
        with patch.dict("sys.modules", {"nsefin": mock_nsefin}):
            result = fetch_and_save(output_dir=tmp, fetch_date=date(2026, 6, 18))
        assert result is not None
        assert result.exists()
        data = json.loads(result.read_text())
        assert data["date"] == "2026-06-18"
        assert data["fii_net_cr"] == 1234.5
        assert data["dii_net_cr"] == -567.8


def test_returns_none_on_nsefin_exception():
    with tempfile.TemporaryDirectory() as tmp:
        mock_nsefin = MagicMock()
        mock_nsefin.get_fii_dii_activity.side_effect = RuntimeError("network error")
        with patch.dict("sys.modules", {"nsefin": mock_nsefin}):
            result = fetch_and_save(output_dir=tmp, fetch_date=date(2026, 6, 18))
        assert result is None


def test_returns_none_on_empty_data():
    with tempfile.TemporaryDirectory() as tmp:
        mock_nsefin = MagicMock()
        mock_nsefin.get_fii_dii_activity.return_value = []
        with patch.dict("sys.modules", {"nsefin": mock_nsefin}):
            result = fetch_and_save(output_dir=tmp, fetch_date=date(2026, 6, 18))
        assert result is None


def test_skips_if_file_already_exists():
    with tempfile.TemporaryDirectory() as tmp:
        existing = Path(tmp) / "2026-06-18.json"
        existing.write_text(
            json.dumps({"date": "2026-06-18", "fii_net_cr": 999.0, "dii_net_cr": 0.0})
        )
        mock_nsefin = MagicMock()
        with patch.dict("sys.modules", {"nsefin": mock_nsefin}):
            result = fetch_and_save(output_dir=tmp, fetch_date=date(2026, 6, 18))
        # Should return existing path without calling nsefin
        assert result == existing
        mock_nsefin.get_fii_dii_activity.assert_not_called()


def test_creates_output_dir_if_missing():
    with tempfile.TemporaryDirectory() as tmp:
        nested = str(Path(tmp) / "deep" / "fii_dii")
        mock_nsefin = MagicMock()
        mock_nsefin.get_fii_dii_activity.return_value = _mock_nsefin_data()
        with patch.dict("sys.modules", {"nsefin": mock_nsefin}):
            result = fetch_and_save(output_dir=nested, fetch_date=date(2026, 6, 18))
        assert result is not None and result.exists()
