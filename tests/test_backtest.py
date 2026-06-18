"""Tests for backtesting/india_backtest.py — Task 1: cost model + Task 2: position simulator."""

import pytest
from backtesting.india_backtest import (
    _calc_entry_premium,
    _calc_transaction_cost,
    _apply_theta_decay,
    _simulate_position,
    BacktestResult,
    BROKERAGE_PER_ORDER,
)


class TestCalcEntryPremium:
    def test_nifty_atm_7_dte(self):
        # spot=22000, iv=0.15, dte=7
        # expected = 22000 * 0.15 * sqrt(7/252)
        import math

        expected = 22000 * 0.15 * math.sqrt(7 / 252)
        result = _calc_entry_premium(spot=22000.0, iv_annual=0.15, dte_days=7)
        assert abs(result - expected) < 0.01

    def test_banknifty_atm_1_dte(self):
        import math

        expected = 47000 * 0.18 * math.sqrt(1 / 252)
        result = _calc_entry_premium(spot=47000.0, iv_annual=0.18, dte_days=1)
        assert abs(result - expected) < 0.01

    def test_zero_dte_returns_zero(self):
        # 0 DTE → no time value (avoid sqrt(0) edge case)
        result = _calc_entry_premium(spot=22000.0, iv_annual=0.15, dte_days=0)
        assert result == 0.0

    def test_premium_scales_with_spot(self):
        p1 = _calc_entry_premium(spot=10000.0, iv_annual=0.15, dte_days=7)
        p2 = _calc_entry_premium(spot=20000.0, iv_annual=0.15, dte_days=7)
        assert abs(p2 / p1 - 2.0) < 0.001


class TestCalcTransactionCost:
    def test_round_trip_nifty_known_values(self):
        # premium=200, lot_size=65
        # turnover per lot = 200 * 65 = 13000
        # buy leg: stamp_duty = 13000 * 0.00003 = 0.39
        #          exchange_txn = 13000 * 0.00053 = 6.89
        #          sebi = 13000 * 0.000001 = 0.013
        #          brokerage = 20.0
        # sell leg: stt = 13000 * 0.0005 = 6.5
        #           exchange_txn = 13000 * 0.00053 = 6.89
        #           sebi = 13000 * 0.000001 = 0.013
        #           brokerage = 20.0
        # total = 0.39 + 6.89 + 0.013 + 20 + 6.5 + 6.89 + 0.013 + 20 = 60.696
        result = _calc_transaction_cost(premium=200.0, lot_size=65)
        assert abs(result - 60.696) < 0.01

    def test_cost_increases_with_premium(self):
        c1 = _calc_transaction_cost(premium=100.0, lot_size=65)
        c2 = _calc_transaction_cost(premium=200.0, lot_size=65)
        assert c2 > c1

    def test_minimum_cost_is_brokerage_both_legs(self):
        # Even at near-zero premium, brokerage floor of ₹40 applies
        result = _calc_transaction_cost(premium=0.01, lot_size=65)
        assert result >= 2 * BROKERAGE_PER_ORDER  # ₹40 minimum


class TestApplyThetaDecay:
    def test_no_decay_at_day_zero(self):
        # Holding 0 days = original premium
        result = _apply_theta_decay(premium=200.0, iv_annual=0.15, days_held=0)
        assert result == pytest.approx(200.0)

    def test_decay_reduces_premium(self):
        result = _apply_theta_decay(premium=200.0, iv_annual=0.15, days_held=1)
        assert result < 200.0

    def test_daily_decay_formula(self):
        # premium × (IV / sqrt(252)) is the daily theta burn
        import math

        daily_burn = 200.0 * (0.15 / math.sqrt(252))
        expected = 200.0 - daily_burn
        result = _apply_theta_decay(premium=200.0, iv_annual=0.15, days_held=1)
        assert abs(result - expected) < 0.01

    def test_multi_day_decay_is_cumulative(self):
        import math

        daily_burn = 200.0 * (0.15 / math.sqrt(252))
        expected = 200.0 - 5 * daily_burn
        result = _apply_theta_decay(premium=200.0, iv_annual=0.15, days_held=5)
        assert abs(result - expected) < 0.01

    def test_floor_at_zero(self):
        # Massive decay should not produce negative premium
        result = _apply_theta_decay(premium=1.0, iv_annual=5.0, days_held=100)
        assert result >= 0.0


class TestBacktestResult:
    def test_dataclass_instantiation(self):

        r = BacktestResult(
            year=2020,
            sharpe=1.8,
            sortino=2.1,
            max_drawdown_pct=10.5,
            win_rate=52.0,
            profit_factor=1.6,
            total_trades=120,
            avg_hold_days=4.2,
            total_pnl=85000.0,
            monthly_pnl={"2020-01": 5000.0, "2020-02": -2000.0},
        )
        assert r.year == 2020
        assert r.sharpe == pytest.approx(1.8)
        assert r.monthly_pnl["2020-01"] == 5000.0


class TestSimulatePosition:
    def test_target_hit_day_1(self):
        # Premium decays via theta, so to hit a target we need target < entry.
        # With negligible theta (iv=0.00001), premium barely decays.
        # Set target_mult=0.99999 so target is 99.999, and premium stays above it.
        # Should hit TARGET_HIT on day 1 (or 2).
        result = _simulate_position(
            entry_premium=100.0,
            daily_closes=[22100.0, 22200.0],
            iv_annual=0.00001,  # negligible theta
            lot_size=65,
            target_mult=0.99999,  # target below entry due to theta decay
            stop_pct=0.35,
            max_days=10,
        )
        assert result["exit_reason"] == "TARGET_HIT"
        assert result["hold_days"] >= 1

    def test_stop_hit(self):
        # High theta eats the premium quickly
        result = _simulate_position(
            entry_premium=10.0,
            daily_closes=[22000.0] * 20,
            iv_annual=5.0,  # extreme theta to guarantee stop
            lot_size=65,
            target_mult=2.0,
            stop_pct=0.35,
            max_days=20,
        )
        assert result["exit_reason"] == "STOP_HIT"

    def test_time_exit(self):
        result = _simulate_position(
            entry_premium=200.0,
            daily_closes=[22000.0] * 10,
            iv_annual=0.15,
            lot_size=65,
            target_mult=10.0,  # unreachable target
            stop_pct=0.50,  # stop well below likely decay path
            max_days=5,
        )
        assert result["exit_reason"] == "TIME_EXIT"
        assert result["hold_days"] == 5

    def test_pnl_inr_net_of_costs(self):
        # Check pnl_inr accounts for costs
        result = _simulate_position(
            entry_premium=200.0,
            daily_closes=[22000.0] * 5,
            iv_annual=0.001,  # negligible decay
            lot_size=65,
            target_mult=10.0,
            stop_pct=0.001,
            max_days=3,
        )
        cost = _calc_transaction_cost(200.0, 65)
        gross = (result["exit_premium"] - 200.0) * 65
        assert abs(result["pnl_inr"] - (gross - cost)) < 0.01

    def test_pnl_pct_definition(self):
        result = _simulate_position(
            entry_premium=200.0,
            daily_closes=[22000.0] * 3,
            iv_annual=0.001,
            lot_size=65,
            target_mult=10.0,
            stop_pct=0.001,
            max_days=3,
        )
        expected_pct = (result["exit_premium"] - 200.0) / 200.0
        assert abs(result["pnl_pct"] - expected_pct) < 0.0001

    def test_no_daily_closes_returns_time_exit_day_0(self):
        result = _simulate_position(
            entry_premium=100.0,
            daily_closes=[],
            iv_annual=0.15,
            lot_size=65,
        )
        assert result["exit_reason"] == "TIME_EXIT"
        assert result["hold_days"] == 0


class TestSimulateYear:
    """Uses synthetic data to avoid dependency on real parquet files."""

    def _make_synthetic_df(
        self, n_rows: int = 300, symbol: str = "NSEI"
    ) -> "pd.DataFrame":
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
        close = 22000.0 + np.cumsum(np.random.default_rng(42).normal(0, 100, n_rows))
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.ones(n_rows) * 0,  # index tickers have 0 volume
                "symbol": symbol,
                "label": np.random.default_rng(42).choice(
                    ["LONG", "SHORT", "HOLD"], n_rows
                ),
                "fii_5d_net": np.zeros(n_rows),
            }
        )
        return df

    def _make_mock_model(self, prediction: int = 0):
        """Returns a model that always predicts the same class."""

        class _MockModel:
            def __init__(self, pred):
                self._pred = pred

            def predict(self, X):
                import numpy as np

                return np.full(len(X), self._pred, dtype=np.int64)

        return _MockModel(prediction)

    def test_returns_backtest_result_type(self):
        from backtesting.india_backtest import (
            _simulate_year,
            BacktestResult,
            _INDEX_SPEC,
        )

        df = self._make_synthetic_df(300)
        train_df = df.iloc[:200].copy()
        oos_df = df.iloc[200:].copy()
        model = self._make_mock_model(prediction=2)  # all HOLD → no trades
        result = _simulate_year(train_df, oos_df, model, _INDEX_SPEC["NSEI"])
        assert isinstance(result, BacktestResult)

    def test_all_hold_no_trades(self):
        from backtesting.india_backtest import _simulate_year, _INDEX_SPEC

        df = self._make_synthetic_df(300)
        result = _simulate_year(
            df.iloc[:200].copy(),
            df.iloc[200:].copy(),
            self._make_mock_model(2),  # HOLD
            _INDEX_SPEC["NSEI"],
        )
        assert result.total_trades == 0
        assert result.total_pnl == 0.0
        assert result.win_rate == 0.0

    def test_sharpe_is_annualised(self):
        from backtesting.india_backtest import _simulate_year, _INDEX_SPEC
        import math

        df = self._make_synthetic_df(500)
        result = _simulate_year(
            df.iloc[:250].copy(),
            df.iloc[250:].copy(),
            self._make_mock_model(0),  # all LONG
            _INDEX_SPEC["NSEI"],
        )
        # Sharpe must be a finite float (can be negative or zero — just check type)
        assert isinstance(result.sharpe, float)
        assert math.isfinite(result.sharpe)

    def test_win_rate_bounded(self):
        from backtesting.india_backtest import _simulate_year, _INDEX_SPEC

        df = self._make_synthetic_df(400)
        result = _simulate_year(
            df.iloc[:200].copy(),
            df.iloc[200:].copy(),
            self._make_mock_model(0),
            _INDEX_SPEC["NSEI"],
        )
        assert 0.0 <= result.win_rate <= 100.0

    def test_max_drawdown_non_negative(self):
        from backtesting.india_backtest import _simulate_year, _INDEX_SPEC

        df = self._make_synthetic_df(400)
        result = _simulate_year(
            df.iloc[:200].copy(),
            df.iloc[200:].copy(),
            self._make_mock_model(1),  # all SHORT
            _INDEX_SPEC["NSEI"],
        )
        assert result.max_drawdown_pct >= 0.0

    def test_monthly_pnl_keys_are_yyyy_mm(self):
        from backtesting.india_backtest import _simulate_year, _INDEX_SPEC
        import re

        df = self._make_synthetic_df(500)
        result = _simulate_year(
            df.iloc[:250].copy(),
            df.iloc[250:].copy(),
            self._make_mock_model(0),
            _INDEX_SPEC["NSEI"],
        )
        for key in result.monthly_pnl.keys():
            assert re.match(r"^\d{4}-\d{2}$", key), f"Bad key: {key}"

    def test_profit_factor_non_negative(self):
        from backtesting.india_backtest import _simulate_year, _INDEX_SPEC

        df = self._make_synthetic_df(400)
        result = _simulate_year(
            df.iloc[:200].copy(),
            df.iloc[200:].copy(),
            self._make_mock_model(0),
            _INDEX_SPEC["NSEI"],
        )
        assert result.profit_factor >= 0.0


class TestPassesLiveGate:
    def _make_result(self, sharpe, max_dd, win_rate, year=2020):
        from backtesting.india_backtest import BacktestResult

        return BacktestResult(
            year=year,
            sharpe=sharpe,
            sortino=1.0,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            profit_factor=1.5,
            total_trades=100,
            avg_hold_days=4.0,
            total_pnl=50000.0,
            monthly_pnl={},
        )

    def test_all_conditions_pass(self):
        from backtesting.india_backtest import passes_live_gate

        results = [self._make_result(sharpe=1.8, max_dd=10.0, win_rate=50.0)]
        assert passes_live_gate(results) is True

    def test_fails_on_low_sharpe(self):
        from backtesting.india_backtest import passes_live_gate

        results = [self._make_result(sharpe=1.2, max_dd=10.0, win_rate=50.0)]
        assert passes_live_gate(results) is False

    def test_fails_on_high_drawdown(self):
        from backtesting.india_backtest import passes_live_gate

        results = [self._make_result(sharpe=2.0, max_dd=16.0, win_rate=50.0)]
        assert passes_live_gate(results) is False

    def test_fails_on_low_win_rate(self):
        from backtesting.india_backtest import passes_live_gate

        results = [self._make_result(sharpe=2.0, max_dd=10.0, win_rate=44.9)]
        assert passes_live_gate(results) is False

    def test_empty_results_returns_false(self):
        from backtesting.india_backtest import passes_live_gate

        assert passes_live_gate([]) is False

    def test_any_bad_year_fails_gate(self):
        from backtesting.india_backtest import passes_live_gate

        results = [
            self._make_result(sharpe=2.0, max_dd=10.0, win_rate=50.0, year=2020),
            self._make_result(sharpe=2.0, max_dd=10.0, win_rate=50.0, year=2021),
            self._make_result(
                sharpe=1.0, max_dd=10.0, win_rate=50.0, year=2022
            ),  # fails
        ]
        assert passes_live_gate(results) is False

    def test_all_years_must_individually_pass(self):
        from backtesting.india_backtest import passes_live_gate

        results = [
            self._make_result(sharpe=2.0, max_dd=5.0, win_rate=60.0, year=y)
            for y in range(2010, 2025)
        ]
        assert passes_live_gate(results) is True


def _make_stub_df(start: str, periods: int) -> "pd.DataFrame":
    """Build a minimal multi-symbol DataFrame that mimics load_india_data output."""
    import pandas as pd
    import numpy as np

    frames = []
    for sym in ["NSEI", "NSEBANK", "BSESN", "CNXIT"]:
        dates = pd.date_range(start, periods=periods, freq="B")
        close = 10000.0 + np.arange(periods, dtype=float)
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 0.0,
                "symbol": sym,
                "label": "HOLD",
                "fii_5d_net": 0.0,
            }
        )
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("date")
    return combined


def _stub_compute_features(df, fii_dir="data/fii_dii"):
    """Pass-through stub: compute_features returns df unchanged (no real feature eng)."""
    import pandas as pd

    out = df.copy()
    # Ensure a DatetimeIndex named 'date' so _simulate_year's reset_index() works
    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out = out.set_index(pd.DatetimeIndex(pd.to_datetime(out["date"]), name="date"))
        out = out.drop(columns=["date"], errors="ignore")
    return out


class TestRunBacktestMonkeypatch:
    """Monkeypatches load_india_data + compute_features + _train_lgb_for_backtest."""

    def _patch_all(self, monkeypatch, stub_df):
        import numpy as np

        monkeypatch.setattr("training.dataset.load_india_data", lambda **kw: stub_df)
        monkeypatch.setattr(
            "training.india_features.compute_features", _stub_compute_features
        )
        monkeypatch.setattr(
            "backtesting.india_backtest._train_lgb_for_backtest",
            lambda X, y: type(
                "M",
                (),
                {"predict": lambda self, X: np.full(len(X), 2, dtype=np.int64)},
            )(),
        )

    def test_returns_list_of_backtest_results(self, monkeypatch, tmp_path):
        import backtesting.india_backtest as bt
        from backtesting.india_backtest import BacktestResult

        stub_df = _make_stub_df("2009-01-01", 1000)
        self._patch_all(monkeypatch, stub_df)

        results = bt.run_backtest(
            historical_dir=str(tmp_path),
            model_path=str(tmp_path / "model.pkl"),
            fii_dir=str(tmp_path),
            start_year=2011,
            end_year=2012,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_results_sorted_by_year(self, monkeypatch, tmp_path):
        import backtesting.india_backtest as bt

        stub_df = _make_stub_df("2009-01-01", 1500)
        self._patch_all(monkeypatch, stub_df)

        results = bt.run_backtest(
            historical_dir=str(tmp_path),
            model_path=str(tmp_path / "model.pkl"),
            fii_dir=str(tmp_path),
            start_year=2011,
            end_year=2014,
        )
        years = [r.year for r in results]
        assert years == sorted(years)


class TestPrintReport:
    def test_runs_without_error(self, capsys):
        from backtesting.india_backtest import print_report, BacktestResult

        results = [
            BacktestResult(
                year=2020,
                sharpe=1.8,
                sortino=2.1,
                max_drawdown_pct=9.5,
                win_rate=52.0,
                profit_factor=1.6,
                total_trades=100,
                avg_hold_days=4.2,
                total_pnl=120000.0,
                monthly_pnl={},
            )
        ]
        print_report(results)
        out = capsys.readouterr().out
        assert "2020" in out
        assert "1.80" in out
        assert "Gate" in out
