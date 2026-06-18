"""Tests for backtesting/india_backtest.py — Task 1: cost model."""

import pytest
from backtesting.india_backtest import (
    _calc_entry_premium,
    _calc_transaction_cost,
    _apply_theta_decay,
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
