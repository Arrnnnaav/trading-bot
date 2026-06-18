"""
Walk-forward backtest for India index options strategy.

Uses LightGBM model predictions (not full 6-agent debate) for speed across
25 years of NSE daily data.  Accurate India transaction cost model and
Black-Scholes theta approximation are applied per simulated trade.

Gate for live trading: Sharpe >1.5, max drawdown <15%, win rate >45%
across ALL OOS years (2010–present).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Transaction cost constants (exact, not approximated)
# ---------------------------------------------------------------------------

STT_SELL_RATE = 0.0005  # 0.05% on sell-side premium only
BROKERAGE_PER_ORDER = 20.0  # ₹20 flat per order (entry + exit = ₹40)
SEBI_CHARGE_RATE = 0.000001  # 0.0001% of turnover (both sides)
STAMP_DUTY_BUY_RATE = 0.00003  # 0.003% on buy-side premium only
EXCHANGE_TXN_RATE = 0.00053  # 0.053% of premium (both sides)

# ---------------------------------------------------------------------------
# Options model constants
# ---------------------------------------------------------------------------

NIFTY_IV_ANNUAL = 0.15
BANKNIFTY_IV_ANNUAL = 0.18
SENSEX_IV_ANNUAL = 0.15
NIFTYIT_IV_ANNUAL = 0.18

# ---------------------------------------------------------------------------
# Position management rules
# ---------------------------------------------------------------------------

STOP_LOSS_PCT = 0.35  # exit if premium drops 35% from entry
TARGET_MULT = 2.0  # exit if premium reaches 2× entry
MAX_HOLD_DAYS = 10  # positional; intraday = 1

# ---------------------------------------------------------------------------
# Live gate thresholds
# ---------------------------------------------------------------------------

GATE_SHARPE = 1.5
GATE_MAX_DD_PCT = 15.0
GATE_WIN_RATE_PCT = 45.0

# ---------------------------------------------------------------------------
# Index spec (mirrors brokers/upstox.py _INDEX_SPEC)
# ---------------------------------------------------------------------------

_INDEX_SPEC: dict[str, dict[str, Any]] = {
    "NSEI": {"lot_size": 65, "iv_annual": NIFTY_IV_ANNUAL, "label": "Nifty 50"},
    "NSEBANK": {
        "lot_size": 30,
        "iv_annual": BANKNIFTY_IV_ANNUAL,
        "label": "Bank Nifty",
    },
    "BSESN": {"lot_size": 20, "iv_annual": SENSEX_IV_ANNUAL, "label": "Sensex"},
    "CNXIT": {"lot_size": 35, "iv_annual": NIFTYIT_IV_ANNUAL, "label": "Nifty IT"},
}

_LABEL_MAP = {"LONG": 0, "SHORT": 1, "HOLD": 2}
_FEATURE_EXCLUDE = frozenset(
    {"label", "date", "symbol", "open_time", "open", "high", "low", "close", "volume"}
)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    year: int
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate: float  # percentage (0–100)
    profit_factor: float
    total_trades: int
    avg_hold_days: float
    total_pnl: float  # ₹ net of all costs
    monthly_pnl: dict  # {"YYYY-MM": float}


# ---------------------------------------------------------------------------
# Cost model helpers
# ---------------------------------------------------------------------------


def _calc_entry_premium(spot: float, iv_annual: float, dte_days: int) -> float:
    """
    Estimate ATM call/put premium via Black-Scholes simplified formula.

    premium ≈ spot × iv_annual × sqrt(dte_days / 252)

    Returns 0.0 for dte_days <= 0.
    """
    if dte_days <= 0:
        return 0.0
    return spot * iv_annual * math.sqrt(dte_days / 252.0)


def _calc_transaction_cost(premium: float, lot_size: int) -> float:
    """
    Round-trip transaction cost for one lot of options.

    Charges applied:
      Buy leg  : stamp duty + exchange txn charge + SEBI + brokerage
      Sell leg : STT + exchange txn charge + SEBI + brokerage

    Returns total ₹ cost for the complete round trip (entry + exit).
    """
    turnover = premium * lot_size  # per-lot notional

    # Buy leg
    stamp_duty = turnover * STAMP_DUTY_BUY_RATE
    exchange_buy = turnover * EXCHANGE_TXN_RATE
    sebi_buy = turnover * SEBI_CHARGE_RATE
    buy_cost = stamp_duty + exchange_buy + sebi_buy + BROKERAGE_PER_ORDER

    # Sell leg
    stt_sell = turnover * STT_SELL_RATE
    exchange_sell = turnover * EXCHANGE_TXN_RATE
    sebi_sell = turnover * SEBI_CHARGE_RATE
    sell_cost = stt_sell + exchange_sell + sebi_sell + BROKERAGE_PER_ORDER

    return buy_cost + sell_cost


def _apply_theta_decay(premium: float, iv_annual: float, days_held: int) -> float:
    """
    Reduce option premium by linear theta approximation.

    Daily theta burn ≈ premium × (iv_annual / sqrt(252))

    Applied cumulatively for `days_held` days; floored at 0.
    """
    if days_held <= 0:
        return premium
    daily_burn = premium * (iv_annual / math.sqrt(252.0))
    decayed = premium - daily_burn * days_held
    return max(0.0, decayed)


def _simulate_position(
    entry_premium: float,
    daily_closes: list[float],
    iv_annual: float,
    lot_size: int,
    target_mult: float = TARGET_MULT,
    stop_pct: float = STOP_LOSS_PCT,
    max_days: int = MAX_HOLD_DAYS,
) -> dict:
    """
    Simulate one options position from entry to exit.

    Steps through daily_closes applying cumulative theta decay each day.
    Checks stop and target against the decayed premium.

    Returns dict:
      exit_reason  : "TARGET_HIT" | "STOP_HIT" | "TIME_EXIT"
      hold_days    : int
      exit_premium : float
      pnl_pct      : float  — (exit_premium - entry_premium) / entry_premium
      pnl_inr      : float  — gross P&L in ₹ net of round-trip transaction cost
    """
    stop_level = entry_premium * (1.0 - stop_pct)
    target_level = entry_premium * target_mult
    cap = min(max_days, len(daily_closes))

    exit_reason = "TIME_EXIT"
    hold_days = cap

    for day in range(1, cap + 1):
        current_premium = _apply_theta_decay(entry_premium, iv_annual, day)
        if current_premium <= stop_level:
            exit_reason = "STOP_HIT"
            hold_days = day
            break
        if current_premium >= target_level:
            exit_reason = "TARGET_HIT"
            hold_days = day
            break

    exit_premium = _apply_theta_decay(entry_premium, iv_annual, hold_days)
    txn_cost = _calc_transaction_cost(entry_premium, lot_size)
    gross_pnl_inr = (exit_premium - entry_premium) * lot_size
    pnl_inr = gross_pnl_inr - txn_cost
    pnl_pct = (
        (exit_premium - entry_premium) / entry_premium if entry_premium > 0 else 0.0
    )

    return {
        "exit_reason": exit_reason,
        "hold_days": hold_days,
        "exit_premium": exit_premium,
        "pnl_pct": pnl_pct,
        "pnl_inr": pnl_inr,
    }


# ---------------------------------------------------------------------------
# Statistical metric helpers
# ---------------------------------------------------------------------------


def _sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe. Returns 0.0 if std == 0 or len < 2."""
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std * math.sqrt(252))


def _sortino(returns: np.ndarray) -> float:
    """Annualised Sortino using downside deviation. Returns 0.0 if no downside."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return float(returns.mean() / downside_std * math.sqrt(252))


def _max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """
    Max peak-to-trough percentage decline in a cumulative P&L array (₹).

    Returns positive percentage (e.g. 12.5 means 12.5% drawdown).
    Returns 0.0 if array is empty or monotonically increasing.
    """
    if len(cumulative_pnl) == 0:
        return 0.0
    peak = cumulative_pnl[0]
    max_dd = 0.0
    for val in cumulative_pnl:
        if val > peak:
            peak = val
        if peak != 0:
            dd = (peak - val) / abs(peak) * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


# ---------------------------------------------------------------------------
# Year OOS simulator
# ---------------------------------------------------------------------------


def _simulate_year(
    train_df,
    oos_df,
    model,
    index_spec: dict,
    fii_dir: str = "data/fii_dii",
    max_hold_days: int = MAX_HOLD_DAYS,
    target_mult: float = TARGET_MULT,
    stop_pct: float = STOP_LOSS_PCT,
) -> BacktestResult:
    """
    Run one OOS year of options backtesting using model predictions.

    Steps:
      1. Compute features on oos_df via training.india_features.compute_features
      2. Predict LONG/SHORT/HOLD for each bar
      3. For LONG/SHORT: simulate options position starting next bar
      4. Aggregate BacktestResult with all metrics

    train_df is unused here (model already fitted by caller) but accepted
    for API consistency with run_backtest.
    """
    import numpy as np
    import pandas as pd
    from training.india_features import compute_features

    lot_size = index_spec["lot_size"]
    iv_annual = index_spec["iv_annual"]

    # Ensure oos_df has a DatetimeIndex named 'date' for compute_features.
    # Drop the 'date' column first so reset_index() won't collide with it later.
    oos_work = oos_df.copy()
    if "date" in oos_work.columns and not isinstance(oos_work.index, pd.DatetimeIndex):
        oos_work = oos_work.set_index(
            pd.DatetimeIndex(pd.to_datetime(oos_work["date"]), name="date")
        ).drop(columns=["date"], errors="ignore")

    # Compute features; if fii_5d_net column already present in input, _fii_5d_net
    # will overwrite it — that is fine.
    # reset_index() promotes the DatetimeIndex back to a 'date' column.
    feat_df = compute_features(oos_work, fii_dir=fii_dir).dropna().reset_index()

    feature_cols = [c for c in feat_df.columns if c not in _FEATURE_EXCLUDE]
    if len(feat_df) < 2 or not feature_cols:
        year_val = 0
        if "date" in oos_df.columns and len(oos_df) > 0:
            year_val = int(pd.to_datetime(oos_df["date"].iloc[0]).year)
        elif len(oos_df) > 0 and isinstance(oos_df.index, pd.DatetimeIndex):
            year_val = int(oos_df.index[0].year)
        return BacktestResult(
            year=year_val,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_hold_days=0.0,
            total_pnl=0.0,
            monthly_pnl={},
        )

    X = feat_df[feature_cols].to_numpy(dtype=np.float32)
    preds = model.predict(X)  # integer class array

    closes = feat_df["close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(feat_df["date"])

    year = int(dates.iloc[0].year)

    # DTE: intraday (max_hold_days==1) → 1 day, positional → 7 days
    dte_days = 7 if max_hold_days > 1 else 1

    trades: list = []
    monthly_buckets: dict = {}

    i = 0
    while i < len(preds) - 1:
        pred = int(preds[i])
        if pred == 2:  # HOLD
            i += 1
            continue

        spot = closes[i]
        entry_premium = _calc_entry_premium(spot, iv_annual, dte_days)
        if entry_premium <= 0.0:
            i += 1
            continue

        remaining_closes = closes[i + 1 :].tolist()

        trade = _simulate_position(
            entry_premium=entry_premium,
            daily_closes=remaining_closes,
            iv_annual=iv_annual,
            lot_size=lot_size,
            target_mult=target_mult,
            stop_pct=stop_pct,
            max_days=max_hold_days,
        )
        trade["entry_date"] = dates.iloc[i]
        trades.append(trade)

        month_key = dates.iloc[i].strftime("%Y-%m")
        monthly_buckets[month_key] = (
            monthly_buckets.get(month_key, 0.0) + trade["pnl_inr"]
        )

        i += max(1, trade["hold_days"])

    if not trades:
        return BacktestResult(
            year=year,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_hold_days=0.0,
            total_pnl=0.0,
            monthly_pnl={},
        )

    pnl_series = np.array([t["pnl_inr"] for t in trades], dtype=np.float64)
    cum_pnl = np.cumsum(pnl_series)

    initial_notional = _calc_entry_premium(closes[0], iv_annual, dte_days) * lot_size
    daily_ret = pnl_series / max(initial_notional, 1.0)

    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    win_rate = float(len(wins) / len(trades) * 100.0)
    profit_factor = (
        (float(wins.sum()) / abs(float(losses.sum()))) if len(losses) > 0 else 0.0
    )
    avg_hold = float(np.mean([t["hold_days"] for t in trades]))

    return BacktestResult(
        year=year,
        sharpe=_sharpe(daily_ret),
        sortino=_sortino(daily_ret),
        max_drawdown_pct=_max_drawdown(cum_pnl),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(trades),
        avg_hold_days=avg_hold,
        total_pnl=float(cum_pnl[-1]),
        monthly_pnl=monthly_buckets,
    )
