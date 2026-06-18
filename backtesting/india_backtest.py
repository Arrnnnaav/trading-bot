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


# ---------------------------------------------------------------------------
# Internal LightGBM training wrapper (importable for monkeypatching in tests)
# ---------------------------------------------------------------------------


def _train_lgb_for_backtest(X_train, y_train):
    """Thin wrapper around training.train_xgb_india._train_lgb for monkeypatching."""
    from training.train_xgb_india import _train_lgb

    return _train_lgb(X_train, y_train, n_estimators=200)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    historical_dir: str = "data/historical",
    model_path: str = "models/xgb_india/model.pkl",
    fii_dir: str = "data/fii_dii",
    start_year: int = 2010,
    end_year: int | None = None,
) -> list[BacktestResult]:
    """
    Expanding-window walk-forward backtest across all 4 India indices.

    For each OOS year, trains a fresh LightGBM on all prior data, then
    runs _simulate_year per index and combines results (trade-count-weighted
    averages for rate metrics; sums for P&L).
    """
    import numpy as np
    import pandas as pd
    from training.dataset import load_india_data
    from training.india_features import compute_features

    if end_year is None:
        end_year = pd.Timestamp.now().year - 1

    # Load all indices; dataset.py merges parquets with 'symbol' column
    df = load_india_data(historical_dir=historical_dir, fii_dir=fii_dir)
    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    combined_results: list[BacktestResult] = []

    for oos_year in range(start_year, end_year + 1):
        cutoff = pd.Timestamp(f"{oos_year}-01-01")
        next_year = pd.Timestamp(f"{oos_year + 1}-01-01")

        train_df = df[df["date"] < cutoff]
        oos_df = df[(df["date"] >= cutoff) & (df["date"] < next_year)]

        if len(train_df) < 500 or len(oos_df) < 20:
            print(
                f"[backtest] {oos_year}: skipped — insufficient data "
                f"(train={len(train_df)}, oos={len(oos_df)})"
            )
            continue

        # Compute features for training set; drop NaN warm-up rows
        train_feat = compute_features(train_df.copy(), fii_dir=fii_dir).dropna()
        feature_cols = [c for c in train_feat.columns if c not in _FEATURE_EXCLUDE]
        if not feature_cols:
            print(f"[backtest] {oos_year}: skipped — no feature columns")
            continue

        X_train = train_feat[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_feat["label"].map(_LABEL_MAP).fillna(2).to_numpy(dtype=np.int64)

        model = _train_lgb_for_backtest(X_train, y_train)

        # Run per-index simulation
        per_index_results: list[BacktestResult] = []
        symbols_in_oos = (
            oos_df["symbol"].unique()
            if "symbol" in oos_df.columns
            else list(_INDEX_SPEC.keys())
        )
        for sym in symbols_in_oos:
            if sym not in _INDEX_SPEC:
                continue
            spec = _INDEX_SPEC[sym]
            sym_train = (
                train_df[train_df["symbol"] == sym]
                if "symbol" in train_df.columns
                else train_df
            )
            sym_oos = (
                oos_df[oos_df["symbol"] == sym]
                if "symbol" in oos_df.columns
                else oos_df
            )
            if len(sym_oos) < 5:
                continue
            r = _simulate_year(sym_train, sym_oos, model, spec, fii_dir=fii_dir)
            per_index_results.append(r)

        if not per_index_results:
            continue

        # Combine per-index results into one year result
        total_trades = sum(r.total_trades for r in per_index_results)
        total_pnl = sum(r.total_pnl for r in per_index_results)

        def _weighted_avg(attr: str) -> float:
            if total_trades == 0:
                return 0.0
            return (
                sum(getattr(r, attr) * r.total_trades for r in per_index_results)
                / total_trades
            )

        combined_monthly: dict[str, float] = {}
        for r in per_index_results:
            for month, pnl in r.monthly_pnl.items():
                combined_monthly[month] = combined_monthly.get(month, 0.0) + pnl

        combined = BacktestResult(
            year=oos_year,
            sharpe=_weighted_avg("sharpe"),
            sortino=_weighted_avg("sortino"),
            max_drawdown_pct=max(r.max_drawdown_pct for r in per_index_results),
            win_rate=_weighted_avg("win_rate"),
            profit_factor=_weighted_avg("profit_factor"),
            total_trades=total_trades,
            avg_hold_days=_weighted_avg("avg_hold_days"),
            total_pnl=total_pnl,
            monthly_pnl=combined_monthly,
        )
        combined_results.append(combined)

        gate_sym = (
            "✅"
            if (
                combined.sharpe > GATE_SHARPE
                and combined.max_drawdown_pct < GATE_MAX_DD_PCT
                and combined.win_rate > GATE_WIN_RATE_PCT
            )
            else "❌"
        )
        print(
            f"[backtest] {oos_year}: Sharpe={combined.sharpe:.2f}  "
            f"MaxDD={combined.max_drawdown_pct:.1f}%  "
            f"WinRate={combined.win_rate:.1f}%  "
            f"Trades={combined.total_trades}  {gate_sym}"
        )

    return sorted(combined_results, key=lambda r: r.year)


def passes_live_gate(results: list[BacktestResult]) -> bool:
    """
    Returns True only if ALL conditions hold across ALL OOS years:
      sharpe > 1.5  AND  max_drawdown_pct < 15.0  AND  win_rate > 45.0

    Returns False if results is empty.
    """
    if not results:
        return False
    return all(
        r.sharpe > GATE_SHARPE
        and r.max_drawdown_pct < GATE_MAX_DD_PCT
        and r.win_rate > GATE_WIN_RATE_PCT
        for r in results
    )


def print_report(results: list[BacktestResult]) -> None:
    """Print formatted backtest summary table to stdout."""
    sep = "─" * 98
    header = (
        f"{'Year':>6} │ {'Sharpe':>7} │ {'Sortino':>8} │ {'MaxDD%':>7} │ "
        f"{'WinRate%':>9} │ {'ProfFact':>9} │ {'Trades':>7} │ "
        f"{'AvgHoldDy':>10} │ {'PnL ₹':>12}"
    )
    print(sep)
    print(header)
    print(sep)
    for r in results:
        pnl_lakh = r.total_pnl / 100_000
        gate_ok = (
            r.sharpe > GATE_SHARPE
            and r.max_drawdown_pct < GATE_MAX_DD_PCT
            and r.win_rate > GATE_WIN_RATE_PCT
        )
        marker = "✅" if gate_ok else "❌"
        print(
            f"{r.year:>6} │ {r.sharpe:>7.2f} │ {r.sortino:>8.2f} │ "
            f"{r.max_drawdown_pct:>6.1f}% │ {r.win_rate:>8.1f}% │ "
            f"{r.profit_factor:>9.2f} │ {r.total_trades:>7d} │ "
            f"{r.avg_hold_days:>9.1f}d │ {pnl_lakh:>10.2f}L  {marker}"
        )
    print(sep)
    gate_pass = passes_live_gate(results)
    gate_label = "PASSED ✅" if gate_pass else "FAILED ❌"
    print(
        f"Gate: {gate_label}  "
        f"(Sharpe>{GATE_SHARPE}, MaxDD<{GATE_MAX_DD_PCT}%, WinRate>{GATE_WIN_RATE_PCT}%)"
    )
    print(sep)
