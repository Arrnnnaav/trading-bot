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
