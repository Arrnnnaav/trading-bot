import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from training.dataset import WINDOW

RESULTS_DIR = Path("backtesting/results")
ATR_PERIOD = 14
STOP_MULT = 2.0  # 2×ATR stop (matches RiskManager crypto config)
TARGET_MULT = 1.8  # 1.8 R:R target


def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> str:
    """Walk forward from signal_idx, return TARGET_HIT / STOP_HIT / EXPIRED."""
    look_ahead = df.iloc[signal_idx + 1 : signal_idx + 97]  # max 24 hours
    for _, row in look_ahead.iterrows():
        if direction == "LONG":
            if row["low"] <= stop_price:
                return "STOP_HIT"
            if row["high"] >= target_price:
                return "TARGET_HIT"
        else:  # SHORT
            if row["high"] >= stop_price:
                return "STOP_HIT"
            if row["low"] <= target_price:
                return "TARGET_HIT"
    return "EXPIRED"


def run_backtest(
    df: pd.DataFrame,
    model,
    window: int = WINDOW,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Walk-forward backtest of ChronosClassifier on a labeled DataFrame.
    Model runs standalone (no DebateEngine) to isolate Chronos-2 signal quality.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    atr = _compute_atr(df)

    signals = []
    equity = 1.0
    equity_curve = [1.0]
    closes = df["close"].to_numpy(dtype=np.float32)

    for i in tqdm(range(window, len(df) - 96), desc="backtest", unit="candle"):
        close_window = closes[i - window : i].copy()
        mean, std = close_window.mean(), close_window.std() + 1e-8
        normalized = (close_window - mean) / std
        tensor = torch.tensor(normalized, dtype=torch.float32).to(device)

        direction, confidence = model.predict(tensor)

        if direction.value == "HOLD":
            equity_curve.append(equity)
            continue

        entry = closes[i]
        atr_val = atr.iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            equity_curve.append(equity)
            continue

        if direction.value == "LONG":
            stop = entry - STOP_MULT * atr_val
            target = entry + TARGET_MULT * STOP_MULT * atr_val
        else:
            stop = entry + STOP_MULT * atr_val
            target = entry - TARGET_MULT * STOP_MULT * atr_val

        outcome = _simulate_trade(df, i, direction.value, entry, stop, target)

        risk_pct = abs(entry - stop) / entry
        if outcome == "TARGET_HIT":
            pnl_pct = risk_pct * TARGET_MULT
        elif outcome == "STOP_HIT":
            pnl_pct = -risk_pct
        else:
            pnl_pct = 0.0

        equity *= 1 + pnl_pct * 0.02  # 2% portfolio risk per trade
        equity_curve.append(equity)

        signals.append(
            {
                "candle_idx": i,
                "direction": direction.value,
                "confidence": round(confidence, 3),
                "entry": round(float(entry), 4),
                "stop": round(float(stop), 4),
                "target": round(float(target), 4),
                "outcome": outcome,
                "pnl_pct": round(pnl_pct, 4),
            }
        )

    # Compute metrics
    total = len(signals)
    wins = sum(1 for s in signals if s["outcome"] == "TARGET_HIT")
    losses = sum(1 for s in signals if s["outcome"] == "STOP_HIT")
    win_rate = wins / total if total else 0.0

    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / peak
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0

    returns = np.diff(eq) / eq[:-1]
    sharpe = (
        float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(24 * 365 / 0.25))
        if len(returns) > 1
        else 0.0
    )

    results = {
        "total_candles": len(df) - window,
        "total_signals": total,
        "signal_rate_pct": round(total / max(len(df) - window, 1) * 100, 2),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "final_equity": round(float(eq[-1]), 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "equity_curve": eq.tolist(),
        "signals": signals,
    }
    return results


def main():
    from training.model import ChronosClassifier
    from training.dataset import load_all_tickers, split_datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model = ChronosClassifier.load("models/chronos_crypto/best.pt").to(device)

    print("Loading test split...")
    df = load_all_tickers()
    _, _, test_ds = split_datasets(df)
    # Reconstruct raw test DataFrame (need OHLCV for trade simulation)
    n = len(df)
    test_start = int(n * 0.90)
    test_df = df.iloc[test_start:].reset_index(drop=True)

    print(f"Running backtest on {len(test_df):,} candles...")
    results = run_backtest(test_df, model, device=device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_json = RESULTS_DIR / f"{ts}-crypto-backtest.json"

    # Save (exclude equity_curve list from JSON to keep file small)
    save_results = {k: v for k, v in results.items() if k != "equity_curve"}
    out_json.write_text(json.dumps(save_results, indent=2))

    print(f"\n{'=' * 50}")
    print(
        f"Signals fired:    {results['total_signals']} ({results['signal_rate_pct']}% of candles)"
    )
    print(f"Win rate:         {results['win_rate']:.1%}")
    print(f"Final equity:     {results['final_equity']:.4f}x  (started 1.0)")
    print(f"Max drawdown:     {results['max_drawdown_pct']:.1f}%")
    print(f"Sharpe ratio:     {results['sharpe_ratio']:.3f}")
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
