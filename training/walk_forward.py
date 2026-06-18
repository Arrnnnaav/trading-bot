"""Walk-forward OOS Sharpe validation for LightGBM India model."""

import numpy as np
import pandas as pd

from training.dataset import load_india_data

_LABEL_MAP = {"LONG": 0, "SHORT": 1, "HOLD": 2}
_DIR_SIGN = {0: 1.0, 1: -1.0, 2: 0.0}  # LONG=long, SHORT=short, HOLD=flat
_FEATURE_EXCLUDE = frozenset(
    {
        "label",
        "date",
        "symbol",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
)


def _sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe from a daily return array. Returns 0 if std==0."""
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(252))


def run_walk_forward(
    historical_dir: str = "data/historical",
    fii_dir: str = "data/fii_dii",
    start_oos_year: int = 2019,
    end_oos_year: int | None = None,
) -> dict[int, float]:
    """
    Expanding-window walk-forward validation for the LightGBM model.

    For each OOS year (start_oos_year … end_oos_year):
      1. Train LightGBM on all data before Jan 1 of that year
      2. Predict direction on that year's data
      3. Simulate: position sign × next-day return → daily P&L
      4. Return annualised Sharpe for that year

    Returns {year: sharpe} dict. Skips years with insufficient data.
    Prints progress per year.
    """
    from training.train_xgb_india import _train_lgb

    df = load_india_data(historical_dir=historical_dir, fii_dir=fii_dir)
    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if end_oos_year is None:
        end_oos_year = pd.Timestamp.now().year - 1

    feature_cols = [c for c in df.columns if c not in _FEATURE_EXCLUDE]
    results: dict[int, float] = {}

    for oos_year in range(start_oos_year, end_oos_year + 1):
        cutoff = pd.Timestamp(f"{oos_year}-01-01")
        next_year = pd.Timestamp(f"{oos_year + 1}-01-01")

        train_df = df[df["date"] < cutoff]
        oos_df = df[(df["date"] >= cutoff) & (df["date"] < next_year)]

        if len(train_df) < 500 or len(oos_df) < 20:
            print(f"OOS {oos_year}: skipped (train={len(train_df)}, oos={len(oos_df)})")
            continue

        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df["label"].map(_LABEL_MAP).to_numpy(dtype=np.int64)
        X_oos = oos_df[feature_cols].to_numpy(dtype=np.float32)

        model = _train_lgb(X_train, y_train, n_estimators=200)
        preds = model.predict(X_oos)  # integer class predictions

        # Simulate: sign × next-bar return
        close_oos = oos_df["close"].to_numpy(dtype=np.float64)
        next_ret = np.diff(close_oos) / np.where(
            close_oos[:-1] != 0, close_oos[:-1], 1.0
        )
        signs = np.array([_DIR_SIGN[int(p)] for p in preds[:-1]])
        daily_ret = signs * next_ret

        sharpe = _sharpe(daily_ret)
        results[oos_year] = sharpe
        print(f"OOS {oos_year}: Sharpe={sharpe:.3f}  ({len(oos_df)} bars)")

    return results


def passes_gate(results: dict[int, float], min_sharpe: float = 1.0) -> bool:
    """Returns True only if every OOS year has Sharpe >= min_sharpe."""
    if not results:
        return False
    return all(s >= min_sharpe for s in results.values())


if __name__ == "__main__":
    results = run_walk_forward()
    ok = passes_gate(results)
    print(f"\nGate {'PASSED ✅' if ok else 'FAILED ❌'}  (threshold Sharpe ≥ 1.0)")
    for year, sharpe in sorted(results.items()):
        marker = "✅" if sharpe >= 1.0 else "❌"
        print(f"  {year}: {sharpe:.3f}  {marker}")
