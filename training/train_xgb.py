"""
train_xgb.py — train XGBoost + LightGBM on technical indicator features.
Compares both against Chronos v2 baseline (macro_f1=0.3804).

Run: python -m training.train_xgb
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

from training.xgb_features import load_features, get_splits, FEATURE_COLS

CHRONOS_V2_BASELINE = 0.3804
MODEL_DIR = Path("models/xgb_technical")
LABEL_NAMES = ["LONG", "SHORT", "HOLD"]


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(np.float64)
    w = counts.sum() / (3 * counts + 1e-8)
    w = np.clip(w, 1.0, None)
    return w


def _sample_weights(y: np.ndarray, class_w: np.ndarray) -> np.ndarray:
    return class_w[y]


def _eval(y_true, y_pred, name: str) -> dict:
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    dist = np.bincount(y_pred, minlength=3) / len(y_pred) * 100
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(f"\n{name}")
    print(
        f"  macro_f1={macro:.4f}  [LONG={per[0]:.3f} SHORT={per[1]:.3f} HOLD={per[2]:.3f}]"
    )
    print(f"  preds%: LONG={dist[0]:.1f} SHORT={dist[1]:.1f} HOLD={dist[2]:.1f}")
    print(
        f"  vs Chronos v2 baseline: {'+' if macro > CHRONOS_V2_BASELINE else ''}{macro - CHRONOS_V2_BASELINE:+.4f}"
    )
    return {
        "macro_f1": round(macro, 4),
        "long_f1": round(float(per[0]), 4),
        "short_f1": round(float(per[1]), 4),
        "hold_f1": round(float(per[2]), 4),
        "confusion": cm.tolist(),
    }


def train_xgboost(X_train, y_train, X_val, y_val, class_w):
    print("\n=== XGBoost ===")
    sw = _sample_weights(y_train, class_w)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        tree_method="hist",
        device="cuda",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sw,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, class_w):
    print("\n=== LightGBM ===")
    sw = _sample_weights(y_train, class_w)

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        early_stopping_rounds=30,
        device="gpu",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sw,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    return model


def main():
    print("Loading data + computing features...")
    df = load_features()
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(df)
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"Features: {len(FEATURE_COLS)}")

    class_w = _class_weights(y_train)
    print(
        f"Class weights: LONG={class_w[0]:.2f} SHORT={class_w[1]:.2f} HOLD={class_w[2]:.2f}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {"chronos_v2_baseline": CHRONOS_V2_BASELINE}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, class_w)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_test_pred = xgb_model.predict(X_test)
    _eval(y_val, xgb_val_pred, "XGBoost — Val")
    xgb_test = _eval(y_test, xgb_test_pred, "XGBoost — Test")
    results["xgboost"] = xgb_test
    xgb_model.save_model(str(MODEL_DIR / "xgb_model.json"))

    # feature importance
    imp = xgb_model.feature_importances_
    top = sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])[:10]
    print("\n  Top-10 features (XGBoost):")
    for fname, score in top:
        print(f"    {fname}: {score:.4f}")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    try:
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, class_w)
        lgb_val_pred = lgb_model.predict(X_val)
        lgb_test_pred = lgb_model.predict(X_test)
        _eval(y_val, lgb_val_pred, "LightGBM — Val")
        lgb_test = _eval(y_test, lgb_test_pred, "LightGBM — Test")
        results["lightgbm"] = lgb_test
        lgb_model.booster_.save_model(str(MODEL_DIR / "lgb_model.txt"))
    except Exception as e:
        print(f"\nLightGBM GPU failed ({e}), retrying CPU...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            early_stopping_rounds=30,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(
            X_train,
            y_train,
            sample_weight=_sample_weights(y_train, class_w),
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
        )
        lgb_test_pred = lgb_model.predict(X_test)
        lgb_test = _eval(y_test, lgb_test_pred, "LightGBM (CPU) — Test")
        results["lightgbm"] = lgb_test
        lgb_model.booster_.save_model(str(MODEL_DIR / "lgb_model.txt"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"CHRONOS v2 baseline:  macro_f1={CHRONOS_V2_BASELINE}")
    for mname in ["xgboost", "lightgbm"]:
        if mname in results:
            r = results[mname]
            verdict = "BEATS" if r["macro_f1"] > CHRONOS_V2_BASELINE else "LOSES TO"
            print(
                f"{mname.upper():12s}:  macro_f1={r['macro_f1']}  → {verdict} Chronos v2"
            )
    print(f"{'=' * 50}")

    (MODEL_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {MODEL_DIR}/results.json")


if __name__ == "__main__":
    main()
