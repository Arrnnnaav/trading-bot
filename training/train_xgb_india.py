"""LightGBM training for India daily signals with Optuna HPO."""

import json
import joblib
import numpy as np
import lightgbm as lgb
import optuna
from pathlib import Path
from sklearn.metrics import f1_score

from training.dataset import load_india_data, split_india_datasets

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path("models/xgb_india")
N_TRIALS = 50


def _train_lgb(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    n_estimators: int = 200,
    verbose: bool = False,
    **params,
) -> lgb.LGBMClassifier:
    """Train one LightGBM model. Exported for walk_forward.py."""
    base = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "class_weight": "balanced",
        "n_estimators": n_estimators,
    }
    base.update(params)
    model = lgb.LGBMClassifier(**base)
    if X_val is not None and y_val is not None:
        model.fit(
            X,
            y,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
        )
    else:
        model.fit(X, y)
    return model


def _objective(trial, X_train, y_train, X_val, y_val) -> float:
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 127),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
    }
    model = _train_lgb(
        X_train, y_train, X_val, y_val, n_estimators=n_estimators, **params
    )
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average="macro", zero_division=0)


def train(
    historical_dir: str = "data/historical",
    fii_dir: str = "data/fii_dii",
    n_trials: int = N_TRIALS,
) -> tuple:
    """
    Full training pipeline:
      1. Load + label + feature-engineer all 4 NSE parquets
      2. Temporal 80/10/10 split
      3. Optuna HPO (n_trials trials)
      4. Retrain final model with best params on train+val combined
      5. Save model.pkl, feature_importance.json, metrics.json
    Returns: (fitted_model, test_macro_f1)
    """
    print("Loading data...")
    df = load_india_data(historical_dir=historical_dir, fii_dir=fii_dir)
    train_ds, val_ds, test_ds = split_india_datasets(df)

    X_train, y_train = train_ds.X, train_ds.y
    X_val, y_val = val_ds.X, val_ds.y
    X_test, y_test = test_ds.X, test_ds.y
    feature_names = train_ds.feature_names

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(
        f"Label dist (train): LONG={(y_train == 0).sum()} SHORT={(y_train == 1).sum()} HOLD={(y_train == 2).sum()}"
    )

    print(f"Running Optuna HPO ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _objective(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = study.best_params
    print(f"Best val macro-F1: {study.best_value:.4f}  params: {best_params}")

    # Final model: train on train+val combined with best params
    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    final_model = _train_lgb(X_full, y_full, **best_params)

    # Evaluate on held-out test set
    test_preds = final_model.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    print(f"Test macro-F1: {test_f1:.4f}")

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "model.pkl")

    importance = dict(zip(feature_names, final_model.feature_importances_.tolist()))
    (MODEL_DIR / "feature_importance.json").write_text(json.dumps(importance, indent=2))

    metrics = {
        "test_macro_f1": test_f1,
        "val_macro_f1": study.best_value,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "best_params": best_params,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {MODEL_DIR}/")

    return final_model, test_f1


if __name__ == "__main__":
    train()
