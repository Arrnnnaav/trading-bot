"""
XGBTechnicalAgent — LightGBM classifier on 27 technical indicators.

Trained on same OHLCV data as ChronosTechnicalAgent but with engineered features
(RSI, MACD, Bollinger, ATR, EMA/SMA ratios, volume, ADX, stochastic, etc.)
instead of raw close prices. Achieves macro_f1=0.4159 vs Chronos v2=0.3804.

Model: models/xgb_technical/lgb_model.txt (LightGBM booster)
Confidence = max class probability, only emits vote if >= 0.50.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

import lightgbm as lgb

from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

MODEL_PATH = Path("models/xgb_technical/lgb_model.txt")
METADATA_PATH = Path("models/xgb_technical/metadata.json")
CONFIDENCE_THRESHOLD = 0.50
LABEL_NAMES = ["LONG", "SHORT", "HOLD"]
DIRECTION_MAP = {
    "LONG": Direction.LONG,
    "SHORT": Direction.SHORT,
    "HOLD": Direction.HOLD,
}


def _klines_to_df(klines: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(klines)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _compute_features(df: pd.DataFrame) -> dict | None:
    """Compute all indicators; return last row as feature dict, or None if too short."""
    if len(df) < 55:
        return None

    # Keep runtime features aligned with the training pipeline.
    try:
        from training.xgb_features import _compute_features as _compute_feature_frame
    except ImportError:
        return None

    feature_frame = _compute_feature_frame(df.copy())
    if feature_frame.empty:
        return None
    last_row = feature_frame.iloc[-1]

    # Symbol one-hots are added by load_features() at training time, not by the
    # shared _compute_features — rebuild them here from the symbol column.
    raw = df.get("symbol")
    symbol = str(raw.iloc[-1]) if raw is not None else ""

    feats = {}
    for f in FEATURE_ORDER:
        if f.startswith("symbol_"):
            feats[f] = 1.0 if f == f"symbol_{symbol}" else 0.0
        else:
            feats[f] = float(last_row[f])

    if any(np.isnan(v) for v in feats.values()):
        return None
    return feats


FEATURE_ORDER = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "roc_10",
    "mom_10",
    "stoch_k",
    "stoch_d",
    "willr_14",
    "cci_20",
    "adx_14",
    "dmp_14",
    "dmn_14",
    "ema9_ratio",
    "ema21_ratio",
    "ema50_ratio",
    "sma20_ratio",
    "sma50_ratio",
    "atr_ratio",
    "bb_pct",
    "bb_bw",
    "vol_ratio",
    "obv_norm",
    "body_ratio",
    "upper_wick",
    "lower_wick",
    "hl_ratio",
    "ret_1",
    "ret_4",
    "ret_16",
    "ret_48",
    "realized_vol_4",
    "realized_vol_16",
    "realized_vol_48",
    "range_atr_ratio",
    "trend_strength",
    "ema9_ema21_spread",
    "ema21_ema50_spread",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "symbol_BTCUSDT",
    "symbol_ETHUSDT",
    "symbol_SOLUSDT",
    "symbol_BNBUSDT",
    "symbol_XRPUSDT",
]


class XGBTechnicalAgent(BaseAgent):
    name = "XGBTechnical"
    _model: lgb.Booster | None = None

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"XGB model not found at {MODEL_PATH}. Run: python -m training.train_xgb"
            )
        self._model = lgb.Booster(model_file=str(MODEL_PATH))
        self._thresholds = self._load_thresholds()

    @staticmethod
    def _load_thresholds() -> dict[str, float]:
        try:
            metadata = json.loads(METADATA_PATH.read_text())
            return metadata.get("class_thresholds", {})
        except Exception:
            return {}

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        df = _klines_to_df(klines)
        df["symbol"] = ticker.replace("/", "")
        feats = _compute_features(df)

        if feats is None:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning="Insufficient data for technical indicators",
            )

        X = np.array([[feats[f] for f in FEATURE_ORDER]], dtype=np.float32)
        probs = self._model.predict(X)[0]  # [p_long, p_short, p_hold]

        label_idx = int(np.argmax(probs))
        confidence = float(probs[label_idx])
        label = LABEL_NAMES[label_idx]
        direction = DIRECTION_MAP[label]
        threshold = getattr(self, "_thresholds", {}).get(label, CONFIDENCE_THRESHOLD)

        if direction == Direction.HOLD or confidence < threshold:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"XGB: {label} @ {confidence:.2f} — below threshold {threshold:.2f}",
            )

        reasoning = (
            f"XGB(lgbm): {label} conf={confidence:.2f} | "
            f"RSI={feats['rsi_14']:.1f} MACD_hist={feats['macd_hist']:.4f} "
            f"ADX={feats['adx_14']:.1f} BB%={feats['bb_pct']:.2f}"
        )
        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=round(min(confidence, 0.90), 3),
            reasoning=reasoning,
        )
