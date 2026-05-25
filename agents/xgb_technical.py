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
from pathlib import Path

import lightgbm as lgb
import pandas_ta as ta

from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

MODEL_PATH = Path("models/xgb_technical/lgb_model.txt")
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

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    feats = {}

    feats["rsi_14"] = ta.rsi(c, length=14).iloc[-1]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    last_c = c.iloc[-1]
    feats["macd_line"] = macd["MACD_12_26_9"].iloc[-1] / last_c
    feats["macd_signal"] = macd["MACDs_12_26_9"].iloc[-1] / last_c
    feats["macd_hist"] = macd["MACDh_12_26_9"].iloc[-1] / last_c

    feats["roc_10"] = ta.roc(c, length=10).iloc[-1]
    feats["mom_10"] = ta.mom(c, length=10).iloc[-1] / last_c

    stoch = ta.stoch(h, l, c, k=14, d=3)
    feats["stoch_k"] = stoch["STOCHk_14_3_3"].iloc[-1]
    feats["stoch_d"] = stoch["STOCHd_14_3_3"].iloc[-1]

    feats["willr_14"] = ta.willr(h, l, c, length=14).iloc[-1]
    feats["cci_20"] = ta.cci(h, l, c, length=20).iloc[-1]

    adx = ta.adx(h, l, c, length=14)
    feats["adx_14"] = adx["ADX_14"].iloc[-1]
    feats["dmp_14"] = adx["DMP_14"].iloc[-1]
    feats["dmn_14"] = adx["DMN_14"].iloc[-1]

    feats["ema9_ratio"] = ta.ema(c, length=9).iloc[-1] / last_c - 1
    feats["ema21_ratio"] = ta.ema(c, length=21).iloc[-1] / last_c - 1
    feats["ema50_ratio"] = ta.ema(c, length=50).iloc[-1] / last_c - 1
    feats["sma20_ratio"] = ta.sma(c, length=20).iloc[-1] / last_c - 1
    feats["sma50_ratio"] = ta.sma(c, length=50).iloc[-1] / last_c - 1

    atr = ta.atr(h, l, c, length=14)
    feats["atr_ratio"] = atr.iloc[-1] / last_c

    bb = ta.bbands(c, length=20, std=2)
    feats["bb_pct"] = bb["BBP_20_2.0_2.0"].iloc[-1]
    feats["bb_bw"] = bb["BBB_20_2.0_2.0"].iloc[-1]

    vol_sma = ta.sma(v, length=20)
    feats["vol_ratio"] = v.iloc[-1] / (vol_sma.iloc[-1] + 1e-8)
    obv = ta.obv(c, v)
    obv_mean = obv.rolling(20).mean().iloc[-1]
    obv_std = obv.rolling(20).std().iloc[-1] + 1e-8
    feats["obv_norm"] = (obv.iloc[-1] - obv_mean) / obv_std

    hl_range = max(h.iloc[-1] - l.iloc[-1], 1e-8)
    body = abs(c.iloc[-1] - o.iloc[-1])
    high_body = max(c.iloc[-1], o.iloc[-1])
    low_body = min(c.iloc[-1], o.iloc[-1])
    feats["body_ratio"] = body / hl_range
    feats["upper_wick"] = (h.iloc[-1] - high_body) / hl_range
    feats["lower_wick"] = (low_body - l.iloc[-1]) / hl_range
    feats["hl_ratio"] = hl_range / last_c

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

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        df = _klines_to_df(klines)
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

        if direction == Direction.HOLD or confidence < CONFIDENCE_THRESHOLD:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"XGB: {label} @ {confidence:.2f} — below threshold",
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
