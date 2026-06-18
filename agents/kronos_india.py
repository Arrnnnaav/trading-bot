"""KronosAgent — NeoQuasar/Kronos-base inference for positional signals."""

import logging
from pathlib import Path
from typing import Optional

from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

_LOG = logging.getLogger(__name__)
_MIN_KLINES = 64
_MIN_CONFIDENCE = 0.60
_DEFAULT_MODEL_PATH = "models/kronos_india/best.pt"


class KronosAgent(BaseAgent):
    name = "kronos_india"

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self._model_path = model_path
        self._model: Optional[object] = None

    def _load_model(self) -> Optional[object]:
        if self._model is not None:
            return self._model
        path = Path(self._model_path)
        if not path.exists():
            return None
        try:
            import torch

            self._model = torch.load(str(path), map_location="cpu", weights_only=False)
            self._model.eval()
            return self._model
        except Exception as exc:
            _LOG.warning("Kronos model load failed: %s", exc)
            return None

    def _prepare_input(self, klines: list[dict]) -> list[list[float]]:
        """Convert last 64 klines to % change OHLCV."""
        window = klines[-_MIN_KLINES:]
        result = []
        for i in range(1, len(window)):
            prev_close = window[i - 1]["close"]
            if prev_close == 0:
                prev_close = 1.0
            result.append(
                [
                    (window[i]["open"] - prev_close) / prev_close,
                    (window[i]["high"] - prev_close) / prev_close,
                    (window[i]["low"] - prev_close) / prev_close,
                    (window[i]["close"] - prev_close) / prev_close,
                    window[i]["volume"] / 1e6,
                ]
            )
        return result

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        if len(klines) < _MIN_KLINES:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"insufficient data: {len(klines)} klines (need {_MIN_KLINES})",
            )
        try:
            model = self._load_model()
            if model is None:
                return AgentVote(
                    agent_name=self.name,
                    direction=Direction.HOLD,
                    confidence=0.0,
                    reasoning="model not trained yet — run training/train_kronos_india.py",
                )
            return self._infer(model, klines)
        except Exception as exc:
            _LOG.warning("KronosAgent inference failed for %s: %s", ticker, exc)
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"inference error: {exc}",
            )

    def _infer(self, model, klines: list[dict]) -> AgentVote:
        x = self._prepare_input(klines)
        probs = model.predict(x)
        long_prob = probs.get("long_prob", 0.0)
        short_prob = probs.get("short_prob", 0.0)
        confidence = max(long_prob, short_prob)
        if confidence < _MIN_CONFIDENCE:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"low confidence: L={long_prob:.2f} S={short_prob:.2f}",
            )
        direction = Direction.LONG if long_prob >= short_prob else Direction.SHORT
        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=round(confidence, 2),
            reasoning=f"Kronos: L={long_prob:.2f} S={short_prob:.2f}",
        )
