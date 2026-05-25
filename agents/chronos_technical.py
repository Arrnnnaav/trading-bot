import numpy as np
import torch
from pathlib import Path
from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

# v2 preferred (attention pool + focal loss trained); falls back to v1
MODEL_PATH_V2 = "models/chronos_crypto_v2/best.pt"
MODEL_PATH_V1 = "models/chronos_crypto/best.pt"
WINDOW = 96


def _load_best_model(device: torch.device):
    """Load v2 if available, else v1. Returns (model, version_str)."""
    if Path(MODEL_PATH_V2).exists():
        state = torch.load(MODEL_PATH_V2, map_location="cpu", weights_only=False)
        if state.get("version") == "v2":
            from training.model_v2 import ChronosClassifierV2

            model = ChronosClassifierV2.load(MODEL_PATH_V2).to(device)
            return model, "v2"
    if Path(MODEL_PATH_V1).exists():
        from training.model import ChronosClassifier

        model = ChronosClassifier.load(MODEL_PATH_V1).to(device)
        return model, "v1"
    raise FileNotFoundError(
        "No Chronos model found. Train first:\n"
        "  python -m training.train_v2   (recommended)\n"
        "  python -m training.train      (baseline)"
    )


class ChronosTechnicalAgent(BaseAgent):
    name = "ChronosTechnical"
    _version = "v1"  # default for __new__-bypassed test instances

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self._version = _load_best_model(self.device)
        self.model.eval()

    def analyze(self, ticker: str, klines: list, market: Market, **kwargs) -> AgentVote:
        if len(klines) < WINDOW:
            return AgentVote(
                agent_name=self.name,
                direction=Direction.HOLD,
                confidence=0.0,
                reasoning=f"Insufficient candles: {len(klines)} < {WINDOW} required",
            )

        closes = np.array([k["close"] for k in klines[-WINDOW:]], dtype=np.float32)
        mean, std = closes.mean(), closes.std() + 1e-8
        normalized = (closes - mean) / std
        close_tensor = torch.tensor(normalized, dtype=torch.float32).to(self.device)

        direction, confidence = self.model.predict(close_tensor)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(close_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        reasoning = (
            f"Chronos-2({self._version}): {direction.value} (p={confidence:.2f}) | "
            f"LONG={probs[0]:.2f} SHORT={probs[1]:.2f} HOLD={probs[2]:.2f} | "
            f"{WINDOW} candles, {ticker}"
        )

        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
        )
