import numpy as np
import torch
from pathlib import Path
from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

MODEL_PATH_DEFAULT = "models/chronos_crypto/best.pt"
WINDOW = 96


class ChronosTechnicalAgent(BaseAgent):
    name = "ChronosTechnical"

    def __init__(self, model_path: str = MODEL_PATH_DEFAULT):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Chronos model not found at '{model_path}'. "
                "Train the model first: python training/train.py"
            )
        from training.model import ChronosClassifier

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChronosClassifier.load(model_path).to(self.device)
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
            f"Chronos-2: {direction.value} (p={confidence:.2f}) | "
            f"LONG={probs[0]:.2f} SHORT={probs[1]:.2f} HOLD={probs[2]:.2f} | "
            f"{WINDOW} candles, {ticker}"
        )

        return AgentVote(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
        )
