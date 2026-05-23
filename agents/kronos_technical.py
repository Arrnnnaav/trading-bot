import numpy as np
from agents.base import BaseAgent
from core.models import AgentVote, Direction, Market

try:
    from kronos import KronosModel, KronosTokenizer

    _KRONOS_AVAILABLE = True
except ImportError:
    _KRONOS_AVAILABLE = False


class KronosTechnicalAgent(BaseAgent):
    name = "KronosTechnical"
    CONFIDENCE_THRESHOLD = 0.50

    def __init__(self, model_name: str = "kronos-base"):
        self.model = None
        self.tokenizer = None
        self._history_embeddings: list = []  # rolling history for pattern match
        self._history_directions: list = []
        if _KRONOS_AVAILABLE:
            self.tokenizer = KronosTokenizer.from_pretrained(model_name)
            self.model = KronosModel.from_pretrained(model_name)

    def _klines_to_array(self, klines: list[dict]):
        return np.array(
            [[k["open"], k["high"], k["low"], k["close"], k["volume"]] for k in klines],
            dtype=np.float32,
        )

    def _encode_klines(self, klines: list[dict]) -> list[float]:
        if not _KRONOS_AVAILABLE or self.model is None:
            arr = self._klines_to_array(klines)
            # Fallback: simple normalized price features
            closes = arr[:, 3]
            features = list((closes - closes.mean()) / (closes.std() + 1e-8))
            return features[:768] + [0.0] * max(0, 768 - len(features))
        import torch

        arr = self._klines_to_array(klines)
        tokens = self.tokenizer(torch.tensor(arr).unsqueeze(0))
        with torch.no_grad():
            hidden = self.model.encode(tokens)[:, -1, :]
        return hidden.squeeze(0).tolist()

    def _forecast(self, embedding: list[float]) -> list[float]:
        if not _KRONOS_AVAILABLE or self.model is None:
            return [0.0] * 5
        import torch

        emb_tensor = torch.tensor(embedding).unsqueeze(0)
        with torch.no_grad():
            out = self.model.generate(emb_tensor, steps=5)
        return out.squeeze(0).tolist()

    def _pattern_match(self, embedding: list[float]) -> dict:
        if len(self._history_embeddings) < 5:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "matched": 0,
                "total": 0,
            }

        emb = np.array(embedding)
        similarities = []
        for hist_emb, hist_dir in zip(
            self._history_embeddings, self._history_directions
        ):
            sim = np.dot(emb, hist_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(hist_emb) + 1e-8
            )
            similarities.append((sim, hist_dir))

        top = sorted(similarities, key=lambda x: x[0], reverse=True)[:10]
        longs = sum(1 for _, d in top if d == "LONG")
        shorts = sum(1 for _, d in top if d == "SHORT")
        total = len(top)

        if longs > shorts:
            return {
                "direction": "LONG",
                "confidence": longs / total,
                "matched": longs,
                "total": total,
            }
        elif shorts > longs:
            return {
                "direction": "SHORT",
                "confidence": shorts / total,
                "matched": shorts,
                "total": total,
            }
        return {"direction": "HOLD", "confidence": 0.0, "matched": 0, "total": total}

    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        embedding = self._encode_klines(klines)
        pattern = self._pattern_match(embedding)
        forecast = self._forecast(embedding)

        direction_str = pattern["direction"]
        confidence = pattern["confidence"]

        if confidence < self.CONFIDENCE_THRESHOLD:
            direction_str = "HOLD"
            confidence = 0.0

        forecast_str = (
            ", ".join(f"{p:.0f}" for p in forecast[:5]) if any(forecast) else "N/A"
        )
        reasoning = (
            f"Pattern match: {pattern['matched']}/{pattern['total']} similar setups resolved {direction_str}. "
            f"5-period forecast: [{forecast_str}]"
        )

        return AgentVote(
            agent_name=self.name,
            direction=Direction(direction_str),
            confidence=confidence,
            reasoning=reasoning,
        )

    def record_outcome(self, embedding: list[float], direction: str):
        self._history_embeddings.append(np.array(embedding))
        self._history_directions.append(direction)
        if len(self._history_embeddings) > 200:
            self._history_embeddings.pop(0)
            self._history_directions.pop(0)
