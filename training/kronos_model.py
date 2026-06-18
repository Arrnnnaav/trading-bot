"""KronosClassifier — lightweight Transformer for NSE daily OHLCV sequences."""

import torch
import torch.nn as nn
from pathlib import Path

LABEL_NAMES = ["LONG", "SHORT", "HOLD"]


class KronosClassifier(nn.Module):
    """
    3-class Transformer classifier for OHLCV % change sequences.

    Input:  [batch, seq_len, 5]  — per-bar % change of open/high/low/close + vol/1M
    Output: [batch, 3]           — logits for LONG / SHORT / HOLD (indices 0/1/2)

    predict() implements the interface KronosAgent expects:
        model.predict(x: list[list[float]]) → {"long_prob", "short_prob", "hold_prob"}
    """

    INPUT_DIM = 5
    N_CLASSES = 3

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(self.INPUT_DIM, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.N_CLASSES),
        )

    def freeze_encoder(self) -> None:
        for p in self.transformer.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.transformer.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, 5] → logits [batch, 3]"""
        x = self.input_proj(x)  # [batch, seq_len, hidden]
        x = self.transformer(x)  # [batch, seq_len, hidden]
        x = x.mean(dim=1)  # mean pool → [batch, hidden]
        return self.head(x)  # [batch, 3]

    def predict(self, x: list) -> dict:
        """
        x: list of [open_pct, high_pct, low_pct, close_pct, vol_M] per bar
           (output of KronosAgent._prepare_input)
        Returns: {"long_prob": float, "short_prob": float, "hold_prob": float}
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, seq, 5]
            logits = self(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return {
            "long_prob": probs[0],
            "short_prob": probs[1],
            "hold_prob": probs[2],
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> "KronosClassifier":
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        return model
