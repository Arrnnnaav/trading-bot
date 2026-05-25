"""
ChronosClassifierV2 — attention pooling + extra stat features + temperature scaling.
Drop-in replacement for ChronosClassifier; save format uses same keys
so agents/chronos_technical.py load() still works.

n_extra=6: momentum/vol stat features fused into head after encoder pooling.
Set n_extra=0 for old-style model (backward compat with checkpoints trained without extras).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional

from chronos import ChronosPipeline

from core.models import Direction
from training.dataset import LABEL_NAMES, N_STAT_FEATURES

CHECKPOINT_DEFAULT = "amazon/chronos-t5-small"
D_MODEL = 512
CONFIDENCE_THRESHOLD = 0.55


class _AttentionPool(nn.Module):
    """Learned weighted sum over encoder token dimension."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: [B, seq, D], mask: [B, seq]
        scores = self.attn(hidden).squeeze(-1)  # [B, seq]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, seq, 1]
        return (hidden * weights).sum(1)  # [B, D]


class ChronosClassifierV2(nn.Module):
    """
    Chronos-2 encoder + attention pooling + optional stat feature fusion + 3-class head + temperature.

    Extra features (n_extra=6 by default):
      ret_1, ret_5, ret_20 — price momentum at 3 horizons
      vol_5, vol_20        — short/medium volatility
      vol_ratio            — vol expansion indicator (vol_5 / vol_20)

    These are concatenated to the pooled encoder output before the MLP head,
    giving the model explicit directional signals alongside the learned representation.
    """

    def __init__(
        self, checkpoint: str = CHECKPOINT_DEFAULT, n_extra: int = N_STAT_FEATURES
    ):
        super().__init__()
        pipeline = ChronosPipeline.from_pretrained(
            checkpoint, device_map="cpu", torch_dtype=torch.float32
        )
        self.tokenizer = pipeline.tokenizer
        self.encoder = pipeline.model.model.encoder
        self._checkpoint = checkpoint
        self._n_extra = n_extra

        self.pool = _AttentionPool(D_MODEL)

        head_in = D_MODEL + n_extra  # 518 with extras, 512 without

        # Head: LayerNorm → skip-connected MLP → classifier
        self.norm = nn.LayerNorm(head_in)
        self.proj = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.skip = nn.Linear(head_in, 128, bias=False)
        self.drop = nn.Dropout(0.25)
        self.classifier = nn.Linear(128, 3)

        # Learnable temperature (log scale so always positive)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(
        self, close: torch.Tensor, extra: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        close: [B, seq_len] — z-score normalised.
        extra: [B, n_extra] — stat features (optional, but required if n_extra > 0).
        Returns logits [B, 3].
        """
        device = close.device
        token_ids, attention_mask, _ = self.tokenizer.context_input_transform(
            close.cpu()
        )
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)

        enc_out = self.encoder(input_ids=token_ids, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state  # [B, seq, D]

        pooled = self.pool(hidden, attention_mask)  # [B, D]

        if self._n_extra > 0 and extra is not None:
            pooled = torch.cat([pooled, extra.to(device)], dim=-1)  # [B, D + n_extra]

        normed = self.norm(pooled)
        h = self.proj(normed)
        h = h + self.skip(normed)  # residual
        h = self.drop(h)
        logits = self.classifier(h)  # [B, 3]

        T = self.log_temperature.exp().clamp(min=0.1)
        return logits / T

    def predict(
        self, close: torch.Tensor, extra: Optional[torch.Tensor] = None
    ) -> Tuple[Direction, float]:
        """
        close: [seq_len] single sample, z-score normalised.
        extra: [n_extra] stat features for this sample (1-D, not batched).
        """
        self.eval()
        with torch.no_grad():
            extra_batch = extra.unsqueeze(0) if extra is not None else None
            logits = self.forward(close.unsqueeze(0), extra=extra_batch)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            confidence = probs.max().item()
            label_idx = probs.argmax().item()
        if confidence < CONFIDENCE_THRESHOLD:
            return Direction.HOLD, 0.0
        return Direction(LABEL_NAMES[label_idx]), confidence

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": "v2",
                "n_extra": self._n_extra,
                "encoder_state": self.encoder.state_dict(),
                "head_state": {
                    "pool": self.pool.state_dict(),
                    "norm": self.norm.state_dict(),
                    "proj": self.proj.state_dict(),
                    "skip": self.skip.state_dict(),
                    "drop": self.drop.state_dict(),
                    "classifier": self.classifier.state_dict(),
                    "log_temperature": self.log_temperature.data,
                },
                "checkpoint": self._checkpoint,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str, checkpoint: str = CHECKPOINT_DEFAULT
    ) -> "ChronosClassifierV2":
        state = torch.load(path, map_location="cpu", weights_only=False)
        ckpt = state.get("checkpoint", checkpoint)
        n_extra = state.get(
            "n_extra", 0
        )  # backward compat: old checkpoints had no extra
        model = cls(ckpt, n_extra=n_extra)
        model.encoder.load_state_dict(state["encoder_state"])
        hs = state["head_state"]
        model.pool.load_state_dict(hs["pool"])
        model.norm.load_state_dict(hs["norm"])
        model.proj.load_state_dict(hs["proj"])
        model.skip.load_state_dict(hs["skip"])
        model.drop.load_state_dict(hs["drop"])
        model.classifier.load_state_dict(hs["classifier"])
        model.log_temperature.data = hs["log_temperature"]
        return model
