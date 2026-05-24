import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple

# Import Chronos — raises ImportError clearly if not installed
from chronos import ChronosPipeline

from core.models import Direction
from training.dataset import LABEL_NAMES

CHECKPOINT_DEFAULT = "amazon/chronos-t5-small"
D_MODEL = 512  # T5-small hidden dimension
CONFIDENCE_THRESHOLD = 0.55


class ChronosClassifier(nn.Module):
    """
    Chronos-2 T5 encoder + 3-class classification head.
    Input: normalized close prices [batch, 96].
    Output: logits [batch, 3] — indices 0=LONG, 1=SHORT, 2=HOLD.
    """

    def __init__(self, checkpoint: str = CHECKPOINT_DEFAULT):
        super().__init__()
        pipeline = ChronosPipeline.from_pretrained(
            checkpoint,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        # Chronos internals: pipeline.model is ChronosModel,
        # pipeline.model.model is T5ForConditionalGeneration,
        # .encoder is the T5Stack encoder.
        self.tokenizer = pipeline.tokenizer
        self.encoder = pipeline.model.model.encoder

        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(0.2),
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        self._checkpoint = checkpoint

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, close: torch.Tensor) -> torch.Tensor:
        """
        close: [batch, seq_len] float32 — z-score normalized close prices.
        Returns logits [batch, 3].
        """
        device = close.device

        token_ids, attention_mask, _ = self.tokenizer.context_input_transform(
            close.cpu()
        )
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)

        enc_out = self.encoder(input_ids=token_ids, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state  # [batch, seq, D_MODEL]

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # [batch, D_MODEL]

        return self.head(pooled)

    def predict(self, close: torch.Tensor) -> Tuple[Direction, float]:
        """
        close: [seq_len] — single sample, z-score normalized.
        Returns (Direction, confidence).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(close.unsqueeze(0))  # [1, 3]
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # [3]
            confidence = probs.max().item()
            label_idx = probs.argmax().item()

        if confidence < CONFIDENCE_THRESHOLD:
            return Direction.HOLD, 0.0
        return Direction(LABEL_NAMES[label_idx]), confidence

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state": self.encoder.state_dict(),
                "head_state": self.head.state_dict(),
                "checkpoint": self._checkpoint,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str, checkpoint: str = CHECKPOINT_DEFAULT
    ) -> "ChronosClassifier":
        state = torch.load(path, map_location="cpu", weights_only=False)
        ckpt = state.get("checkpoint", checkpoint)
        model = cls(ckpt)
        model.encoder.load_state_dict(state["encoder_state"])
        model.head.load_state_dict(state["head_state"])
        return model
