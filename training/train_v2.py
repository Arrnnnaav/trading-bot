"""
train_v2.py — improved training for ChronosClassifierV2.

Improvements over train.py (v1):
  - Focal loss (gamma=2) instead of weighted CE — handles HOLD dominance better
  - Class weights uncapped (no 5x clip) — full inverse-frequency
  - Attention pooling model (ChronosClassifierV2)
  - Per-class F1 logging (LONG / SHORT / HOLD) — detects HOLD collapse
  - LR warmup (linear 3% steps) for both phases
  - Gradient accumulation (Phase 2 effective batch = 128)
  - Temperature is jointly learned during Phase 2
  - Saves to models/chronos_crypto_v2/best.pt

Run: python -m training.train_v2
"""

import json
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from training.model_v2 import ChronosClassifierV2
from training.focal_loss import FocalLoss
from training.dataset import (
    load_all_tickers,
    split_datasets,
    LABEL_NAMES,
)

MODEL_DIR = Path("models/chronos_crypto_v2")
CHECKPOINT = "amazon/chronos-t5-small"

# Phase 1: frozen encoder, head only — focal loss, warmup
PHASE1_EPOCHS = 3  # extra epoch since focal harder to fit initially
PHASE1_LR = 8e-4
PHASE1_BATCH = 64
PHASE1_WARMUP_FRAC = 0.05  # 5% steps linear warmup

# Phase 2: full fine-tune — cosine LR + gradient accumulation
PHASE2_EPOCHS = 4
PHASE2_LR = 8e-6
PHASE2_BATCH = 32
PHASE2_ACCUM = 4  # effective batch = 32 * 4 = 128
PHASE2_WARMUP_FRAC = 0.03


def _compute_full_class_weights(dataset) -> torch.Tensor:
    """Inverse-frequency weights, NO cap — let the imbalance drive the loss."""
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    total = counts.sum()
    weights = total / (3 * counts + 1e-8)
    print(
        f"  Class weights (uncapped): LONG={weights[0]:.2f}  SHORT={weights[1]:.2f}  HOLD={weights[2]:.2f}"
    )
    return torch.tensor(weights, dtype=torch.float32)


def _warmup_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: ChronosClassifierV2,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    criterion,
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (close_batch, label_batch) in enumerate(
        tqdm(loader, leave=False, desc="train")
    ):
        close_batch = close_batch.to(device)
        label_batch = label_batch.to(device)
        logits = model(close_batch)
        loss = criterion(logits, label_batch) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
    # flush remaining
    if (step + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(loader)


def eval_epoch(
    model: ChronosClassifierV2,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for close_batch, label_batch in tqdm(loader, leave=False, desc="eval"):
            close_batch = close_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(close_batch)
            loss = criterion(logits, label_batch)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(label_batch.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(
        all_labels, all_preds, average=None, zero_division=0, labels=[0, 1, 2]
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    # Prediction distribution
    pred_dist = np.bincount(all_preds, minlength=3)
    pred_pct = pred_dist / len(all_preds) * 100

    return {
        "loss": total_loss / len(loader),
        "macro_f1": macro_f1,
        "long_f1": float(per_class[0]),
        "short_f1": float(per_class[1]),
        "hold_f1": float(per_class[2]),
        "pred_long_pct": float(pred_pct[0]),
        "pred_short_pct": float(pred_pct[1]),
        "pred_hold_pct": float(pred_pct[2]),
        "confusion": cm.tolist(),
    }


def _fmt_metrics(m: Dict) -> str:
    return (
        f"loss={m['loss']:.4f}  macro_f1={m['macro_f1']:.4f}  "
        f"[LONG={m['long_f1']:.3f} SHORT={m['short_f1']:.3f} HOLD={m['hold_f1']:.3f}]  "
        f"preds%: LONG={m['pred_long_pct']:.1f} SHORT={m['pred_short_pct']:.1f} HOLD={m['pred_hold_pct']:.1f}"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    df = load_all_tickers()
    train_ds, val_ds, test_ds = split_datasets(df)
    class_weights = _compute_full_class_weights(train_ds)
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "config.json").write_text(
        json.dumps(
            {
                "checkpoint": CHECKPOINT,
                "version": "v2",
                "window": 96,
                "label_names": LABEL_NAMES,
                "confidence_threshold": 0.55,
                "architecture": "attention_pool+residual_head+temperature",
            },
            indent=2,
        )
    )

    model = ChronosClassifierV2(CHECKPOINT).to(device)
    best_f1 = 0.0

    # ── Phase 1: head only ─────────────────────────────────────────────────
    print("\n=== Phase 1: head training (encoder frozen) ===")
    model.freeze_encoder()
    # Temperature should be fixed in Phase 1 (only head learns)
    model.log_temperature.requires_grad = False

    focal = FocalLoss(gamma=2.0, weight=class_weights.to(device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=PHASE1_LR, weight_decay=1e-2)

    train_loader = DataLoader(
        train_ds, batch_size=PHASE1_BATCH, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=PHASE1_BATCH, shuffle=False, num_workers=2
    )

    total_steps = PHASE1_EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * PHASE1_WARMUP_FRAC)
    scheduler = _warmup_scheduler(optimizer, total_steps, warmup_steps)
    print(f"  LR warmup: {warmup_steps} steps → cosine decay over {total_steps} total")

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, focal
        )
        val_m = eval_epoch(model, val_loader, device)
        print(
            f"  Epoch {epoch}/{PHASE1_EPOCHS}  train_loss={train_loss:.4f}  {_fmt_metrics(val_m)}"
        )
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            model.save(str(MODEL_DIR / "best.pt"))
            print(f"  ✓ New best F1={best_f1:.4f} — saved")

    # ── Phase 2: full fine-tune ────────────────────────────────────────────
    print("\n=== Phase 2: full fine-tune (encoder unfrozen) ===")
    model.unfreeze_encoder()
    model.log_temperature.requires_grad = True  # learn temperature now

    # Separate param groups: lower LR for encoder, higher for head+temp
    encoder_params = list(model.encoder.parameters())
    head_params = [
        p for p in model.parameters() if not any(p is ep for ep in encoder_params)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": PHASE2_LR},
            {"params": head_params, "lr": PHASE2_LR * 10},  # head learns 10× faster
        ],
        weight_decay=1e-2,
    )

    train_loader_p2 = DataLoader(
        train_ds, batch_size=PHASE2_BATCH, shuffle=True, num_workers=2
    )
    total_steps_p2 = PHASE2_EPOCHS * (len(train_loader_p2) // PHASE2_ACCUM)
    warmup_steps_p2 = int(total_steps_p2 * PHASE2_WARMUP_FRAC)
    scheduler_p2 = _warmup_scheduler(optimizer, total_steps_p2, warmup_steps_p2)
    print(
        f"  Effective batch={PHASE2_BATCH * PHASE2_ACCUM}  LR warmup={warmup_steps_p2} steps"
    )

    patience = 3
    no_improve = 0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader_p2, optimizer, scheduler_p2, device, focal, PHASE2_ACCUM
        )
        val_m = eval_epoch(model, val_loader, device)
        T = model.log_temperature.exp().item()
        print(
            f"  Epoch {epoch}/{PHASE2_EPOCHS}  train_loss={train_loss:.4f}  "
            f"T={T:.3f}  {_fmt_metrics(val_m)}"
        )
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            model.save(str(MODEL_DIR / "best.pt"))
            print(f"  ✓ New best F1={best_f1:.4f} — saved")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (patience={patience})")
                break

    # Final eval on test set
    test_loader = DataLoader(
        test_ds, batch_size=PHASE2_BATCH, shuffle=False, num_workers=2
    )
    test_m = eval_epoch(model, test_loader, device)
    print(f"\nTest set:  {_fmt_metrics(test_m)}")
    print(f"Confusion matrix (LONG/SHORT/HOLD):\n{np.array(test_m['confusion'])}")

    print(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint: {MODEL_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
