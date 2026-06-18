import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from training.model import ChronosClassifier
from training.dataset import load_all_tickers, split_datasets, compute_class_weights

MODEL_DIR = Path("models/kronos_india")
CHECKPOINT = "amazon/chronos-t5-small"

# Phase 1: freeze encoder, train head only
PHASE1_EPOCHS = 2
PHASE1_LR = 1e-3
PHASE1_BATCH = 64

# Phase 2: unfreeze all, fine-tune end-to-end
PHASE2_EPOCHS = 3
PHASE2_LR = 1e-5
PHASE2_BATCH = 32


def _unpack_batch(batch):
    if len(batch) == 3:
        close_batch, _stat_batch, label_batch = batch
        return close_batch, label_batch
    close_batch, label_batch = batch
    return close_batch, label_batch


def train_epoch(
    model: ChronosClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor = None,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    total_loss = 0.0
    for batch in tqdm(loader, leave=False, desc="train"):
        close_batch, label_batch = _unpack_batch(batch)
        close_batch = close_batch.to(device)
        label_batch = label_batch.to(device)
        optimizer.zero_grad()
        logits = model(close_batch)
        loss = criterion(logits, label_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(
    model: ChronosClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="eval"):
            close_batch, label_batch = _unpack_batch(batch)
            close_batch = close_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(close_batch)
            loss = criterion(logits, label_batch)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(label_batch.cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": total_loss / len(loader), "macro_f1": macro_f1}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    df = load_all_tickers()
    train_ds, val_ds, test_ds = split_datasets(df)
    class_weights = compute_class_weights(train_ds)
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    # Save test split indices for backtesting
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "config.json").write_text(
        json.dumps(
            {
                "checkpoint": CHECKPOINT,
                "window": 96,
                "label_names": ["LONG", "SHORT", "HOLD"],
                "confidence_threshold": 0.55,
            },
            indent=2,
        )
    )

    model = ChronosClassifier(CHECKPOINT).to(device)
    best_f1 = 0.0

    # Phase 1: train head only
    print("\n=== Phase 1: training head (encoder frozen) ===")
    model.freeze_encoder()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=PHASE1_LR)
    train_loader = DataLoader(
        train_ds, batch_size=PHASE1_BATCH, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=PHASE1_BATCH, shuffle=False, num_workers=2
    )

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = eval_epoch(model, val_loader, device)
        print(
            f"  Epoch {epoch}/{PHASE1_EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            model.save(str(MODEL_DIR / "best.pt"))
            print(f"  New best F1={best_f1:.4f} — checkpoint saved")

    # Phase 2: fine-tune all
    print("\n=== Phase 2: fine-tuning full model ===")
    model.unfreeze_encoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE2_EPOCHS
    )
    train_loader = DataLoader(
        train_ds, batch_size=PHASE2_BATCH, shuffle=True, num_workers=2
    )

    patience = 2
    no_improve = 0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()
        print(
            f"  Epoch {epoch}/{PHASE2_EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            model.save(str(MODEL_DIR / "best.pt"))
            print(f"  New best F1={best_f1:.4f} — checkpoint saved")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint: {MODEL_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
