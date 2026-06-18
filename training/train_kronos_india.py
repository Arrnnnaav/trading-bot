"""2-phase Kronos fine-tuning on NSE daily data."""

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from training.dataset import KronosDataset, load_india_data
from training.focal_loss import FocalLoss
from training.kronos_model import KronosClassifier

MODEL_DIR = Path("models/kronos_india")

# Phase 1: frozen encoder, head only
PHASE1_EPOCHS = 20
PHASE1_LR = 1e-4
PHASE1_BATCH = 64

# Phase 2: full fine-tune, gradient accumulation
PHASE2_EPOCHS = 10
PHASE2_LR = 1e-5
PHASE2_BATCH = 32
GRAD_ACCUM = 4


def _split_kronos(df, train_frac=0.80, val_frac=0.10):
    n = len(df)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    return (
        KronosDataset(df.iloc[:t].reset_index(drop=True)),
        KronosDataset(df.iloc[t:v].reset_index(drop=True)),
        KronosDataset(df.iloc[v:].reset_index(drop=True)),
    )


def _eval_f1(
    model: KronosClassifier, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(y.tolist())
    return f1_score(all_labels, all_preds, average="macro", zero_division=0)


def train(
    historical_dir: str = "data/historical",
    fii_dir: str = "data/fii_dii",
    device_str: str = "cpu",
) -> tuple:
    """
    2-phase Kronos fine-tuning.

    Phase 1 (20 epochs, lr=1e-4): encoder frozen, head only.
    Phase 2 (10 epochs, lr=1e-5): full fine-tune with grad accumulation (accum=4).
    Saves best.pt on val macro-F1 improvement.

    Returns (model, best_val_f1).
    """
    device = torch.device(device_str)

    print("Loading data...")
    df = load_india_data(historical_dir=historical_dir, fii_dir=fii_dir)
    train_ds, val_ds, _ = _split_kronos(df)

    if len(train_ds) == 0:
        raise ValueError(
            "KronosDataset empty — need ≥64 rows per symbol after labeling"
        )

    # Class weights for focal loss (from training labels)
    y_all = [train_ds[i][1] for i in range(len(train_ds))]
    counts = np.bincount(y_all, minlength=3).astype(np.float32)
    weights = torch.tensor(
        np.clip(counts.sum() / (3.0 * counts + 1e-8), 1.0, 5.0),
        dtype=torch.float32,
    ).to(device)
    criterion = FocalLoss(gamma=2.0, weight=weights)

    model = KronosClassifier(hidden_dim=64, n_heads=4, n_layers=2).to(device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # ------------------------------------------------------------------ #
    # Phase 1: frozen encoder, head only                                  #
    # ------------------------------------------------------------------ #
    model.freeze_encoder()
    opt1 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=PHASE1_LR
    )
    train_loader1 = DataLoader(train_ds, batch_size=PHASE1_BATCH, shuffle=True)

    print(f"Phase 1: {PHASE1_EPOCHS} epochs, lr={PHASE1_LR}, batch={PHASE1_BATCH}")
    for epoch in range(PHASE1_EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader1:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt1.zero_grad()
            loss.backward()
            opt1.step()
            total_loss += loss.item()
        val_f1 = _eval_f1(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save(str(MODEL_DIR / "best.pt"))
        print(
            f"  ep{epoch + 1:02d}  loss={total_loss / len(train_loader1):.4f}  val_f1={val_f1:.4f}  best={best_f1:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Phase 2: full fine-tune with gradient accumulation                  #
    # ------------------------------------------------------------------ #
    model.unfreeze_encoder()
    opt2 = torch.optim.Adam(model.parameters(), lr=PHASE2_LR)
    train_loader2 = DataLoader(train_ds, batch_size=PHASE2_BATCH, shuffle=True)

    print(
        f"Phase 2: {PHASE2_EPOCHS} epochs, lr={PHASE2_LR}, batch={PHASE2_BATCH}, accum={GRAD_ACCUM}"
    )
    for epoch in range(PHASE2_EPOCHS):
        model.train()
        opt2.zero_grad()
        for step, (x, y) in enumerate(train_loader2):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y) / GRAD_ACCUM
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step()
                opt2.zero_grad()
        val_f1 = _eval_f1(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save(str(MODEL_DIR / "best.pt"))
        print(f"  ep{epoch + 1:02d}  val_f1={val_f1:.4f}  best={best_f1:.4f}")

    print(f"Done. Best val macro-F1: {best_f1:.4f}  →  {MODEL_DIR / 'best.pt'}")
    return model, best_f1


if __name__ == "__main__":
    train()
