"""
training/monitor.py — parse train logs, detect overfitting/underfitting, print diagnosis.

Usage:
  python -m training.monitor                        # check both v1 and v2
  python -m training.monitor models/chronos_crypto/train_log.txt
"""

import re
import sys
from pathlib import Path


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+"
    r"train_loss=([\d.]+)\s+"
    r"(?:T=[\d.]+\s+)?"  # optional temperature (v2)
    r".*?val_loss=([\d.]+)\s+"
    r".*?(?:val_f1|macro_f1)=([\d.]+)"  # v1 uses val_f1, v2 uses macro_f1
    r"(?:\s+\[LONG=([\d.]+)\s+SHORT=([\d.]+)\s+HOLD=([\d.]+)\])?"
    r"(?:.*?preds%:\s*LONG=([\d.]+)\s+SHORT=([\d.]+)\s+HOLD=([\d.]+))?",
    re.IGNORECASE,
)
PHASE_RE = re.compile(r"=== Phase (\d+):", re.IGNORECASE)
BEST_RE = re.compile(r"New best F1=([\d.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    epochs = []
    current_phase = 1
    best_f1 = 0.0

    for line in lines:
        pm = PHASE_RE.search(line)
        if pm:
            current_phase = int(pm.group(1))
            continue
        bm = BEST_RE.search(line)
        if bm:
            best_f1 = float(bm.group(1))
        em = EPOCH_RE.search(line)
        if em:
            g = em.groups()
            epochs.append(
                {
                    "phase": current_phase,
                    "epoch": int(g[0]),
                    "of": int(g[1]),
                    "train_loss": float(g[2]),
                    "val_loss": float(g[3]),
                    "macro_f1": float(g[4]),
                    "long_f1": float(g[5]) if g[5] else None,
                    "short_f1": float(g[6]) if g[6] else None,
                    "hold_f1": float(g[7]) if g[7] else None,
                    "pred_long_pct": float(g[8]) if g[8] else None,
                    "pred_short_pct": float(g[9]) if g[9] else None,
                    "pred_hold_pct": float(g[10]) if g[10] else None,
                }
            )

    return {"epochs": epochs, "best_f1": best_f1, "raw_lines": len(lines)}


def diagnose(epochs: list) -> list[str]:
    issues = []
    if not epochs:
        return ["No epoch results yet — training still on first epoch or not started."]

    last = epochs[-1]
    f1s = [e["macro_f1"] for e in epochs]
    train_losses = [e["train_loss"] for e in epochs]
    val_losses = [e["val_loss"] for e in epochs]

    # --- HOLD collapse ---
    if last["pred_hold_pct"] is not None and last["pred_hold_pct"] > 90:
        issues.append(
            f"HOLD COLLAPSE: model predicting HOLD {last['pred_hold_pct']:.0f}% of time. "
            "Needs higher class weights or stronger focal gamma."
        )
    if last["long_f1"] is not None and last["long_f1"] < 0.05:
        issues.append(
            f"LONG F1={last['long_f1']:.3f} — LONG class not learned. "
            "Increase LONG weight or use focal loss gamma=3."
        )
    if last["short_f1"] is not None and last["short_f1"] < 0.05:
        issues.append(f"SHORT F1={last['short_f1']:.3f} — SHORT class not learned.")

    # --- Loss/F1 divergence (HOLD collapse indicator) ---
    if len(epochs) >= 2:
        f1_delta = epochs[-1]["macro_f1"] - epochs[-2]["macro_f1"]
        loss_delta = epochs[-1]["val_loss"] - epochs[-2]["val_loss"]
        if loss_delta < -0.005 and f1_delta < -0.001:
            issues.append(
                f"HOLD COLLAPSE SIGNAL: val_loss improving ({loss_delta:+.4f}) but macro_f1 dropping "
                f"({f1_delta:+.4f}). Model learning confident HOLD predictions. "
                "Fix: focal loss (v2) or stronger class weights."
            )

    # --- Overfitting ---
    if len(epochs) >= 3:
        val_trend = val_losses[-1] - val_losses[-3]
        train_trend = train_losses[-1] - train_losses[-3]
        gap = last["val_loss"] - last["train_loss"]
        if val_trend > 0.02 and train_trend < 0:
            issues.append(
                f"OVERFITTING: train_loss falling ({train_trend:+.3f}) but val_loss rising "
                f"({val_trend:+.3f}). Add dropout or weight decay; reduce Phase 2 LR."
            )
        elif gap > 0.3:
            issues.append(
                f"OVERFITTING signal: val_loss - train_loss = {gap:.3f}. "
                "Consider stronger dropout or L2 regularization."
            )

    # --- Underfitting ---
    if last["macro_f1"] < 0.25:
        issues.append(
            f"UNDERFITTING: macro_f1={last['macro_f1']:.3f} — barely above random (0.33 expected for 3 class). "
            "Check if encoder is actually training; LR may be too low."
        )
    if len(train_losses) >= 3 and abs(train_losses[-1] - train_losses[-3]) < 0.005:
        issues.append(
            "STALLED: train_loss flat for 3 epochs — LR too low or gradient vanishing."
        )

    # --- Healthy signs ---
    if not issues:
        if last["macro_f1"] > 0.40:
            issues.append(
                f"HEALTHY: macro_f1={last['macro_f1']:.3f} — good signal across all classes."
            )
        else:
            issues.append(
                f"MARGINAL: macro_f1={last['macro_f1']:.3f} — training but F1 still low. "
                "Continue training; may improve in Phase 2."
            )

    return issues


def print_report(log_path: Path):
    if not log_path.exists():
        print(f"  Not found: {log_path}")
        return
    data = parse_log(log_path)
    epochs = data["epochs"]
    print(f"\n{'=' * 60}")
    print(f"Log: {log_path}  ({data['raw_lines']} lines)")
    print(f"Epochs recorded: {len(epochs)}  |  Best F1: {data['best_f1']:.4f}")
    print()

    if epochs:
        print("  Epoch history:")
        for e in epochs:
            tag = f"[P{e['phase']}]"
            line = (
                f"    {tag} {e['epoch']}/{e['of']}  "
                f"train={e['train_loss']:.4f}  val={e['val_loss']:.4f}  "
                f"macro_f1={e['macro_f1']:.4f}"
            )
            if e["long_f1"] is not None:
                line += f"  LONG={e['long_f1']:.3f} SHORT={e['short_f1']:.3f} HOLD={e['hold_f1']:.3f}"
            if e["pred_hold_pct"] is not None:
                line += f"  preds%: L={e['pred_long_pct']:.0f} S={e['pred_short_pct']:.0f} H={e['pred_hold_pct']:.0f}"
            print(line)

    print()
    print("  Diagnosis:")
    for d in diagnose(epochs):
        print(f"    → {d}")
    print()


def main():
    if len(sys.argv) > 1:
        print_report(Path(sys.argv[1]))
    else:
        print_report(Path("models/chronos_crypto/train_log.txt"))
        print_report(Path("models/chronos_crypto_v2/train_log.txt"))


if __name__ == "__main__":
    main()
