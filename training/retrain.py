"""
Incremental fine-tune ChronosClassifier from newly accumulated
live signal outcomes in data/chronos_outcomes.jsonl.

Run manually when: wc -l data/chronos_outcomes.jsonl >= 500
Or schedule weekly via cron.
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from training.model import ChronosClassifier
from training.dataset import LABEL_MAP, WINDOW

OUTCOMES_PATH = "data/chronos_outcomes.jsonl"
MODEL_PATH = "models/kronos_india/best.pt"
MIN_NEW_SAMPLES = 500
EPOCHS = 2
LR = 5e-6
BATCH = 16


class OutcomesDataset(Dataset):
    """Dataset built from chronos_outcomes.jsonl + original training parquet."""

    def __init__(self, outcomes_path: str, data_dir: str = "data/training"):
        lines = Path(outcomes_path).read_text().strip().split("\n")
        records = [json.loads(l) for l in lines if l.strip()]

        self.samples = []
        ticker_dfs = {}
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]:
            p = Path(data_dir) / f"{symbol}_15m.parquet"
            if p.exists():
                ticker_dfs[symbol[:3]] = pd.read_parquet(p)

        for r in records:
            ticker_prefix = r["ticker"].split("/")[0].upper()
            df = ticker_dfs.get(ticker_prefix)
            if df is None:
                continue
            label = LABEL_MAP.get(r["predicted"])
            if label is None:
                continue
            if len(df) < WINDOW:
                continue
            closes = df["close"].to_numpy(dtype=np.float32)[-WINDOW:]
            mean, std = closes.mean(), closes.std() + 1e-8
            normalized = (closes - mean) / std
            self.samples.append((torch.tensor(normalized), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    outcomes = Path(OUTCOMES_PATH)
    if not outcomes.exists():
        print(f"No outcomes file at {OUTCOMES_PATH}")
        return

    n_lines = sum(1 for _ in outcomes.open())
    if n_lines < MIN_NEW_SAMPLES:
        print(f"Only {n_lines} outcomes — need {MIN_NEW_SAMPLES} to retrain. Skipping.")
        return

    print(f"Retraining on {n_lines} outcomes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChronosClassifier.load(MODEL_PATH).to(device)
    model.unfreeze_encoder()

    ds = OutcomesDataset(OUTCOMES_PATH)
    if len(ds) < 50:
        print(f"Only {len(ds)} valid samples after matching to OHLCV. Skipping.")
        return

    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for close_batch, label_batch in loader:
            close_batch, label_batch = close_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            logits = model(close_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/{EPOCHS}  loss={total_loss / len(loader):.4f}")

    model.save(MODEL_PATH)
    print(f"Retrained model saved to {MODEL_PATH}")

    # Archive processed outcomes so next run starts fresh
    archive = Path(OUTCOMES_PATH).with_suffix(".jsonl.bak")
    outcomes.rename(archive)
    print(f"Outcomes archived to {archive}")


if __name__ == "__main__":
    main()
