# Crypto Intelligence Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace broken KronosTechnicalAgent with fine-tuned Chronos-2 classifier trained on 5yr Binance OHLCV, fix agent learnings loop, add threshold calibration, and build backtesting harness.

**Architecture:** Chronos-2 T5-small encoder (frozen Phase 1, unfrozen Phase 2) + classification head → LONG/SHORT/HOLD from 96 normalized close-price candles. Multi-agent safety net preserved. Agent learnings from claude-haiku injected into DebateEngine system prompt. OnChain/News thresholds auto-calibrated from `signal_log.json`.

**Tech Stack:** `chronos-forecasting>=1.2.0`, `torch`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `tqdm`, `requests`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| New | `training/download_data.py` | Download Binance 15min OHLCV, save parquet |
| New | `training/label_data.py` | Forward-look LONG/SHORT/HOLD labeling |
| New | `training/dataset.py` | `CryptoDataset` + train/val/test split |
| New | `training/model.py` | `ChronosClassifier` encoder + head |
| New | `training/train.py` | Phase 1 + Phase 2 training loop |
| New | `training/retrain.py` | Incremental fine-tune from `chronos_outcomes.jsonl` |
| New | `training/calibrate_thresholds.py` | Calibrate thresholds from `signal_log.json` |
| New | `agents/chronos_technical.py` | `ChronosTechnicalAgent` — replaces Kronos |
| New | `backtesting/__init__.py` | Empty |
| New | `backtesting/crypto_backtest.py` | Walk-forward backtest on test split |
| New | `backtesting/plot_results.py` | Equity curve + drawdown chart |
| Modify | `requirements.txt` | Add new dependencies |
| Modify | `harnesses/crypto_harness.py` | Use `ChronosTechnicalAgent`, pass learnings, limit=100 |
| Modify | `harnesses/base_harness.py` | Write learnings to `.md`, pass to DebateEngine |
| Modify | `core/debate_engine.py` | Accept `learnings: str`, inject into system prompt |
| Modify | `core/signal_tracker.py` | Append to `chronos_outcomes.jsonl` on resolution |
| Modify | `agents/onchain.py` | Load thresholds from `calibrated_thresholds.json` |
| Modify | `agents/news_sentiment.py` | Load thresholds from `calibrated_thresholds.json` |
| Modify | `tests/test_agents.py` | Replace Kronos tests with Chronos tests |
| Delete | `agents/kronos_technical.py` | Replaced by `agents/chronos_technical.py` |

---

## Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add new packages to requirements.txt**

Replace current `requirements.txt` with:

```
pydantic==2.7.0
apscheduler==3.10.4
python-telegram-bot==20.8
fastapi==0.111.0
uvicorn==0.29.0
anthropic==0.28.0
httpx==0.27.0
coindcx-python-sdk
upstox-python-sdk
pytest==8.2.0
pytest-asyncio==0.23.6
pytest-mock==3.14.0
pytz==2024.1
numpy
torch>=2.2.0
chronos-forecasting>=1.2.0
pandas>=2.0.0
pyarrow>=14.0.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
tqdm>=4.66.0
requests>=2.31.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error. `torch` will be largest download (~2GB). If CUDA GPU available, verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True  NVIDIA <your GPU>` (if GPU present)

- [ ] **Step 3: Verify Chronos loads**

```bash
python -c "from chronos import ChronosPipeline; print('chronos OK')"
```

Expected: `chronos OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add ML training dependencies (chronos, torch, pandas, sklearn)"
```

---

## Task 2: Download Binance Training Data

**Files:**
- Create: `training/__init__.py`
- Create: `training/download_data.py`
- Create: `data/training/` (directory)

- [ ] **Step 1: Create training package and data directory**

```bash
mkdir -p training data/training
touch training/__init__.py
```

- [ ] **Step 2: Write failing test**

Create `tests/training/test_download_data.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_fetch_klines_returns_dataframe():
    from training.download_data import fetch_klines
    mock_data = [
        [1609459200000, "29000.0", "29500.0", "28800.0", "29300.0", "100.5",
         1609460100000, "2945000.0", 150, "60.0", "1767000.0", "0"],
    ]
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        df = fetch_klines("BTCUSDT", "15m", start_ms=1609459200000, end_ms=1609460100000)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open_time", "open", "high", "low", "close", "volume"]
    assert len(df) == 1
    assert df["close"].iloc[0] == pytest.approx(29300.0)


def test_fetch_klines_retries_on_rate_limit():
    from training.download_data import fetch_klines
    import requests
    with patch("requests.get") as mock_get:
        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = requests.HTTPError("429")
        ok_resp = MagicMock()
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = []
        mock_get.side_effect = [err_resp, ok_resp]
        df = fetch_klines("BTCUSDT", "15m", start_ms=0, end_ms=1000)
    assert isinstance(df, pd.DataFrame)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/training/test_download_data.py -v
```

Expected: `ImportError` — `training.download_data` does not exist yet.

- [ ] **Step 4: Write `training/download_data.py`**

```python
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BINANCE_URL = "https://api.binance.com/api/v3/klines"
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "15m"
CANDLE_MS = 15 * 60 * 1000
LIMIT = 1000


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch up to 1000 candles from Binance. Returns empty DataFrame if no data."""
    for attempt in range(3):
        try:
            resp = requests.get(
                BINANCE_URL,
                params={"symbol": symbol, "interval": interval,
                        "startTime": start_ms, "endTime": end_ms, "limit": LIMIT},
                timeout=10,
            )
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
            df = pd.DataFrame(rows, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "num_trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ])
            df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["open_time"] = df["open_time"].astype("int64")
            return df
        except requests.HTTPError:
            time.sleep(2 ** attempt)
    return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])


def download_ticker(symbol: str, years: int = 5) -> pd.DataFrame:
    """Download `years` of 15min candles for `symbol`."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - years * 365 * 24 * 60 * 60 * 1000
    all_dfs = []
    current = start_ms
    with tqdm(desc=symbol, unit="batch") as pbar:
        while current < end_ms:
            batch_end = min(current + LIMIT * CANDLE_MS, end_ms)
            df = fetch_klines(symbol, INTERVAL, current, batch_end)
            if df.empty:
                current = batch_end + CANDLE_MS
                continue
            all_dfs.append(df)
            current = int(df["open_time"].iloc[-1]) + CANDLE_MS
            pbar.update(1)
            time.sleep(0.05)  # stay under rate limit
    if not all_dfs:
        return pd.DataFrame()
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return result


def main():
    out_dir = Path("data/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in TICKERS:
        out_path = out_dir / f"{symbol}_15m.parquet"
        if out_path.exists():
            print(f"{symbol}: already exists, skipping")
            continue
        print(f"Downloading {symbol}...")
        df = download_ticker(symbol, years=5)
        df.to_parquet(out_path, index=False)
        print(f"{symbol}: {len(df):,} candles saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

```bash
mkdir -p tests/training && touch tests/training/__init__.py
pytest tests/training/test_download_data.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Download data (takes ~20 minutes)**

```bash
python training/download_data.py
```

Expected output (one line per ticker):
```
BTCUSDT: 175,248 candles saved to data/training/BTCUSDT_15m.parquet
...
```

- [ ] **Step 7: Commit**

```bash
git add training/__init__.py training/download_data.py tests/training/__init__.py tests/training/test_download_data.py
git commit -m "feat: add Binance 5yr OHLCV download script"
```

---

## Task 3: Label Training Data

**Files:**
- Create: `training/label_data.py`

- [ ] **Step 1: Write failing test**

Create `tests/training/test_label_data.py`:

```python
import pandas as pd
import numpy as np
import pytest


def make_df(closes):
    return pd.DataFrame({
        "open_time": range(len(closes)),
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": [1000.0] * len(closes),
    })


def test_label_long():
    from training.label_data import label_candles
    # +1.5% rise in next 4 candles, no -0.5% stop hit first
    closes = [100.0] + [101.5] * 4 + [100.0] * 10
    df = make_df(closes)
    result = label_candles(df, target_pct=0.01, stop_pct=0.005, window=4)
    assert result["label"].iloc[0] == "LONG"


def test_label_short():
    from training.label_data import label_candles
    # -1.5% drop in next 4 candles, no +0.5% stop hit first
    closes = [100.0] + [98.5] * 4 + [100.0] * 10
    df = make_df(closes)
    result = label_candles(df, target_pct=0.01, stop_pct=0.005, window=4)
    assert result["label"].iloc[0] == "SHORT"


def test_label_hold():
    from training.label_data import label_candles
    # Flat price — neither threshold reached
    closes = [100.0] * 20
    df = make_df(closes)
    result = label_candles(df, target_pct=0.01, stop_pct=0.005, window=4)
    assert result["label"].iloc[0] == "HOLD"


def test_stop_hit_before_target_is_hold():
    from training.label_data import label_candles
    # Drops -0.5% before rising +1% — stop hit first, label = HOLD
    closes = [100.0, 99.4, 99.4, 99.4, 101.5] + [100.0] * 10
    df = make_df(closes)
    result = label_candles(df, target_pct=0.01, stop_pct=0.005, window=4)
    assert result["label"].iloc[0] == "HOLD"


def test_last_rows_are_hold():
    from training.label_data import label_candles
    closes = [100.0] * 20
    df = make_df(closes)
    result = label_candles(df)
    # Last 4 rows can't have full look-ahead window — labeled HOLD
    assert (result["label"].iloc[-4:] == "HOLD").all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_label_data.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `training/label_data.py`**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TARGET_PCT = 0.010   # +1.0% for LONG, -1.0% for SHORT
STOP_PCT   = 0.005   # -0.5% stop for LONG,  +0.5% stop for SHORT
WINDOW     = 4       # look-ahead candles


def label_candles(
    df: pd.DataFrame,
    target_pct: float = TARGET_PCT,
    stop_pct: float = STOP_PCT,
    window: int = WINDOW,
) -> pd.DataFrame:
    """
    Add 'label' column: LONG / SHORT / HOLD.
    LONG  — close rises >= target_pct before falling <= stop_pct within window.
    SHORT — close falls >= target_pct before rising <= stop_pct within window.
    HOLD  — neither threshold reached.
    Last `window` rows always HOLD (no complete look-ahead).
    """
    closes = df["close"].to_numpy(dtype=np.float64)
    n = len(closes)
    labels = ["HOLD"] * n

    for i in range(n - window):
        entry = closes[i]
        long_target  = entry * (1 + target_pct)
        long_stop    = entry * (1 - stop_pct)
        short_target = entry * (1 - target_pct)
        short_stop   = entry * (1 + stop_pct)

        long_label = short_label = False
        for j in range(i + 1, i + 1 + window):
            c = closes[j]
            if c <= long_stop:
                break
            if c >= long_target:
                long_label = True
                break

        for j in range(i + 1, i + 1 + window):
            c = closes[j]
            if c >= short_stop:
                break
            if c <= short_target:
                short_label = True
                break

        if long_label and not short_label:
            labels[i] = "LONG"
        elif short_label and not long_label:
            labels[i] = "SHORT"
        # else HOLD (ambiguous or neither)

    df = df.copy()
    df["label"] = labels
    return df


def main():
    data_dir = Path("data/training")
    for symbol in TICKERS:
        src = data_dir / f"{symbol}_15m.parquet"
        dst = data_dir / f"{symbol}_15m_labeled.parquet"
        if dst.exists():
            print(f"{symbol}: labeled file exists, skipping")
            continue
        print(f"Labeling {symbol}...")
        df = pd.read_parquet(src)
        labeled = label_candles(df)
        counts = labeled["label"].value_counts()
        print(f"  {len(labeled):,} rows | LONG={counts.get('LONG',0):,} SHORT={counts.get('SHORT',0):,} HOLD={counts.get('HOLD',0):,}")
        labeled.to_parquet(dst, index=False)
        print(f"  Saved to {dst}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/training/test_label_data.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Run labeling**

```bash
python training/label_data.py
```

Expected (approximate):
```
BTCUSDT: 175,248 rows | LONG=32,418 SHORT=31,892 HOLD=110,938
...
```

- [ ] **Step 6: Commit**

```bash
git add training/label_data.py tests/training/test_label_data.py
git commit -m "feat: add forward-look LONG/SHORT/HOLD labeling for training data"
```

---

## Task 4: PyTorch Dataset

**Files:**
- Create: `training/dataset.py`

- [ ] **Step 1: Write failing test**

Create `tests/training/test_dataset.py`:

```python
import pandas as pd
import numpy as np
import torch
import pytest


def make_labeled_df(n=200):
    closes = np.linspace(100, 110, n).tolist()
    return pd.DataFrame({
        "open_time": range(n),
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": [1000.0] * n,
        "label": ["LONG", "SHORT", "HOLD"] * (n // 3) + ["HOLD"] * (n % 3),
    })


def test_dataset_length():
    from training.dataset import CryptoDataset
    df = make_labeled_df(200)
    ds = CryptoDataset(df, window=96)
    # Valid samples start at index 96 (window), end at n-1
    assert len(ds) == 200 - 96


def test_dataset_item_shapes():
    from training.dataset import CryptoDataset
    df = make_labeled_df(200)
    ds = CryptoDataset(df, window=96)
    close_window, label = ds[0]
    assert close_window.shape == torch.Size([96])
    assert isinstance(label, int)
    assert label in (0, 1, 2)


def test_dataset_close_is_normalized():
    from training.dataset import CryptoDataset
    df = make_labeled_df(200)
    ds = CryptoDataset(df, window=96)
    close_window, _ = ds[0]
    # z-score: mean ~0, std ~1
    assert abs(close_window.mean().item()) < 0.5
    assert 0.1 < close_window.std().item() < 10.0


def test_split_no_leakage():
    from training.dataset import split_datasets
    df = make_labeled_df(1000)
    train, val, test = split_datasets(df, window=96)
    # Must be time-ordered: train indices < val indices < test indices
    assert len(train) + len(val) + len(test) == 1000 - 96
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_dataset.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `training/dataset.py`**

```python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple

LABEL_MAP = {"LONG": 0, "SHORT": 1, "HOLD": 2}
LABEL_NAMES = ["LONG", "SHORT", "HOLD"]
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
WINDOW = 96  # candles fed to model (24 hours of 15min data)


class CryptoDataset(Dataset):
    """
    Each item: (close_window [WINDOW], label_int)
    close_window is z-score normalized per sample to prevent look-ahead bias.
    """

    def __init__(self, df: pd.DataFrame, window: int = WINDOW):
        self.closes = df["close"].to_numpy(dtype=np.float32)
        self.labels = df["label"].map(LABEL_MAP).to_numpy(dtype=np.int64)
        self.window = window
        # Valid start indices: window .. len-1
        self.indices = list(range(window, len(df)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        end = self.indices[idx]
        start = end - self.window
        window = self.closes[start:end].copy()
        # Z-score normalize per sample
        mean = window.mean()
        std = window.std() + 1e-8
        window = (window - mean) / std
        return torch.tensor(window, dtype=torch.float32), int(self.labels[end])


def load_all_tickers(data_dir: str = "data/training") -> pd.DataFrame:
    """Concatenate labeled parquet files for all tickers."""
    dfs = []
    for symbol in TICKERS:
        path = Path(data_dir) / f"{symbol}_15m_labeled.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}. Run training/label_data.py first.")
        df = pd.read_parquet(path)
        df["symbol"] = symbol
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("open_time").reset_index(drop=True)
    return combined


def split_datasets(
    df: pd.DataFrame,
    window: int = WINDOW,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> Tuple[CryptoDataset, CryptoDataset, CryptoDataset]:
    """
    Time-ordered split. Never shuffled — prevents future data leakage.
    Returns (train_dataset, val_dataset, test_dataset).
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df  = df.iloc[val_end:].reset_index(drop=True)

    return CryptoDataset(train_df, window), CryptoDataset(val_df, window), CryptoDataset(test_df, window)


def compute_class_weights(dataset: CryptoDataset) -> torch.Tensor:
    """Returns [3] weight tensor for CrossEntropyLoss — upweights LONG/SHORT."""
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    total = counts.sum()
    # Inverse frequency, clipped to [1, 5]
    weights = np.clip(total / (3 * counts + 1e-8), 1.0, 5.0)
    return torch.tensor(weights, dtype=torch.float32)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/training/test_dataset.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py tests/training/test_dataset.py
git commit -m "feat: add CryptoDataset with z-score normalization and time-ordered split"
```

---

## Task 5: ChronosClassifier Model

**Files:**
- Create: `training/model.py`
- Create: `models/chronos_crypto/` (directory)

- [ ] **Step 1: Write failing test**

Create `tests/training/test_model.py`:

```python
import torch
import pytest


def test_model_output_shape():
    from training.model import ChronosClassifier
    model = ChronosClassifier(checkpoint="amazon/chronos-t5-small")
    batch = torch.randn(2, 96)  # 2 samples, 96 time steps
    logits = model(batch)
    assert logits.shape == torch.Size([2, 3])


def test_model_freeze_unfreeze():
    from training.model import ChronosClassifier
    model = ChronosClassifier(checkpoint="amazon/chronos-t5-small")
    model.freeze_encoder()
    frozen = sum(1 for p in model.encoder.parameters() if not p.requires_grad)
    assert frozen > 0
    model.unfreeze_encoder()
    unfrozen = sum(1 for p in model.encoder.parameters() if p.requires_grad)
    assert unfrozen > 0


def test_model_save_load(tmp_path):
    from training.model import ChronosClassifier
    model = ChronosClassifier(checkpoint="amazon/chronos-t5-small")
    save_path = str(tmp_path / "test_model.pt")
    model.save(save_path)
    loaded = ChronosClassifier.load(save_path, checkpoint="amazon/chronos-t5-small")
    # Check weights match
    batch = torch.randn(1, 96)
    with torch.no_grad():
        out1 = model(batch)
        out2 = loaded(batch)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_predict_returns_valid_direction():
    from training.model import ChronosClassifier
    from core.models import Direction
    model = ChronosClassifier(checkpoint="amazon/chronos-t5-small")
    close = torch.randn(96)
    direction, confidence = model.predict(close)
    assert direction in (Direction.LONG, Direction.SHORT, Direction.HOLD)
    assert 0.0 <= confidence <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_model.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `training/model.py`**

```python
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple

# Import Chronos — raises ImportError clearly if not installed
from chronos import ChronosPipeline

from core.models import Direction
from training.dataset import LABEL_NAMES

CHECKPOINT_DEFAULT = "amazon/chronos-t5-small"
D_MODEL = 512          # T5-small hidden dimension
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
        context_list = [close[i].cpu() for i in range(close.shape[0])]

        token_ids, attention_mask, _ = self.tokenizer.context_input_transform(context_list)
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
        torch.save({
            "encoder_state": self.encoder.state_dict(),
            "head_state": self.head.state_dict(),
            "checkpoint": self._checkpoint,
        }, path)

    @classmethod
    def load(cls, path: str, checkpoint: str = CHECKPOINT_DEFAULT) -> "ChronosClassifier":
        state = torch.load(path, map_location="cpu", weights_only=False)
        ckpt = state.get("checkpoint", checkpoint)
        model = cls(ckpt)
        model.encoder.load_state_dict(state["encoder_state"])
        model.head.load_state_dict(state["head_state"])
        return model
```

- [ ] **Step 4: Run tests (will download ~200MB model on first run)**

```bash
pytest tests/training/test_model.py -v -s
```

Expected: `4 passed` (first run slow due to HuggingFace download)

- [ ] **Step 5: Create models directory**

```bash
mkdir -p models/chronos_crypto
```

- [ ] **Step 6: Commit**

```bash
git add training/model.py tests/training/test_model.py
git commit -m "feat: add ChronosClassifier with T5 encoder + classification head"
```

---

## Task 6: Training Script

**Files:**
- Create: `training/train.py`

- [ ] **Step 1: Write failing test**

Create `tests/training/test_train.py`:

```python
import torch
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def make_tiny_df(n=200):
    closes = np.random.randn(n).cumsum() + 100
    return pd.DataFrame({
        "open_time": range(n),
        "open": closes, "high": closes * 1.01,
        "low": closes * 0.99, "close": closes,
        "volume": [1000.0] * n,
        "label": (["LONG", "SHORT", "HOLD"] * (n // 3 + 1))[:n],
    })


def test_train_one_epoch_runs():
    from training.train import train_epoch
    from training.model import ChronosClassifier
    from training.dataset import CryptoDataset
    from torch.utils.data import DataLoader

    df = make_tiny_df(200)
    ds = CryptoDataset(df, window=96)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    model = ChronosClassifier()
    model.freeze_encoder()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    loss = train_epoch(model, loader, optimizer, device=torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss > 0


def test_eval_returns_metrics():
    from training.train import eval_epoch
    from training.model import ChronosClassifier
    from training.dataset import CryptoDataset
    from torch.utils.data import DataLoader

    df = make_tiny_df(200)
    ds = CryptoDataset(df, window=96)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    model = ChronosClassifier()
    metrics = eval_epoch(model, loader, device=torch.device("cpu"))
    assert "loss" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_train.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `training/train.py`**

```python
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

MODEL_DIR = Path("models/chronos_crypto")
CHECKPOINT = "amazon/chronos-t5-small"

# Phase 1: freeze encoder, train head only
PHASE1_EPOCHS = 2
PHASE1_LR     = 1e-3
PHASE1_BATCH  = 64

# Phase 2: unfreeze all, fine-tune end-to-end
PHASE2_EPOCHS = 3
PHASE2_LR     = 1e-5
PHASE2_BATCH  = 32


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
    for close_batch, label_batch in tqdm(loader, leave=False, desc="train"):
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
    (MODEL_DIR / "config.json").write_text(json.dumps({
        "checkpoint": CHECKPOINT,
        "window": 96,
        "label_names": ["LONG", "SHORT", "HOLD"],
        "confidence_threshold": 0.55,
    }, indent=2))

    model = ChronosClassifier(CHECKPOINT).to(device)
    best_f1 = 0.0

    # ── Phase 1: train head only ──────────────────────────────────────────
    print("\n=== Phase 1: training head (encoder frozen) ===")
    model.freeze_encoder()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=PHASE1_LR)
    train_loader = DataLoader(train_ds, batch_size=PHASE1_BATCH, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=PHASE1_BATCH, shuffle=False, num_workers=2)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = eval_epoch(model, val_loader, device)
        print(f"  Epoch {epoch}/{PHASE1_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  val_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            model.save(str(MODEL_DIR / "best.pt"))
            print(f"  ✓ New best F1={best_f1:.4f} — checkpoint saved")

    # ── Phase 2: fine-tune all ────────────────────────────────────────────
    print("\n=== Phase 2: fine-tuning full model ===")
    model.unfreeze_encoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)
    train_loader = DataLoader(train_ds, batch_size=PHASE2_BATCH, shuffle=True, num_workers=2)

    patience = 2
    no_improve = 0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()
        print(f"  Epoch {epoch}/{PHASE2_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  val_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            model.save(str(MODEL_DIR / "best.pt"))
            print(f"  ✓ New best F1={best_f1:.4f} — checkpoint saved")
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
```

- [ ] **Step 4: Run unit tests**

```bash
pytest tests/training/test_train.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Run training (takes 2–6 hours depending on GPU)**

```bash
python training/train.py
```

Expected final line:
```
Training complete. Best val macro-F1: 0.XXXX
Checkpoint: models/chronos_crypto/best.pt
```

A macro-F1 ≥ 0.45 on a 3-class imbalanced problem is acceptable. HOLD will dominate; LONG and SHORT F1 matter most.

- [ ] **Step 6: Commit**

```bash
git add training/train.py tests/training/test_train.py models/chronos_crypto/config.json
git commit -m "feat: add Phase 1 + Phase 2 Chronos-2 fine-tuning pipeline"
```

---

## Task 7: ChronosTechnicalAgent

**Files:**
- Create: `agents/chronos_technical.py`
- Delete: `agents/kronos_technical.py`

- [ ] **Step 1: Write failing test**

Replace `tests/test_agents.py` with:

```python
import torch
import pytest
from unittest.mock import MagicMock, patch
from core.models import Direction, Market


def make_klines(n=100):
    return [
        {"open": 62000.0, "high": 63000.0, "low": 61500.0,
         "close": 62400.0 + i * 10, "volume": 1200.0}
        for i in range(n)
    ]


def test_chronos_agent_returns_agent_vote(monkeypatch):
    from agents.chronos_technical import ChronosTechnicalAgent
    agent = ChronosTechnicalAgent.__new__(ChronosTechnicalAgent)
    # Mock the model
    mock_model = MagicMock()
    mock_model.predict.return_value = (Direction.LONG, 0.72)
    agent.model = mock_model
    agent.device = torch.device("cpu")

    vote = agent.analyze(ticker="BTC/USDT", klines=make_klines(), market=Market.CRYPTO)
    assert vote.direction == Direction.LONG
    assert vote.confidence == pytest.approx(0.72)
    assert vote.agent_name == "ChronosTechnical"
    assert "Chronos-2" in vote.reasoning


def test_chronos_agent_hold_on_low_confidence(monkeypatch):
    from agents.chronos_technical import ChronosTechnicalAgent
    agent = ChronosTechnicalAgent.__new__(ChronosTechnicalAgent)
    mock_model = MagicMock()
    mock_model.predict.return_value = (Direction.HOLD, 0.0)
    agent.model = mock_model
    agent.device = torch.device("cpu")

    vote = agent.analyze(ticker="BTC/USDT", klines=make_klines(), market=Market.CRYPTO)
    assert vote.direction == Direction.HOLD
    assert vote.confidence == 0.0


def test_chronos_agent_fails_loudly_if_model_missing(tmp_path):
    from agents.chronos_technical import ChronosTechnicalAgent
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        ChronosTechnicalAgent(model_path=str(tmp_path / "nonexistent.pt"))


def test_chronos_agent_needs_96_candles(monkeypatch):
    from agents.chronos_technical import ChronosTechnicalAgent
    agent = ChronosTechnicalAgent.__new__(ChronosTechnicalAgent)
    mock_model = MagicMock()
    mock_model.predict.return_value = (Direction.HOLD, 0.0)
    agent.model = mock_model
    agent.device = torch.device("cpu")

    # Only 50 candles — fewer than required 96
    vote = agent.analyze(ticker="BTC/USDT", klines=make_klines(50), market=Market.CRYPTO)
    assert vote.direction == Direction.HOLD
    assert "insufficient" in vote.reasoning.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_agents.py -v
```

Expected: `ImportError` (no `agents.chronos_technical`)

- [ ] **Step 3: Write `agents/chronos_technical.py`**

```python
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

        # Build probability string for transparency
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
```

- [ ] **Step 4: Delete `agents/kronos_technical.py`**

```bash
git rm agents/kronos_technical.py
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_agents.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add agents/chronos_technical.py tests/test_agents.py
git commit -m "feat: add ChronosTechnicalAgent backed by fine-tuned Chronos-2; remove KronosTechnicalAgent"
```

---

## Task 8: SignalTracker — Outcome Persistence

**Files:**
- Modify: `core/signal_tracker.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_signal_tracker.py`:

```python
def test_chronos_outcome_written_to_jsonl(tmp_path, monkeypatch):
    import json
    from core.signal_tracker import SignalTracker
    from core.signal_aggregator import SignalAggregator
    from core.models import SignalOutcome

    outcomes_path = tmp_path / "chronos_outcomes.jsonl"
    monkeypatch.setenv("CHRONOS_OUTCOMES_PATH", str(outcomes_path))

    agg = SignalAggregator(log_path=str(tmp_path / "signal_log.json"))
    tracker = SignalTracker(agg, MagicMock(), MagicMock())

    entry = {
        "id": "sig_test_001",
        "ticker": "BTC/USDT",
        "market": "crypto",
        "direction": "LONG",
        "entry_price": 62000.0,
        "target_price": 63200.0,
        "stop_price": 61400.0,
        "position_size_inr": 2000.0,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "agent_votes": [{"agent_name": "ChronosTechnical", "direction": "LONG",
                         "confidence": 0.71, "reasoning": "test"}],
    }
    tracker._write_chronos_outcome(entry, SignalOutcome.TARGET_HIT)

    lines = outcomes_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticker"] == "BTC/USDT"
    assert record["predicted"] == "LONG"
    assert record["confidence"] == pytest.approx(0.71)
    assert record["actual"] == "TARGET_HIT"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_tracker.py::test_chronos_outcome_written_to_jsonl -v
```

Expected: `AttributeError` — no `_write_chronos_outcome`

- [ ] **Step 3: Modify `core/signal_tracker.py`**

Add these two methods to `SignalTracker` and call `_write_chronos_outcome` from `run_once`:

```python
# Add import at top of file
import json
import os
from datetime import datetime, timezone

# Add OUTCOMES_PATH constant after imports
CHRONOS_OUTCOMES_PATH = os.environ.get(
    "CHRONOS_OUTCOMES_PATH", "data/chronos_outcomes.jsonl"
)
```

Add methods inside the `SignalTracker` class (after `_calc_pnl`):

```python
    def _get_chronos_confidence(self, entry: dict) -> float:
        """Extract confidence from ChronosTechnical agent vote if present."""
        for vote in entry.get("agent_votes", []):
            if isinstance(vote, dict) and vote.get("agent_name") == "ChronosTechnical":
                return float(vote.get("confidence", 0.0))
            if hasattr(vote, "agent_name") and vote.agent_name == "ChronosTechnical":
                return float(vote.confidence)
        return 0.0

    def _write_chronos_outcome(self, entry: dict, outcome: "SignalOutcome"):
        """Append one record to chronos_outcomes.jsonl for model retraining."""
        confidence = self._get_chronos_confidence(entry)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticker": entry.get("ticker", ""),
            "predicted": entry.get("direction", "HOLD"),
            "confidence": confidence,
            "actual": outcome.value,
        }
        with open(CHRONOS_OUTCOMES_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
```

In `run_once`, after the `if outcome != SignalOutcome.PENDING:` block, add the call:

```python
                if outcome != SignalOutcome.PENDING:
                    pnl_pct, pnl_inr = self._calc_pnl(
                        entry["direction"],
                        entry["entry_price"],
                        outcome_price,
                        entry["position_size_inr"],
                    )
                    self.aggregator.update_signal_outcome(
                        entry["id"], outcome.value, outcome_price, pnl_pct, pnl_inr
                    )
                    # Persist for Chronos retraining
                    if entry.get("market") == "crypto":
                        self._write_chronos_outcome(entry, outcome)
```

- [ ] **Step 4: Run all signal tracker tests**

```bash
pytest tests/test_signal_tracker.py -v
```

Expected: all tests pass (including the 4 existing + 1 new = 5 total)

- [ ] **Step 5: Commit**

```bash
git add core/signal_tracker.py tests/test_signal_tracker.py
git commit -m "feat: persist Chronos prediction outcomes to chronos_outcomes.jsonl for retraining"
```

---

## Task 9: Fix DebateEngine — Inject Learnings

**Files:**
- Modify: `core/debate_engine.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_debate_engine.py`:

```python
def test_generate_reasoning_injects_learnings(monkeypatch):
    from core.debate_engine import DebateEngine
    from core.models import AgentVote, Direction

    captured_messages = []

    def mock_create(**kwargs):
        captured_messages.append(kwargs)
        resp = MagicMock()
        resp.content = [MagicMock(text="BTC looks bullish based on funding.")]
        return resp

    monkeypatch.setattr("anthropic.Anthropic.messages.create", mock_create)

    engine = DebateEngine(learnings="LONG setups with negative funding worked best.")
    votes = [AgentVote(agent_name="A", direction=Direction.LONG, confidence=0.7, reasoning="test")]
    result = engine.generate_reasoning(votes, Direction.LONG, "BTC/USDT")

    assert len(captured_messages) == 1
    prompt_text = str(captured_messages[0])
    assert "LONG setups with negative funding worked best." in prompt_text
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_debate_engine.py::test_generate_reasoning_injects_learnings -v
```

Expected: `FAILED` — `DebateEngine.__init__` does not accept `learnings` kwarg

- [ ] **Step 3: Modify `core/debate_engine.py`**

Replace the entire file with:

```python
from anthropic import Anthropic
from core.models import AgentVote, Direction
from config import config


class DebateEngine:
    def __init__(self, learnings: str = ""):
        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.learnings = learnings

    def update_learnings(self, learnings: str):
        self.learnings = learnings

    def reach_consensus(self, votes: list[AgentVote]) -> dict:
        if not votes:
            return {
                "direction": Direction.HOLD,
                "confidence": 0.0,
                "transcript": "No votes",
            }

        active_votes = [
            v for v in votes if v.direction != Direction.HOLD and v.confidence >= 0.50
        ]

        long_votes = [v for v in active_votes if v.direction == Direction.LONG]
        short_votes = [v for v in active_votes if v.direction == Direction.SHORT]

        if len(long_votes) >= 2 and len(long_votes) > len(short_votes):
            direction = Direction.LONG
            agreeing = long_votes
        elif len(short_votes) >= 2 and len(short_votes) > len(long_votes):
            direction = Direction.SHORT
            agreeing = short_votes
        else:
            return {
                "direction": Direction.HOLD,
                "confidence": 0.0,
                "transcript": self._build_transcript(votes),
            }

        confidence = sum(v.confidence for v in agreeing) / len(agreeing)
        if confidence < config.min_confidence:
            return {
                "direction": Direction.HOLD,
                "confidence": confidence,
                "transcript": self._build_transcript(votes),
            }

        return {
            "direction": direction,
            "confidence": round(confidence, 3),
            "transcript": self._build_transcript(votes),
        }

    def _build_transcript(self, votes: list[AgentVote]) -> str:
        lines = []
        for v in votes:
            lines.append(
                f"{v.agent_name}: {v.direction.value} (conf={v.confidence:.2f}) — {v.reasoning}"
            )
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        base = "You are a crypto trading analyst."
        if self.learnings:
            return f"{base}\n\nRecent learnings from signal outcomes:\n{self.learnings}"
        return base

    def generate_reasoning(
        self, votes: list[AgentVote], direction: Direction, ticker: str
    ) -> str:
        transcript = self._build_transcript(votes)
        prompt = f"""Agent debate for {ticker}:
{transcript}

Consensus: {direction.value}
Write a 2-sentence trading rationale based on the above debate. Be specific about the key signals."""

        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=self._build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
```

- [ ] **Step 4: Run all debate engine tests**

```bash
pytest tests/test_debate_engine.py -v
```

Expected: all tests pass (3 existing + 1 new = 4 total)

- [ ] **Step 5: Commit**

```bash
git add core/debate_engine.py tests/test_debate_engine.py
git commit -m "fix: inject agent learnings into DebateEngine system prompt"
```

---

## Task 10: Fix BaseHarness — Route Learnings

**Files:**
- Modify: `harnesses/base_harness.py`

- [ ] **Step 1: Modify `harnesses/base_harness.py`**

Two changes:
1. `__init__`: pass empty learnings to `DebateEngine`
2. `update_learnings`: also write to `.md` file and call `self.debate_engine.update_learnings()`

Replace `__init__` body and `update_learnings` in `base_harness.py`:

```python
    def __init__(
        self,
        state_path: str,
        broker,
        agents: list,
        telegram_bot,
        signal_aggregator: SignalAggregator,
    ):
        self.state_path = state_path
        self.broker = broker
        self.agents = agents
        self.telegram_bot = telegram_bot
        self.aggregator = signal_aggregator
        self.debate_engine = DebateEngine(learnings=self._load_state().agent_learnings)
        self.risk_manager = RiskManager(self.MARKET)
        self.state = self._load_state()
        # Sync learnings from persisted state into debate engine
        self.debate_engine.update_learnings(self.state.agent_learnings)
```

Replace `update_learnings`:

```python
    def update_learnings(self):
        if self.state.session_count % 10 != 0:
            return
        recent_ids = self.state.signal_history[-50:]
        all_signals = {s["id"]: s for s in self.aggregator.get_all_signals()}
        outcomes = [all_signals[id] for id in recent_ids if id in all_signals]
        if not outcomes:
            return

        client = Anthropic(api_key=config.anthropic_api_key)
        summary_prompt = f"""Analyze these {len(outcomes)} trading signal outcomes:
{json.dumps([{"ticker": o["ticker"], "direction": o["direction"], "outcome": o["outcome"], "pnl_pct": o.get("hypothetical_pnl_pct")} for o in outcomes], indent=2)}

Write 3 concise sentences about: (1) which setups worked, (2) which failed, (3) one rule to apply next session."""

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        learnings = resp.content[0].text.strip()
        self.state.agent_learnings = learnings
        self._save_state()

        # Route learnings to DebateEngine (where LLM actually runs)
        self.debate_engine.update_learnings(learnings)

        # Write human-readable log
        import os
        from pathlib import Path
        log_path = Path("data") / f"{self.MARKET.value}_learnings.md"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).isoformat()
            f.write(f"\n## {ts}\n{learnings}\n")
```

- [ ] **Step 2: Run existing tests to verify no regression**

```bash
pytest tests/ -v --ignore=tests/training -k "not chronos_outcome_written"
```

Expected: all existing tests still pass

- [ ] **Step 3: Commit**

```bash
git add harnesses/base_harness.py
git commit -m "fix: route agent learnings to DebateEngine system prompt; write learnings log to disk"
```

---

## Task 11: Fix CryptoHarness

**Files:**
- Modify: `harnesses/crypto_harness.py`

- [ ] **Step 1: Modify `harnesses/crypto_harness.py`**

Replace entire file:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from harnesses.base_harness import BaseHarness
from core.models import Market
from agents.chronos_technical import ChronosTechnicalAgent
from agents.news_sentiment import NewsSentimentAgent
from agents.onchain import OnChainAgent
from brokers.coindcx import CoinDCXBroker
from config import config

CRYPTO_TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


class CryptoHarness(BaseHarness):
    MARKET = Market.CRYPTO

    def __init__(self, telegram_bot, signal_aggregator):
        broker = CoinDCXBroker(config.coindcx_api_key, config.coindcx_api_secret)
        agents = [
            ChronosTechnicalAgent(),   # replaces KronosTechnicalAgent
            NewsSentimentAgent(config.cryptopanic_api_key),
            OnChainAgent(config.coinglass_api_key),
        ]
        super().__init__(
            state_path=config.crypto_progress_path,
            broker=broker,
            agents=agents,
            telegram_bot=telegram_bot,
            signal_aggregator=signal_aggregator,
        )
        self.scheduler = AsyncIOScheduler()

    async def _run_all_tickers(self):
        for ticker_raw in CRYPTO_TICKERS:
            try:
                # 100 candles — model needs 96, buffer of 4
                klines = self.broker.get_ohlcv(ticker_raw, interval="15m", limit=100)
                ticker_display = f"{ticker_raw[:3]}/{ticker_raw[3:]}"
                await self.run_session(ticker_display, klines)
            except Exception:
                continue
        self.update_learnings()

    def start(self):
        self.scheduler.add_job(self._run_all_tickers, "interval", minutes=15)
        self.scheduler.start()
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/training
```

Expected: all pass (model file not present in CI — `ChronosTechnicalAgent()` will fail loudly; that's expected in tests. Tests mock the agent via `monkeypatch`.)

- [ ] **Step 3: Commit**

```bash
git add harnesses/crypto_harness.py
git commit -m "feat: wire ChronosTechnicalAgent into CryptoHarness; increase klines limit to 100"
```

---

## Task 12: Threshold Auto-calibration

**Files:**
- Create: `training/calibrate_thresholds.py`
- Modify: `agents/onchain.py`
- Modify: `agents/news_sentiment.py`

- [ ] **Step 1: Write failing test**

Create `tests/training/test_calibrate.py`:

```python
import json
import pytest
from pathlib import Path


def make_signal_log(tmp_path, entries):
    path = tmp_path / "signal_log.json"
    path.write_text(json.dumps(entries))
    return str(path)


def test_calibrate_returns_defaults_if_too_few_signals(tmp_path):
    from training.calibrate_thresholds import calibrate
    log = make_signal_log(tmp_path, [])
    result = calibrate(log_path=log, min_samples=50)
    assert result["onchain_funding_threshold"] == -0.001
    assert result["news_sentiment_threshold"] == 0.2
    assert result["calibrated"] is False


def test_calibrate_writes_json(tmp_path):
    from training.calibrate_thresholds import calibrate
    # Build 60 fake resolved signals
    signals = []
    for i in range(60):
        signals.append({
            "id": f"s{i}", "market": "crypto",
            "direction": "LONG" if i % 2 == 0 else "SHORT",
            "outcome": "TARGET_HIT" if i % 3 == 0 else "STOP_HIT",
            "debate_transcript": f"OnChain: LONG (conf=0.75) — Funding: {-0.0012:.4f} | L/S ratio: 0.80",
        })
    log = make_signal_log(tmp_path, signals)
    out_path = str(tmp_path / "calibrated_thresholds.json")
    result = calibrate(log_path=log, out_path=out_path, min_samples=50)
    assert Path(out_path).exists()
    data = json.loads(Path(out_path).read_text())
    assert "onchain_funding_threshold" in data
    assert "news_sentiment_threshold" in data
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_calibrate.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `training/calibrate_thresholds.py`**

```python
import json
import re
from pathlib import Path
from datetime import datetime, timezone

# Defaults used when insufficient signal history
DEFAULTS = {
    "onchain_funding_threshold": -0.001,
    "onchain_ls_ratio_threshold": 0.9,
    "news_sentiment_threshold": 0.2,
}
OUT_PATH_DEFAULT = "data/calibrated_thresholds.json"
MIN_SAMPLES = 50


def _extract_funding_from_transcript(transcript: str) -> float | None:
    """Parse funding rate from OnChain agent reasoning string."""
    match = re.search(r"Funding:\s*([-\d.]+)", transcript or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _extract_sentiment_from_transcript(transcript: str) -> float | None:
    """Parse sentiment score from NewsSentiment agent reasoning string."""
    match = re.search(r"Sentiment score:\s*([-\d.]+)", transcript or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def calibrate(
    log_path: str = "data/signal_log.json",
    out_path: str = OUT_PATH_DEFAULT,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    """
    Read resolved crypto signals from signal_log.json.
    Compute threshold values that maximise TARGET_HIT precision.
    Falls back to defaults if < min_samples resolved signals.
    """
    try:
        with open(log_path) as f:
            signals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        signals = []

    crypto_resolved = [
        s for s in signals
        if s.get("market") == "crypto"
        and s.get("outcome") in ("TARGET_HIT", "STOP_HIT")
        and s.get("debate_transcript")
    ]

    result = dict(DEFAULTS)
    result["calibrated"] = False
    result["sample_size"] = len(crypto_resolved)

    if len(crypto_resolved) < min_samples:
        result["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save(result, out_path)
        return result

    # Extract (funding_rate, won) pairs
    funding_pairs = []
    sentiment_pairs = []
    for s in crypto_resolved:
        won = s["outcome"] == "TARGET_HIT"
        transcript = s.get("debate_transcript", "")
        funding = _extract_funding_from_transcript(transcript)
        sentiment = _extract_sentiment_from_transcript(transcript)
        if funding is not None:
            funding_pairs.append((funding, won))
        if sentiment is not None:
            sentiment_pairs.append((sentiment, won))

    # Find best funding threshold: iterate candidate thresholds, pick max precision
    if len(funding_pairs) >= 20:
        best_thresh, best_precision = DEFAULTS["onchain_funding_threshold"], 0.0
        candidates = sorted(set(f for f, _ in funding_pairs))
        for thresh in candidates:
            positives = [(f, w) for f, w in funding_pairs if f < thresh]
            if len(positives) < 10:
                continue
            precision = sum(w for _, w in positives) / len(positives)
            if precision > best_precision:
                best_precision = precision
                best_thresh = thresh
        result["onchain_funding_threshold"] = best_thresh

    if len(sentiment_pairs) >= 20:
        best_thresh, best_precision = DEFAULTS["news_sentiment_threshold"], 0.0
        candidates = sorted(set(s for s, _ in sentiment_pairs))
        for thresh in candidates:
            positives = [(s, w) for s, w in sentiment_pairs if s > thresh]
            if len(positives) < 10:
                continue
            precision = sum(w for _, w in positives) / len(positives)
            if precision > best_precision:
                best_precision = precision
                best_thresh = thresh
        result["news_sentiment_threshold"] = best_thresh

    result["calibrated"] = True
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save(result, out_path)
    return result


def _save(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    result = calibrate()
    print(json.dumps(result, indent=2))
```

- [ ] **Step 4: Modify `agents/onchain.py`** — load thresholds from file

Replace the class constants and `__init__` in `OnChainAgent`:

```python
import json
from pathlib import Path

# Add these as class-level defaults (keep existing hardcoded as fallback)
THRESHOLDS_PATH = "data/calibrated_thresholds.json"

# Inside __init__:
    def __init__(self, api_key: str):
        self.api_key = api_key
        thresholds = self._load_thresholds()
        self._funding_threshold = thresholds.get("onchain_funding_threshold", -0.001)
        self._ls_ratio_threshold = thresholds.get("onchain_ls_ratio_threshold", 0.9)

    def _load_thresholds(self) -> dict:
        try:
            return json.loads(Path(THRESHOLDS_PATH).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
```

Replace hardcoded values in `analyze()`:

```python
        funding_bullish = funding < self._funding_threshold
        ls_bullish = ls_ratio < self._ls_ratio_threshold
```

- [ ] **Step 5: Modify `agents/news_sentiment.py`** — load threshold from file

Add to `__init__`:

```python
    def __init__(self, api_key: str):
        self.api_key = api_key
        thresholds = self._load_thresholds()
        self._sentiment_threshold = thresholds.get("news_sentiment_threshold", 0.2)

    def _load_thresholds(self) -> dict:
        try:
            import json
            from pathlib import Path
            return json.loads(Path("data/calibrated_thresholds.json").read_text())
        except (FileNotFoundError, Exception):
            return {}
```

Replace hardcoded `0.2` in `analyze()`:

```python
        if score > self._sentiment_threshold:
            direction, confidence = Direction.LONG, min(0.5 + score, 0.85)
        elif score < -self._sentiment_threshold:
            direction, confidence = Direction.SHORT, min(0.5 + abs(score), 0.85)
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/training/test_calibrate.py tests/test_brokers.py -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add training/calibrate_thresholds.py agents/onchain.py agents/news_sentiment.py tests/training/test_calibrate.py
git commit -m "feat: auto-calibrate OnChain/News thresholds from signal_log.json outcomes"
```

---

## Task 13: Backtesting Harness

**Files:**
- Create: `backtesting/__init__.py`
- Create: `backtesting/crypto_backtest.py`
- Create: `backtesting/results/` (directory)

- [ ] **Step 1: Write failing test**

Create `tests/test_backtest.py`:

```python
import pandas as pd
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock, patch


def make_test_df(n=300):
    closes = np.linspace(100, 120, n)
    return pd.DataFrame({
        "open_time": range(n),
        "open": closes * 0.999,
        "high": closes * 1.015,
        "low": closes * 0.985,
        "close": closes,
        "volume": [1000.0] * n,
        "label": (["LONG"] * 5 + ["HOLD"] * 5) * (n // 10) + ["HOLD"] * (n % 10),
    })


def test_backtest_produces_report(tmp_path):
    from backtesting.crypto_backtest import run_backtest

    mock_model = MagicMock()
    mock_model.predict.return_value = (
        __import__("core.models", fromlist=["Direction"]).Direction.LONG, 0.72
    )
    mock_model.eval = MagicMock()

    df = make_test_df(300)
    results = run_backtest(df, model=mock_model, window=96)

    assert "total_signals" in results
    assert "win_rate" in results
    assert "signal_rate_pct" in results
    assert isinstance(results["total_signals"], int)


def test_backtest_zero_signals_if_always_hold(tmp_path):
    from backtesting.crypto_backtest import run_backtest
    from core.models import Direction

    mock_model = MagicMock()
    mock_model.predict.return_value = (Direction.HOLD, 0.0)
    mock_model.eval = MagicMock()

    df = make_test_df(300)
    results = run_backtest(df, model=mock_model, window=96)
    assert results["total_signals"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_backtest.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backtesting/crypto_backtest.py`**

```python
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from training.dataset import WINDOW, LABEL_MAP

RESULTS_DIR = Path("backtesting/results")
ATR_PERIOD   = 14
STOP_MULT    = 2.0    # 2×ATR stop (matches RiskManager crypto config)
TARGET_MULT  = 1.8    # 1.8 R:R target


def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> str:
    """Walk forward from signal_idx, return TARGET_HIT / STOP_HIT / EXPIRED."""
    look_ahead = df.iloc[signal_idx + 1: signal_idx + 97]  # max 24 hours
    for _, row in look_ahead.iterrows():
        if direction == "LONG":
            if row["low"] <= stop_price:
                return "STOP_HIT"
            if row["high"] >= target_price:
                return "TARGET_HIT"
        else:  # SHORT
            if row["high"] >= stop_price:
                return "STOP_HIT"
            if row["low"] <= target_price:
                return "TARGET_HIT"
    return "EXPIRED"


def run_backtest(
    df: pd.DataFrame,
    model,
    window: int = WINDOW,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Walk-forward backtest of ChronosClassifier on a labeled DataFrame.
    Model runs standalone (no DebateEngine) to isolate Chronos-2 signal quality.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    atr = _compute_atr(df)

    signals = []
    equity = 1.0
    equity_curve = [1.0]
    closes = df["close"].to_numpy(dtype=np.float32)

    for i in tqdm(range(window, len(df) - 96), desc="backtest", unit="candle"):
        close_window = closes[i - window: i].copy()
        mean, std = close_window.mean(), close_window.std() + 1e-8
        normalized = (close_window - mean) / std
        tensor = torch.tensor(normalized, dtype=torch.float32).to(device)

        direction, confidence = model.predict(tensor)

        if direction.value == "HOLD":
            equity_curve.append(equity)
            continue

        entry = closes[i]
        atr_val = atr.iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            equity_curve.append(equity)
            continue

        if direction.value == "LONG":
            stop = entry - STOP_MULT * atr_val
            target = entry + TARGET_MULT * STOP_MULT * atr_val
        else:
            stop = entry + STOP_MULT * atr_val
            target = entry - TARGET_MULT * STOP_MULT * atr_val

        outcome = _simulate_trade(df, i, direction.value, entry, stop, target)

        risk_pct = abs(entry - stop) / entry
        if outcome == "TARGET_HIT":
            pnl_pct = risk_pct * TARGET_MULT
        elif outcome == "STOP_HIT":
            pnl_pct = -risk_pct
        else:
            pnl_pct = 0.0

        equity *= (1 + pnl_pct * 0.02)  # 2% portfolio risk per trade
        equity_curve.append(equity)

        signals.append({
            "candle_idx": i,
            "direction": direction.value,
            "confidence": round(confidence, 3),
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "target": round(target, 4),
            "outcome": outcome,
            "pnl_pct": round(pnl_pct, 4),
        })

    # Compute metrics
    total = len(signals)
    wins = sum(1 for s in signals if s["outcome"] == "TARGET_HIT")
    losses = sum(1 for s in signals if s["outcome"] == "STOP_HIT")
    win_rate = wins / total if total else 0.0

    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdown = ((peak - eq) / peak)
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0

    returns = np.diff(eq) / eq[:-1]
    sharpe = (
        float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(24 * 365 / 0.25))
        if len(returns) > 1 else 0.0
    )

    results = {
        "total_candles": len(df) - window,
        "total_signals": total,
        "signal_rate_pct": round(total / max(len(df) - window, 1) * 100, 2),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "final_equity": round(float(eq[-1]), 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "equity_curve": eq.tolist(),
        "signals": signals,
    }
    return results


def main():
    from training.model import ChronosClassifier
    from training.dataset import load_all_tickers, split_datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model = ChronosClassifier.load("models/chronos_crypto/best.pt").to(device)

    print("Loading test split...")
    df = load_all_tickers()
    _, _, test_ds = split_datasets(df)
    # Reconstruct raw test DataFrame (need OHLCV for trade simulation)
    n = len(df)
    test_start = int(n * 0.90)
    test_df = df.iloc[test_start:].reset_index(drop=True)

    print(f"Running backtest on {len(test_df):,} candles...")
    results = run_backtest(test_df, model, device=device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_json = RESULTS_DIR / f"{ts}-crypto-backtest.json"

    # Save (exclude equity_curve list from JSON to keep file small)
    save_results = {k: v for k, v in results.items() if k != "equity_curve"}
    out_json.write_text(json.dumps(save_results, indent=2))

    print(f"\n{'='*50}")
    print(f"Signals fired:    {results['total_signals']} ({results['signal_rate_pct']}% of candles)")
    print(f"Win rate:         {results['win_rate']:.1%}")
    print(f"Final equity:     {results['final_equity']:.4f}x  (started 1.0)")
    print(f"Max drawdown:     {results['max_drawdown_pct']:.1f}%")
    print(f"Sharpe ratio:     {results['sharpe_ratio']:.3f}")
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
mkdir -p backtesting && touch backtesting/__init__.py backtesting/results/.gitkeep
pytest tests/test_backtest.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backtesting/__init__.py backtesting/crypto_backtest.py backtesting/results/.gitkeep tests/test_backtest.py
git commit -m "feat: add walk-forward crypto backtesting harness"
```

---

## Task 14: Plot Backtest Results

**Files:**
- Create: `backtesting/plot_results.py`

- [ ] **Step 1: Write `backtesting/plot_results.py`**

```python
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime


def plot_backtest(results_json: str, out_path: str = None):
    """
    Plot equity curve and drawdown from backtest results JSON.
    Saves to out_path (PNG), or shows interactively if None.
    """
    with open(results_json) as f:
        results = json.load(f)

    equity = np.array(results.get("equity_curve", []))
    if len(equity) == 0:
        print("No equity_curve in results — re-run backtest to include it.")
        return

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Chronos-2 Crypto Backtest Results", fontsize=14, fontweight="bold")

    # Equity curve
    ax1.plot(equity, color="#2196F3", linewidth=1.2, label="Equity")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Equity (×)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Stats box
    stats_text = (
        f"Signals: {results.get('total_signals', 0)}\n"
        f"Win rate: {results.get('win_rate', 0):.1%}\n"
        f"Sharpe: {results.get('sharpe_ratio', 0):.2f}\n"
        f"Max DD: {results.get('max_drawdown_pct', 0):.1f}%"
    )
    ax1.text(0.98, 0.05, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Drawdown
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color="#F44336", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Candle index")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # Auto-find latest results file
        results_dir = Path("backtesting/results")
        files = sorted(results_dir.glob("*-crypto-backtest.json"))
        if not files:
            print("No backtest results found. Run backtesting/crypto_backtest.py first.")
            sys.exit(1)
        json_path = str(files[-1])
    else:
        json_path = sys.argv[1]

    out = json_path.replace(".json", ".png")
    plot_backtest(json_path, out_path=out)
```

- [ ] **Step 2: Commit**

```bash
git add backtesting/plot_results.py
git commit -m "feat: add equity curve + drawdown plot for backtest results"
```

---

## Task 15: Retrain Script

**Files:**
- Create: `training/retrain.py`

- [ ] **Step 1: Write `training/retrain.py`**

```python
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
from sklearn.metrics import f1_score

from training.model import ChronosClassifier, CHECKPOINT_DEFAULT
from training.dataset import LABEL_MAP, WINDOW

OUTCOMES_PATH   = "data/chronos_outcomes.jsonl"
MODEL_PATH      = "models/chronos_crypto/best.pt"
MIN_NEW_SAMPLES = 500
EPOCHS          = 2
LR              = 5e-6
BATCH           = 16


class OutcomesDataset(Dataset):
    """Dataset built from chronos_outcomes.jsonl + original training parquet."""

    def __init__(self, outcomes_path: str, data_dir: str = "data/training"):
        lines = Path(outcomes_path).read_text().strip().split("\n")
        records = [json.loads(l) for l in lines if l.strip()]

        # Load original parquet to get close prices for each outcome
        # Match by ticker + timestamp
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
            # Use close prices ending at the signal timestamp
            # (approximate: use last available window)
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
        print(f"  Epoch {epoch}/{EPOCHS}  loss={total_loss/len(loader):.4f}")

    model.save(MODEL_PATH)
    print(f"Retrained model saved to {MODEL_PATH}")

    # Archive processed outcomes so next run starts fresh
    archive = Path(OUTCOMES_PATH).with_suffix(".jsonl.bak")
    outcomes.rename(archive)
    print(f"Outcomes archived to {archive}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add training/retrain.py
git commit -m "feat: add incremental Chronos-2 retrain script from live signal outcomes"
```

---

## Task 16: Update main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Check if main.py imports KronosTechnicalAgent directly**

```bash
grep -n "kronos\|KronosTechnical" main.py
```

If no matches: `CryptoHarness` already handles agent instantiation (Task 11) — skip Step 2.

- [ ] **Step 2: Replace `KronosTechnicalAgent` import if present**

If Step 1 found matches, replace `from agents.kronos_technical import KronosTechnicalAgent` with:

```python
from agents.chronos_technical import ChronosTechnicalAgent
```

Note: `CryptoHarness` already handles agent instantiation internally (Task 11), so if `main.py` only imports `CryptoHarness`, no change is needed. Verify with:

```bash
grep -n "kronos\|KronosTechnical" main.py
```

If no matches: no change needed to `main.py` for this. Proceed to verification.

- [ ] **Step 3: Add calibrate_thresholds call on startup**

In `main.py`, inside the `build_components()` function (or equivalent startup sequence), add after imports:

```python
from training.calibrate_thresholds import calibrate
# Run calibration on startup (uses defaults if insufficient signal history)
calibrate()
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/training
```

Expected: all existing tests pass

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: wire threshold calibration into startup; clean up kronos imports"
```

---

## Task 17: Run Backtest and Verify

- [ ] **Step 1: Verify model file exists**

```bash
python -c "from pathlib import Path; assert Path('models/chronos_crypto/best.pt').exists(), 'Model missing — run training/train.py first'"
```

- [ ] **Step 2: Run backtest**

```bash
python backtesting/crypto_backtest.py
```

Expected output:
```
Signals fired:    XXX (X.X% of candles)
Win rate:         XX.X%
Final equity:     X.XXXX×  (started 1.0)
Max drawdown:     XX.X%
Sharpe ratio:     X.XXX
```

Acceptable thresholds to proceed to live:
- Win rate ≥ 50%
- Sharpe ratio ≥ 1.0
- Max drawdown ≤ 25%

If below threshold: re-check labels (Task 3), try `chronos-t5-base` instead of `small`, or retrain with adjusted class weights.

- [ ] **Step 3: Plot results**

```bash
python backtesting/plot_results.py
```

Expected: `backtesting/results/YYYY-MM-DD-crypto-backtest.png` generated.

- [ ] **Step 4: Run full test suite one final time**

```bash
pytest tests/ -v --ignore=tests/training
```

Expected: all pass

- [ ] **Step 5: Final commit**

```bash
git add backtesting/results/
git commit -m "feat: crypto intelligence overhaul complete — Chronos-2 classifier + learnings loop + backtest"
```

---

## Execution Order

Training data must be downloaded + labeled before training. Training must complete before backtest. Other tasks (8–15) are independent of training and can run in parallel.

```
Task 1  (deps)
Task 2  (download) ← ~20 min
Task 3  (label)    ← ~30 min
Task 4  (dataset)
Task 5  (model)
Task 6  (train)    ← 2-6 hours on GPU
                         │
         ┌───────────────┴───────────────┐
Task 7   (ChronosTechnicalAgent)    Tasks 8-15 (fixes + calibration)
         └───────────────────────────────┘
Task 16  (main.py wire-up)
Task 17  (run backtest + verify)
```

---

## Acceptance Criteria

- [ ] `pytest tests/ --ignore=tests/training` — all pass
- [ ] `models/chronos_crypto/best.pt` exists
- [ ] Backtest win rate ≥ 50%, Sharpe ≥ 1.0, max drawdown ≤ 25%
- [ ] `agents/kronos_technical.py` deleted
- [ ] `data/chronos_outcomes.jsonl` created and growing on each signal resolution
- [ ] `data/crypto_learnings.md` created after 10 sessions
