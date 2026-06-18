import sys
sys.path.insert(0, "D:/PROJECTS/trading-bot")
from training.train_kronos_india import train
import time

print("Starting Kronos GPU training...", flush=True)
t0 = time.time()
model, best_f1 = train(device_str="cuda")
elapsed = time.time() - t0
print(f"COMPLETE: best_val_f1={best_f1:.4f}  time={elapsed:.0f}s", flush=True)
