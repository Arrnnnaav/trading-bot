import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
    ax1.text(
        0.98,
        0.05,
        stats_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

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
        results_dir = Path("backtesting/results")
        files = sorted(results_dir.glob("*-crypto-backtest.json"))
        if not files:
            print(
                "No backtest results found. Run backtesting/crypto_backtest.py first."
            )
            sys.exit(1)
        json_path = str(files[-1])
    else:
        json_path = sys.argv[1]

    out = json_path.replace(".json", ".png")
    plot_backtest(json_path, out_path=out)
