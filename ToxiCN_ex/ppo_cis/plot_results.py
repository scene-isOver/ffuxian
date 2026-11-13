"""Plot the PPO-CIS comparison table as F1 / latency / classifier-call charts.

Reads ``artifacts/reports/ppo_cis_results.csv`` (produced by ``train_ppo``) and
writes a three-panel PNG to ``artifacts/reports/ppo_cis_comparison.png``.

Usage (from the ToxiCN_ex directory):
    python -m ppo_cis.plot_results
    python -m ppo_cis.plot_results --csv path/to/results.csv --out chart.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")           # headless-safe: works on servers without a display
import matplotlib.pyplot as plt

from .config import REPORT_DIR

# 默认读取 train_ppo 写出的 CSV，默认输出到 reports 目录。
DEFAULT_CSV = REPORT_DIR / "ppo_cis_results.csv"
DEFAULT_OUT = REPORT_DIR / "ppo_cis_comparison.png"


# 读 CSV：抽方法名和 F1 / 平均延迟 / 平均调用数三列，缺值或非数字记为 NaN。
def load_rows(path: Path) -> Tuple[List[str], List[float], List[float], List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")

    def column(name: str) -> List[float]:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(name, "")))
            except (TypeError, ValueError):
                values.append(float("nan"))
        return values

    methods = [row["method"] for row in rows]
    return methods, column("f1"), column("avg_latency_ms"), column("avg_classifiers")


# 画单个柱状图面板：PPO-CIS 用红色突出，其余基线统一蓝色，柱顶标数值。
def bar_panel(ax, methods: List[str], values: List[float], title: str,
              ylabel: str, fmt: str) -> None:
    colors = ["#d62728" if name == "PPO-CIS" else "#4c72b0" for name in methods]
    bars = ax.bar(range(len(methods)), values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        if value == value:      # skip NaN
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt.format(value), ha="center", va="bottom", fontsize=8)


# 三联图：F1 / 平均延迟 / 平均调用数，保存为 PNG。
def make_plots(csv_path: Path = DEFAULT_CSV, out_path: Path = DEFAULT_OUT,
               dpi: int = 160) -> Path:
    """Render the three comparison panels and return the PNG path."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {csv_path}\n"
            "Run `python -m ppo_cis.train_ppo` (or `run_all --step ppo`) first."
        )

    methods, f1, latency, calls = load_rows(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bar_panel(axes[0], methods, f1, "F1 score", "F1", "{:.4f}")
    axes[0].set_ylim(0, 1.05)
    bar_panel(axes[1], methods, latency, "Average latency", "milliseconds", "{:.1f}")
    bar_panel(axes[2], methods, calls, "Average classifier calls", "calls", "{:.2f}")
    fig.suptitle("PPO-CIS two-stage cascade vs. static baselines on ToxiCN")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PPO-CIS comparison charts")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dpi", default=160, type=int)
    args = parser.parse_args()
    print(f"Saved: {make_plots(Path(args.csv), Path(args.out), args.dpi)}")


if __name__ == "__main__":
    main()
