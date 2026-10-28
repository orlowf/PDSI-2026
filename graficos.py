"""Chart the aggregated metrics written by avaliar_lote.py.

Small multiples: rows = precision / recall / F1, columns = goals / key events,
grouped bars = score rule, x = top-k per half.

Usage:
    uv run graficos.py outputs/avaliacao_holdout_metricas.csv --title "6 jogos sorteados do valid"
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference categorical palette, slots 1-3 (validated all-pairs for small multiples).
VARIANT_STYLE = {
    "min(vol,tom)": ("min(volume, tom)", "#2a78d6"),
    "so volume": ("só volume", "#eb6834"),
    "vol+bonus tom": ("volume + bônus de tom", "#1baf7a"),
}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TARGETS = {"goal": "Gols", "key": "Eventos-chave"}
METRICS = {"precision": "Precisão", "recall": "Revocação", "f1": "F1"}
K_ORDER = ["1", "3", "5", "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default=None, help="PNG path (default: next to the CSV)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.metrics_csv)))
    val = {(r["variant"], r["target"], r["k"], m): float(r[m]) for r in rows for m in METRICS}
    gt = {r["target"]: int(r["ground_truth"]) for r in rows}

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2})
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharey=True, facecolor=SURFACE)
    x = np.arange(len(K_ORDER))
    width = 0.26

    for row, (metric, metric_label) in enumerate(METRICS.items()):
        for col, (target, target_label) in enumerate(TARGETS.items()):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            for i, (variant, (label, color)) in enumerate(VARIANT_STYLE.items()):
                heights = [val[(variant, target, k, metric)] for k in K_ORDER]
                bars = ax.bar(x + (i - 1) * width, heights, width, color=color, label=label,
                              edgecolor=SURFACE, linewidth=2, zorder=3)
                if metric == "f1":  # selective direct labels: only the headline metric
                    ax.bar_label(bars, fmt="%.2f", fontsize=8, color=TEXT_2, padding=2)
            ax.set_ylim(0, 1)
            ax.set_xticks(x, ["top-1", "top-3", "top-5", "todas"])
            ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(length=0)
            if row == 0:
                ax.set_title(f"{target_label} ({gt[target]} no gabarito)", fontsize=12, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(args.title or "Comparação das regras de score", fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, "Micro-média sobre todos os tempos · tolerância ±15 s · top-k contado em cada tempo",
             ha="center", color=TEXT_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))

    out = Path(args.out) if args.out else Path(args.metrics_csv).with_suffix(".png")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"chart: {out}")


if __name__ == "__main__":
    main()
