"""Does the detector find goals better when the commentator gets more excited?

Joins outputs/emocao_jogos.csv (excitement per game) with the per-half goal ranks
written by avaliar_lote.py, and charts goal recall against how many dB the voice
rises at goals.

Usage:
    uv run cruzar_emocao.py outputs/avaliacao_lote.csv outputs/avaliacao_holdout.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graficos import GRID, SURFACE, TEXT, TEXT_2, VARIANT_STYLE

RULE = "vol+bonus tom"  # current default combine_score
EXCITEMENT = ["peak_rise_db", "goal_rise_db", "shouts_per_90"]


def spearman(a, b):
    rank = lambda x: np.argsort(np.argsort(x))
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def short_name(game):
    """'2015-04-11 - 19-30 Burnley 0 - 1 Arsenal' -> 'Burnley 0-1 Arsenal '15'."""
    return f"{game.split(' ', 3)[-1].replace(' - ', '-')} '{game[2:4]}"


def place_labels(ax, x, y, labels, min_px=11):
    """Greedy: try a few offsets per point and keep the first that doesn't collide with an earlier label."""
    offsets = [(6, 3, "left"), (6, -9, "left"), (6, 13, "left"), (6, -19, "left"), (6, -29, "left")]
    placed = []  # (x_px, y_px, ha) of placed label anchors
    for xi, yi, text in sorted(zip(x, y, labels), key=lambda t: (-t[1], t[0])):
        px, py = ax.transData.transform((xi, yi))
        for dx, dy, ha in offsets:
            ax_, ay = px + dx * 1.8, py + dy * 1.8  # points -> px at 130 dpi
            width = 6.5 * len(text)  # rough label width in px at fontsize 7.5
            x0, x1 = (ax_, ax_ + width) if ha == "left" else (ax_ - width, ax_)
            if all(abs(ay - qy) > min_px * 1.8 or x1 < q0 or x0 > q1 for q0, q1, qy in placed):
                break
        placed.append((x0, x1, ay))
        leader = None if dy == 3 else {"arrowstyle": "-", "color": TEXT_2, "lw": 0.6, "shrinkA": 0, "shrinkB": 4}
        ax.annotate(text, (xi, yi), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=7.5, color=TEXT_2, arrowprops=leader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ranks_csvs", nargs="+")
    ap.add_argument("--excitement", default="outputs/emocao_jogos.csv")
    ap.add_argument("--out", default="outputs/emocao_vs_deteccao.csv")
    args = ap.parse_args()

    # found[game][variant] = [goals in top-3, goals found at all, goals]
    found = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for path in args.ranks_csvs:
        for r in csv.DictReader(open(path)):
            for variant in VARIANT_STYLE:
                ranks = r[f"ranks {variant}"].split()
                acc = found[r["game"]][variant]
                acc[0] += sum(x != "-" and int(x) <= 3 for x in ranks)
                acc[1] += sum(x != "-" for x in ranks)
                acc[2] += len(ranks)

    rows = []
    for e in csv.DictReader(open(args.excitement)):
        row = {"game": e["game"], **{k: float(e[k]) for k in EXCITEMENT}}
        for variant in VARIANT_STYLE:
            top3, any_, n = found[e["game"]][variant]
            row[f"recall@3 {variant}"] = round(top3 / n, 2)
            row[f"recall {variant}"] = round(any_ / n, 2)
        row["goals"] = found[e["game"]][RULE][2]
        rows.append(row)
    rows.sort(key=lambda r: -r["goal_rise_db"])

    print(f"rule: {RULE}\n{'game':52s} {'goals':>5} {'goal dB':>8} {'recall@3':>9} {'recall':>7}")
    for r in rows:
        print(f"{r['game'][:52]:52s} {r['goals']:>5} {r['goal_rise_db']:>8} "
              f"{r[f'recall@3 {RULE}']:>9} {r[f'recall {RULE}']:>7}")
    print("\nSpearman between excitement and goal recall (12 games)")
    for variant in VARIANT_STYLE:
        for k in ("recall@3", "recall"):
            cells = [f"{x}={spearman([r[x] for r in rows], [r[f'{k} {variant}'] for r in rows]):+.2f}" for x in EXCITEMENT]
            print(f"  {variant:14s} {k:9s} " + "  ".join(cells))

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Scatter: one point per game, current rule only (single series, so no legend).
    color = VARIANT_STYLE[RULE][1]
    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, facecolor=SURFACE)
    x = np.array([r["goal_rise_db"] for r in rows])
    for ax, (k, title) in zip(axes, (("recall@3", "Gols no top-3 do tempo"), ("recall", "Gols achados (todas as detecções)"))):
        y = np.array([r[f"{k} {RULE}"] for r in rows])
        ax.set_facecolor(SURFACE)
        ax.scatter(x, y, s=[30 + 12 * r["goals"] for r in rows], color=color, edgecolor=SURFACE, lw=2, zorder=3)
        ax.set_xlim(2, 16)
        ax.set_ylim(-0.1, 1.08)
        place_labels(ax, x, y, [short_name(r["game"]) for r in rows])
        rho = spearman(x, y)
        ax.set_title(f"{title}\nSpearman = {rho:+.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("quanto a voz sobe nos gols (dB acima da mediana)")
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
    axes[0].set_ylabel("revocação de gols")
    fig.suptitle(f"Empolgação da narração × detecção de gols (regra: {VARIANT_STYLE[RULE][0]})",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "Um ponto por jogo · tamanho = nº de gols · tolerância ±15 s", ha="center",
             color=TEXT_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=130, facecolor=SURFACE)
    print(f"\ntable: {out}\nchart: {png}")


if __name__ == "__main__":
    main()
