"""Generate the held-out comparison figures from the Checkpoint 2 CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

METHODS = {
    "so volume": ("Só volume", "#657688"),
    "vol+bonus tom": ("Volume + tom", "#19618A"),
    "wav2vec 2.0": ("wav2vec 2.0", "#AC532C"),
}
CLASSES = {
    "finalizacao no alvo": "Finalização no alvo",
    "finalizacao para fora": "Finalização para fora",
    "gol": "Gol",
    "penalti": "Pênalti",
    "expulsao": "Expulsão",
}


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def label_pct(value):
    return f"{value:.0%}"


def compare_metrics(rows, dest):
    rows = {
        (row["selecao"], row["metodo"]): row
        for row in rows if row["alvo"] == "lances"
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), sharey=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.22, top=.76, wspace=.14)
    fig.text(.08, .923, "Detecção de lances por áudio", fontsize=20, weight="bold")
    annotated = rows["todos", next(iter(METHODS))]["anotados"]
    fig.text(.08, .855, f"Avaliação interna · {annotated} lances anotados · tolerância ±15 s", fontsize=11)
    x = np.arange(3)
    width = .235
    ymax = max(float(row[field]) for row in rows.values() for field in ("precisao", "revocacao", "f1"))
    for ax, (selection, title) in zip(axes, (("todos", "Todos os alertas"), ("top3", "Até 3 alertas por tempo"))):
        for i, (name, (label, color)) in enumerate(METHODS.items()):
            row = rows[selection, name]
            values = [float(row[field]) for field in ("precisao", "revocacao", "f1")]
            bars = ax.bar(x + (i - 1) * width, values, width * .9, label=label, color=color)
            ax.bar_label(bars, labels=[label_pct(v) for v in values], padding=3, fontsize=9)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=10)
        ax.set_xticks(x, ("Precisão", "Revocação", "F1"))
        ax.set_ylim(0, ymax * 1.23)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(.5, .035))
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def compare_classes(rows, dest):
    rows = {
        (row["classe"], row["metodo"]): row for row in rows if row["selecao"] == "todos"
    }
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    fig.subplots_adjust(left=.205, right=.96, bottom=.19, top=.78)
    fig.text(.04, .92, "Onde os métodos acertam", fontsize=20, weight="bold")
    fig.text(.04, .858, "Acertos sobre anotações de cada tipo · avaliação interna · todos os alertas", fontsize=11)
    y = np.arange(len(CLASSES))
    height = .23
    for i, (name, (label, color)) in enumerate(METHODS.items()):
        group = [rows[family, name] for family in CLASSES]
        values = [float(row["revocacao"] or 0) for row in group]
        bars = ax.barh(y + (i - 1) * height, values, height * .89, label=label, color=color)
        counts = [f'{row["acertos"]}/{row["anotados"]}' if row["anotados"] != "0" else "" for row in group]
        ax.bar_label(bars, labels=counts, padding=4, fontsize=9)
    for i, family in enumerate(CLASSES):
        if rows[family, next(iter(METHODS))]["anotados"] == "0":
            ax.text(.015, i, "Sem anotações no lote", va="center", color="#536171", fontsize=10)
    ax.set_yticks(y, list(CLASSES.values()))
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=.22)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(.54, .025))
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avaliacao", type=Path, default=Path("outputs/checkpoint-2/avaliacao"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/checkpoint-2/figuras"))
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "text.color": "#182837",
        "axes.labelcolor": "#182837", "axes.titlecolor": "#182837",
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.edgecolor": "#B3BDC7", "figure.facecolor": "#F8FAFC",
    }):
        compare_metrics(read_csv(args.avaliacao / "avaliacao_metricas.csv"), args.saida / "comparacao.png")
        compare_classes(read_csv(args.avaliacao / "avaliacao_por_classe.csv"), args.saida / "por-tipo.png")


if __name__ == "__main__":
    main()
