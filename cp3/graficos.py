"""Figuras do teste do Checkpoint 3 (PNG para os slides, fundo claro)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import rankdata

from avaliar_arousal import FAMILIES
from cp3.conjunto import PASSO_S
from cp3.posproc import suavizar

SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # validado: 3 cores, todos os pares, modo claro
COR_CP3, COR_CP2 = SERIES[0], MUDO

NOMES = {"classico": "LightGBM", "tcn": "TCN 1D", "cnn2d": "CNN 2D",
         "vol+bonus tom": "Regra (CP1)", "wav2vec 4 s": "Wav2vec (CP2)", "fusao": "Fusão (CP2)"}
CLASSES = {"gol": "Gol", "finalizacao no alvo": "Finalização no alvo", "finalizacao para fora": "Finalização para fora",
           "penalti": "Pênalti", "expulsao": "Expulsão"}
ROTULO_EVENTO = {"Goal": "gol", "Penalty": "pênalti", "Shots on target": "no alvo",
                 "Shots off target": "para fora", "Red card": "expulsão", "Yellow->red card": "expulsão"}


def _estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO_2, "xtick.color": MUDO, "ytick.color": TEXTO_2,
        "text.color": TEXTO, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })


def _cor(metodo):
    return COR_CP2 if metodo in ("vol+bonus tom", "wav2vec 4 s", "fusao") else COR_CP3


def _legenda_grupos(fig):
    fig.legend(handles=[Patch(color=COR_CP3, label="Treinado do zero (CP3)"), Patch(color=COR_CP2, label="Checkpoint 1 e 2")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))


def _barras(ax, rotulos, valores, cores, titulo, formato):
    y = np.arange(len(rotulos))[::-1]
    ax.barh(y, valores, height=0.6, color=cores)
    ax.set_yticks(y, rotulos)
    ax.set_title(titulo, loc="left", fontsize=12, color=TEXTO)
    ax.grid(axis="y", visible=False)
    limite = max(max(valores) * 1.25, 1e-6)
    ax.set_xlim(0, limite)
    for yi, v in zip(y, valores):
        ax.text(v + limite * 0.02, yi, formato(v), va="center", fontsize=10, color=TEXTO_2)


def _painel_metricas(destino, linhas, titulo):
    rotulos = [NOMES.get(l["metodo"], l["metodo"]) for l in linhas]
    cores = [_cor(l["metodo"]) for l in linhas]
    fig, eixos = plt.subplots(1, 3, figsize=(13, 0.55 * len(linhas) + 1.8), sharey=True)
    for ax, (chave, nome) in zip(eixos, (("precisao", "Precisão"), ("revocacao", "Recall"), ("f1", "F1"))):
        formato = (lambda v: f"{v:.3f}") if chave == "f1" else (lambda v: f"{100 * v:.1f}%")
        _barras(ax, rotulos, [l[chave] for l in linhas], cores, nome, formato)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=13, color=TEXTO)
    _legenda_grupos(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def _linha_do_tempo(destino, tempos, curvas, bases):
    modelos = list(curvas)
    i = max(range(len(tempos)), key=lambda k: (any(r == "Goal" for _, r in tempos[k].eventos), len(tempos[k].eventos)))
    t = tempos[i]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    minutos = (np.arange(t.n_passos) + 0.5) * PASSO_S / 60
    for cor, nome in zip(SERIES, modelos):
        curva = suavizar(curvas[nome][i], PASSO_S)
        posicao = 100 * (rankdata(curva) - 1) / max(len(curva) - 1, 1)
        ax.plot(minutos, posicao, color=cor, lw=1.2, label=NOMES.get(nome, nome))
        ax.text(minutos[-1] + 0.3, posicao[-1], NOMES.get(nome, nome), color=TEXTO_2, fontsize=9, va="center")
    for segundo, rotulo in t.eventos:
        ax.axvline(segundo / 60, color=TEXTO_2, lw=0.8, ls=(0, (3, 3)))
        ax.text(segundo / 60, 103, ROTULO_EVENTO.get(rotulo, rotulo), rotation=90, fontsize=8, color=TEXTO_2,
                ha="center", va="bottom")
    if "fusao" in bases:
        alertas = next(p[3] for p in bases["fusao"] if (p[0], p[1]) == (t.jogo, t.numero))
        ax.plot([a / 60 for a, _ in alertas], [-6] * len(alertas), "^", ms=8, color=MUDO, label="Alertas da fusão (CP2)")
    ax.set_ylim(-10, 100)
    ax.set_xlabel("minuto do tempo de jogo")
    ax.set_ylabel("posição da curva no tempo (percentil)")
    ax.set_title(f"{Path(t.jogo).name} · {t.numero}º tempo · linhas tracejadas = lances anotados",
                 loc="left", fontsize=12, color=TEXTO, pad=34)
    ax.legend(loc="upper left", bbox_to_anchor=(1.07, 1), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def figuras_teste(destino, metricas, por_classe, tempos, curvas, bases):
    _estilo()
    destino = Path(destino)
    principal = [l for l in metricas if l["ponto"] in ("f1", "cp2")]
    _painel_metricas(destino / "comparacao.png", principal, "Teste · 100 jogos · todos os alertas")
    mesmo_k = [l for l in metricas if l["ponto"] == "mesmo_k" or (l["metodo"], l["ponto"]) == ("fusao", "cp2")]
    _painel_metricas(destino / "comparacao_mesmo_k.png", mesmo_k, "Teste · mesmo número de alertas que a fusão")

    linhas = [l for l in por_classe if l["ponto"] in ("f1", "cp2")]
    metodos = list(dict.fromkeys(l["metodo"] for l in linhas))
    fig, eixos = plt.subplots(1, len(CLASSES), figsize=(17, 0.5 * len(metodos) + 1.8), sharey=True)
    for ax, (classe, nome) in zip(eixos, CLASSES.items()):
        dados = {l["metodo"]: l for l in linhas if l["classe"] == classe}
        valores = [dados[m]["acertos"] / dados[m]["anotados"] if dados[m]["anotados"] else 0 for m in metodos]
        _barras(ax, [NOMES.get(m, m) for m in metodos], valores, [_cor(m) for m in metodos],
                f"{nome} ({dados[metodos[0]]['anotados']})", lambda v: f"{100 * v:.0f}%")
    fig.suptitle("Recall por tipo de lance no teste", x=0.01, ha="left", fontsize=13, color=TEXTO)
    _legenda_grupos(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(destino / "por_classe.png", dpi=160)
    plt.close(fig)

    _linha_do_tempo(destino / "linha_do_tempo.png", tempos, curvas, bases)
