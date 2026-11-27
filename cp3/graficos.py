"""Figuras do teste do Checkpoint 3 (PNG para os slides, fundo claro)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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

NOMES = {"classico": "Gradient boosting", "tcn": "TCN 1D", "cnn2d": "CNN 2D",
         "vol+bonus tom": "Regra (CP1)", "wav2vec 4 s": "Wav2vec (CP2)", "fusao": "Fusão (CP2)"}
CLASSES = {"gol": "Gol", "finalizacao no alvo": "Finalização no alvo", "finalizacao para fora": "Finalização para fora",
           "penalti": "Pênalti", "expulsao": "Expulsão"}
CAMPOS_INTEIROS = ("jogos", "acertos", "alertas", "anotados", "tempo", "ordem")
CAMPOS_REAIS = ("precisao", "revocacao", "f1", "pico_s", "escore", "evento_s")


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


def alertas_do_tempo(alertas, metodo, ponto, jogo, numero):
    return [float(l["pico_s"]) for l in alertas if (l["metodo"], l["ponto"], l["jogo"], int(l["tempo"]))
            == (metodo, ponto, jogo, numero) and l["pico_s"] != ""]


def linhas_por_classe(por_classe):
    """Recall por classe com o mesmo número de alertas que a fusão do CP2, para a comparação ser justa."""
    return [l for l in por_classe if l["ponto"] == "mesmo_k" or (l["metodo"], l["ponto"]) == ("fusao", "cp2")]


def _linha_do_tempo(destino, tempos, curvas, alertas):
    modelo = next(iter(curvas))
    i = max(range(len(tempos)), key=lambda k: (any(r == "Goal" for _, r in tempos[k].eventos), len(tempos[k].eventos)))
    t = tempos[i]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    minutos = (np.arange(t.n_passos) + 0.5) * PASSO_S / 60
    ax.plot(minutos, suavizar(curvas[modelo][i], PASSO_S, janela_s=10.0), color=COR_CP3, lw=1.2,
            label=f"Escore do {NOMES.get(modelo, modelo)} (média de 10 s)")
    for segundo, rotulo in t.eventos:
        gol = rotulo == "Goal"
        ax.axvline(segundo / 60, color=TEXTO if gol else MUDO, lw=1.4 if gol else 0.8, ls="-" if gol else (0, (3, 3)))
    ax.plot([], [], color=TEXTO, lw=1.4, label="Gol anotado")
    ax.plot([], [], color=MUDO, lw=0.8, ls=(0, (3, 3)), label="Outro lance anotado")
    topo = ax.get_ylim()[1]
    proprios = alertas_do_tempo(alertas, modelo, "f1", t.jogo, t.numero)
    ax.plot([a / 60 for a in proprios], [topo * 1.06] * len(proprios), "v", ms=8, color=COR_CP3,
            label=f"Alertas do {NOMES.get(modelo, modelo)}", clip_on=False)
    fusao = alertas_do_tempo(alertas, "fusao", "cp2", t.jogo, t.numero)
    ax.plot([a / 60 for a in fusao], [-topo * 0.06] * len(fusao), "^", ms=8, color=MUDO,
            label="Alertas da fusão (CP2)", clip_on=False)
    ax.set_ylim(0, topo)
    ax.set_xlabel("minuto do tempo de jogo")
    ax.set_ylabel("probabilidade de lance")
    ax.tick_params(axis="x", pad=16)
    ax.set_title(f"{Path(t.jogo).name} · {t.numero}º tempo", loc="left", fontsize=12, color=TEXTO, pad=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def figuras_teste(destino, metricas, por_classe, tempos, curvas, alertas):
    _estilo()
    destino = Path(destino)
    principal = [l for l in metricas if l["ponto"] in ("f1", "cp2")]
    _painel_metricas(destino / "comparacao.png", principal, "Teste · 100 jogos · todos os alertas")
    mesmo_k = [l for l in metricas if l["ponto"] == "mesmo_k" or (l["metodo"], l["ponto"]) == ("fusao", "cp2")]
    _painel_metricas(destino / "comparacao_mesmo_k.png", mesmo_k, "Teste · mesmo número de alertas que a fusão")

    linhas = linhas_por_classe(por_classe)
    metodos = list(dict.fromkeys(l["metodo"] for l in linhas))
    fig, eixos = plt.subplots(1, len(CLASSES), figsize=(17, 0.5 * len(metodos) + 1.8), sharey=True)
    for ax, (classe, nome) in zip(eixos, CLASSES.items()):
        dados = {l["metodo"]: l for l in linhas if l["classe"] == classe}
        valores = [dados[m]["acertos"] / dados[m]["anotados"] if dados[m]["anotados"] else 0 for m in metodos]
        _barras(ax, [NOMES.get(m, m) for m in metodos], valores, [_cor(m) for m in metodos],
                f"{nome} ({dados[metodos[0]]['anotados']})", lambda v: f"{100 * v:.0f}%")
    fig.suptitle("Recall por tipo de lance no teste · mesmo número de alertas que a fusão", x=0.01, ha="left", fontsize=13, color=TEXTO)
    _legenda_grupos(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(destino / "por_classe.png", dpi=160)
    plt.close(fig)

    _linha_do_tempo(destino / "linha_do_tempo.png", tempos, curvas, alertas)


def _ler(caminho):
    with Path(caminho).open(newline="", encoding="utf-8") as stream:
        linhas = list(csv.DictReader(stream))
    for linha in linhas:
        for campo in CAMPOS_INTEIROS:
            if linha.get(campo, "") != "":
                linha[campo] = int(linha[campo])
        for campo in CAMPOS_REAIS:
            if linha.get(campo, "") != "":
                linha[campo] = float(linha[campo])
    return linhas


def main(destino=None, tempos=None, curvas=None):
    """Refaz as figuras a partir dos CSVs e das curvas do teste, sem recalcular os modelos."""
    destino = Path(destino or "outputs/checkpoint-3/teste")
    if tempos is None:
        from cp3.conjunto import carregar_tempo, tempos_prontos
        from cp3.cv import RAIZ
        from cp3.divisao import ler_divisao

        tempos = [carregar_tempo(j, n) for j, n in tempos_prontos(ler_divisao()["teste"])]
        curvas = {}
        for nome in ("classico", "tcn", "cnn2d"):
            arquivo = np.load(RAIZ / f"curvas_teste_{nome}.npz")
            curvas[nome] = [arquivo[f"{t.jogo}|{t.numero}"] for t in tempos]
    figuras_teste(destino, _ler(destino / "metricas.csv"), _ler(destino / "por_classe.csv"), tempos, curvas,
                  _ler(destino / "alertas.csv"))


if __name__ == "__main__":
    main()
