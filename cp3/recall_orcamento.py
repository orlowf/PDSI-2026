"""Recall fora da dobra em função do número de alertas por tempo (só jogos de treino; o teste não é lido)."""

import argparse
import csv
import json

import numpy as np

from cp3.avaliacao import avaliar
from cp3.conjunto import PASSO_S
from cp3.posproc import top_k

ORCAMENTOS = (5, 10, 15, 20, 27, 35, 45, 60, 90)
TOLERANCIA_S = 15.0


def cobertura(previsoes, tolerancia_s=TOLERANCIA_S):
    """Lances com pelo menos um alerta a até `tolerancia_s`; um alerta pode cobrir vários lances (corte de 30 s)."""
    achados = total = 0
    for _jogo, _tempo, eventos, alertas in previsoes:
        horarios = np.array([s for s, _ in alertas])
        for segundo, _rotulo in eventos:
            total += 1
            achados += bool(len(horarios)) and bool(np.min(np.abs(horarios - segundo)) <= tolerancia_s)
    return achados, total


def fracao_lances_proximos(tempos, limite_s):
    distancias = [np.diff(sorted(s for s, _ in t.eventos)) for t in tempos if len(t.eventos) > 1]
    distancias = np.concatenate(distancias) if distancias else np.array([])
    return float(np.mean(distancias < limite_s)) if len(distancias) else 0.0


def linhas_orcamento(nome, curvas, tempos, orcamentos=ORCAMENTOS, d_s=10):
    linhas = []
    for k in orcamentos:
        previsoes = [(t.jogo, t.numero, t.eventos, top_k(p, PASSO_S, k, d_s)) for t, p in zip(tempos, curvas)]
        r = avaliar(previsoes)
        achados, total = cobertura(previsoes)
        linha = dict(metodo=nome, alertas_por_tempo=k, d=d_s, recall_um_a_um=r["revocacao"],
                     recall_cobertura=achados / total if total else 0.0)
        linha.update({f"recall_{c}": (a / n if n else 0.0) for c, (a, n) in r["por_classe"].items()})
        linhas.append(linha)
    return linhas


def main(argv=None):
    from cp3.conjunto import carregar_tempo, tempos_prontos
    from cp3.cv import RAIZ, SAIDA, _curvas
    from cp3.divisao import ler_divisao

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelos", nargs="+", default=["classico", "tcn", "cnn2d"])
    args = parser.parse_args(argv)
    tempos = [carregar_tempo(j, n) for j, n in tempos_prontos(ler_divisao()["treino"])]
    linhas = []
    for nome in args.modelos:
        sigma = json.loads((SAIDA / f"config_{nome}.json").read_text(encoding="utf-8"))["sigma"]
        linhas += linhas_orcamento(nome, _curvas(RAIZ, nome, sigma, tempos), tempos)
    rng = np.random.default_rng(2026)
    linhas += linhas_orcamento("aleatorio", [rng.random(t.n_passos) for t in tempos], tempos)
    destino = SAIDA / "recall_orcamento.csv"
    with destino.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)
    for limite in (10, 20, 30):
        print(f"lances a menos de {limite} s do anterior: {100 * fracao_lances_proximos(tempos, limite):.1f}%")
    for l in linhas:
        print(f"{l['metodo']:10s} k={l['alertas_por_tempo']:3d} um-a-um={100 * l['recall_um_a_um']:5.1f}% "
              f"cobertura={100 * l['recall_cobertura']:5.1f}%")


if __name__ == "__main__":
    main()
