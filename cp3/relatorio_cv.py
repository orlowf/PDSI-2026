"""Resumo fora da dobra: ponto escolhido pelo F1 e mesmo número de alertas que a fusão."""

import argparse
import csv
import json
from pathlib import Path

from avaliar_alternativas_checkpoint2 import measure
from cp3.conjunto import PASSO_S
from cp3.cv import RAIZ, SAIDA, _curvas, tabela_mesmo_k
from cp3.posproc import picos

CAMPOS = ("modelo", "ponto", "sigma", "percentil", "d", "acertos", "alertas", "anotados",
          "precisao", "revocacao", "f1")


def resumo_cv(modelos, tempos, k_por_tempo, raiz=RAIZ, saida=SAIDA):
    linhas = []
    for nome in modelos:
        config = json.loads((Path(saida) / f"config_{nome}.json").read_text(encoding="utf-8"))
        curvas = _curvas(raiz, nome, config["sigma"], tempos)
        previsoes = [(t.jogo, t.numero, t.eventos, picos(p, PASSO_S, config["percentil"], config["d"]))
                     for t, p in zip(tempos, curvas)]
        metricas, *_ = measure(previsoes)
        base = dict(modelo=nome, sigma=config["sigma"], d=config["d"])
        linhas.append(dict(base, ponto="f1", percentil=config["percentil"], **metricas))
        mesmo_k = tabela_mesmo_k(nome, tempos, k_por_tempo, config["sigma"], config["d"], raiz)
        linhas.append(dict(base, ponto="mesmo_k", percentil="", **mesmo_k))
    _gravar(Path(saida) / "cv_resumo.csv", linhas)
    return linhas


def _gravar(destino, linhas):
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CAMPOS)
        writer.writeheader()
        writer.writerows({c: linha.get(c, "") for c in CAMPOS} for linha in linhas)


def main(argv=None):
    from cp3.avaliacao import alertas_da_fusao, linhas_de_base
    from cp3.conjunto import carregar_tempo, tempos_prontos
    from cp3.divisao import ler_divisao

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelos", nargs="+", default=["classico", "tcn", "cnn2d"])
    args = parser.parse_args(argv)
    divisao = ler_divisao()
    tempos = [carregar_tempo(j, n) for j, n in tempos_prontos(divisao["treino"])]
    fusao = linhas_de_base(sorted({t.jogo for t in tempos}))["fusao"]
    k = alertas_da_fusao(fusao)
    modelos = [m for m in args.modelos if (SAIDA / f"config_{m}.json").is_file()]
    linhas = resumo_cv(modelos, tempos, k)
    chaves = {(t.jogo, t.numero) for t in tempos}
    referencia, *_ = measure([p for p in fusao if (p[0], p[1]) in chaves])
    linhas.append(dict(modelo="fusao_cp2", ponto="referencia", **referencia))
    _gravar(SAIDA / "cv_resumo.csv", linhas)
    for linha in linhas:
        print(f"{linha['modelo']:10s} {linha['ponto']:10s} alertas={linha['alertas']:6d} "
              f"acertos={linha['acertos']:6d} P={linha['precisao']:.3f} R={linha['revocacao']:.3f} "
              f"F1={linha['f1']:.3f}")


if __name__ == "__main__":
    main()
