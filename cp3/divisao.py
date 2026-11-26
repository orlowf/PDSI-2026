"""Sorteia os jogos de teste do Checkpoint 3 e as dobras da validação cruzada."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SAIDA_PADRAO = Path("outputs/checkpoint-3/divisao.json")


def _ler_jogos(path, coluna_lote=True):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        linhas = list(csv.DictReader(stream))
    if coluna_lote:
        return {linha["jogo"]: linha["lote"] for linha in linhas}
    return {linha["jogo"] for linha in linhas}


def _competicao(jogo):
    return jogo.split("/")[0]


def _cotas(tamanhos, total):
    """Divide `total` proporcionalmente aos tamanhos, pelo método do maior resto."""
    soma = sum(tamanhos.values())
    exatas = {chave: total * n / soma for chave, n in tamanhos.items()}
    cotas = {chave: int(valor) for chave, valor in exatas.items()}
    faltam = total - sum(cotas.values())
    for chave in sorted(exatas, key=lambda c: (-(exatas[c] - cotas[c]), c))[:faltam]:
        cotas[chave] += 1
    return cotas


def montar_divisao(manifesto, pendencias, semente=2026, n_teste=100, n_dobras=5):
    lotes = _ler_jogos(manifesto)
    fora = _ler_jogos(pendencias, coluna_lote=False)
    elegiveis = sorted(jogo for jogo in lotes if jogo not in fora)
    rng = np.random.default_rng(semente)

    avaliacao = defaultdict(list)
    for jogo in elegiveis:
        if lotes[jogo] == "avaliacao":
            avaliacao[_competicao(jogo)].append(jogo)
    cotas = _cotas({c: len(j) for c, j in avaliacao.items()}, n_teste)
    teste = []
    for competicao in sorted(avaliacao):
        jogos = avaliacao[competicao]
        escolhidos = rng.permutation(len(jogos))[:cotas[competicao]]
        teste.extend(jogos[i] for i in escolhidos)
    teste = sorted(teste)

    treino = [jogo for jogo in elegiveis if jogo not in set(teste)]
    por_competicao = defaultdict(list)
    for jogo in treino:
        por_competicao[_competicao(jogo)].append(jogo)
    dobras, proxima = {}, 0
    for competicao in sorted(por_competicao):
        jogos = por_competicao[competicao]
        for i in rng.permutation(len(jogos)):
            dobras[jogos[i]] = proxima % n_dobras
            proxima += 1
    return {"semente": semente, "teste": teste, "treino": treino, "dobras": dobras}


def ler_divisao(path=SAIDA_PADRAO):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifesto", type=Path, required=True)
    parser.add_argument("--pendencias", type=Path, required=True)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    parser.add_argument("--semente", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.saida.exists():
        sys.exit(f"{args.saida} já existe; a divisão não pode ser refeita.")
    divisao = montar_divisao(args.manifesto, args.pendencias, args.semente)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(divisao, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"teste: {len(divisao['teste'])} jogos · treino: {len(divisao['treino'])} jogos → {args.saida}")


if __name__ == "__main__":
    main()
