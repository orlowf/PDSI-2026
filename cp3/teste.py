"""Avaliação única dos modelos do Checkpoint 3 nos 100 jogos de teste."""

import argparse
import csv
import json
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

from cp3.avaliacao import alertas_da_fusao, avaliar, por_jogo
from cp3.conjunto import PASSO_S
from cp3.cv import N_DOBRAS, RAIZ, SAIDA, base_modelo
from cp3.posproc import picos, suavizar, top_k

CAMPOS_METRICAS = ("metodo", "ponto", "jogos", "acertos", "alertas", "anotados", "precisao", "revocacao", "f1")


def hash_configuracao(saida, modelos):
    resumo = sha256()
    for nome in ["divisao.json", *sorted(f"config_{m}.json" for m in modelos)]:
        resumo.update(nome.encode())
        resumo.update((Path(saida) / nome).read_bytes())
    return resumo.hexdigest()


def verificar_trava(destino, hash_atual):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    trava = destino / "trava.json"
    if trava.is_file():
        anterior = json.loads(trava.read_text(encoding="utf-8"))["hash"]
        if anterior != hash_atual:
            sys.exit("A divisão ou a configuração mudou depois da avaliação no teste; o teste não é reaberto.")
        return
    trava.write_text(json.dumps({"hash": hash_atual, "data": datetime.now().isoformat(timespec="seconds")},
                                indent=1) + "\n", encoding="utf-8")


def curva_media(modelo, estados, t):
    return np.mean([modelo.prever(estado, t) for estado in estados], axis=0).astype(np.float32)


def linhas_teste(curvas, tempos, configs, bases):
    """curvas: modelo -> uma curva por tempo; bases: método do CP2 -> previsões já prontas."""
    conjuntos = []
    k = alertas_da_fusao(bases["fusao"]) if "fusao" in bases else None
    for nome, por_tempo in curvas.items():
        config = configs[nome]
        suaves = [suavizar(p, PASSO_S) for p in por_tempo]
        conjuntos.append((nome, "f1", [(t.jogo, t.numero, t.eventos, picos(p, PASSO_S, config["percentil"], config["d"]))
                                       for t, p in zip(tempos, suaves)]))
        if k is not None:
            conjuntos.append((nome, "mesmo_k", [(t.jogo, t.numero, t.eventos, top_k(p, PASSO_S, k[(t.jogo, t.numero)], config["d"]))
                                                for t, p in zip(tempos, suaves)]))
    conjuntos += [(metodo, "cp2", previsoes) for metodo, previsoes in bases.items()]

    metricas, por_classe, jogos, alertas = [], [], [], []
    for metodo, ponto, previsoes in conjuntos:
        r = avaliar(previsoes)
        grupo = dict(metodo=metodo, ponto=ponto)
        metricas.append(dict(grupo, jogos=len({p[0] for p in previsoes}),
                             **{c: r[c] for c in CAMPOS_METRICAS[3:]}))
        for classe, (acertos, anotados) in r["por_classe"].items():
            por_classe.append(dict(grupo, classe=classe, acertos=acertos, anotados=anotados,
                                   revocacao=acertos / anotados if anotados else ""))
        jogos += [dict(grupo, **linha) for linha in por_jogo(previsoes)]
        alertas += [dict(grupo, **linha) for linha in r["detalhes"]]
    return metricas, por_classe, jogos, alertas


def _gravar(destino, linhas):
    with Path(destino).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)


def _modelos():
    from cp3.classico import Classico
    from cp3.treino_rede import RedeCNN, RedeTCN

    return {"classico": Classico(cache=RAIZ / "classico"), "tcn": RedeTCN(), "cnn2d": RedeCNN()}


def main(argv=None):
    from cp3.avaliacao import linhas_de_base
    from cp3.conjunto import carregar_tempo, tempos_prontos
    from cp3.divisao import ler_divisao

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelos", nargs="+", default=["classico", "tcn", "cnn2d"])
    args = parser.parse_args(argv)
    ausentes = [m for m in args.modelos if not (SAIDA / f"config_{m}.json").is_file()]
    if ausentes:
        sys.exit(f"Sem configuração congelada para: {', '.join(ausentes)}")
    divisao = ler_divisao()
    prontos = tempos_prontos(divisao["teste"])
    if len(prontos) != 2 * len(divisao["teste"]):
        sys.exit(f"Só {len(prontos)} de {2 * len(divisao['teste'])} tempos de teste foram extraídos.")
    destino = SAIDA / "teste"
    verificar_trava(destino, hash_configuracao(SAIDA, args.modelos))

    tempos = [carregar_tempo(j, n) for j, n in prontos]
    fabricas = _modelos()
    configs, curvas = {}, {}
    for nome in args.modelos:
        configs[nome] = json.loads((SAIDA / f"config_{nome}.json").read_text(encoding="utf-8"))
        modelo = fabricas[nome]
        estados = [modelo.carregar(base_modelo(RAIZ, nome, configs[nome]["sigma"], d)) for d in range(N_DOBRAS)]
        curvas[nome] = [curva_media(modelo, estados, t) for t in tempos]
        np.savez_compressed(RAIZ / f"curvas_teste_{nome}.npz",
                            **{f"{t.jogo}|{t.numero}": c for t, c in zip(tempos, curvas[nome])})
    bases = linhas_de_base(divisao["teste"])
    metricas, por_classe, jogos, alertas = linhas_teste(curvas, tempos, configs, bases)
    for nome, linhas in (("metricas", metricas), ("por_classe", por_classe), ("por_jogo", jogos), ("alertas", alertas)):
        _gravar(destino / f"{nome}.csv", linhas)
    for linha in metricas:
        print(f"{linha['metodo']:14s} {linha['ponto']:8s} alertas={linha['alertas']:6d} acertos={linha['acertos']:5d} "
              f"P={linha['precisao']:.3f} R={linha['revocacao']:.3f} F1={linha['f1']:.3f}")
    from cp3.graficos import figuras_teste

    figuras_teste(destino, metricas, por_classe, tempos, curvas, alertas)


if __name__ == "__main__":
    main()
