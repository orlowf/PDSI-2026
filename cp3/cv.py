"""Validação cruzada por jogo e escolha do ponto de operação fora da dobra."""

import csv
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from avaliar_alternativas_checkpoint2 import measure
from cp3.conjunto import PASSO_S
from cp3.posproc import picos, suavizar, top_k

RAIZ = Path(".cache/checkpoint-3")
SAIDA = Path("outputs/checkpoint-3")
SIGMAS = (2, 4, 6)
PERCENTIS = (95, 97, 98, 99, 99.5)
DS = (10, 20, 30)
N_DOBRAS = 5
FRACAO_VALIDACAO = 0.1


class Modelo(Protocol):
    def treinar(self, treino, validacao, sigma_s): ...
    def prever(self, estado, t) -> np.ndarray: ...
    def salvar(self, estado, base: Path) -> None: ...
    def carregar(self, base: Path): ...


def caminho_oof(raiz, nome, sigma, jogo, numero):
    return Path(raiz) / "oof" / nome / f"sigma{sigma}" / jogo / f"{numero}.npy"


def base_modelo(raiz, nome, sigma, dobra):
    return Path(raiz) / "modelos" / nome / f"sigma{sigma}" / f"dobra{dobra}"


def _separar_validacao(tempos, dobra):
    jogos = sorted({t.jogo for t in tempos})
    rng = np.random.default_rng(1000 + dobra)
    n = max(1, round(FRACAO_VALIDACAO * len(jogos)))
    validacao = {jogos[i] for i in rng.permutation(len(jogos))[:n]}
    return [t for t in tempos if t.jogo not in validacao], [t for t in tempos if t.jogo in validacao]


def rodar_cv(nome, modelo, tempos, dobras, sigmas=SIGMAS, raiz=RAIZ):
    for sigma in sigmas:
        for dobra in range(N_DOBRAS):
            base = base_modelo(raiz, nome, sigma, dobra)
            if base.with_suffix(".ok").is_file():
                continue
            fora = [t for t in tempos if dobras[t.jogo] != dobra]
            dentro = [t for t in tempos if dobras[t.jogo] == dobra]
            treino, validacao = _separar_validacao(fora, dobra)
            estado = modelo.treinar(treino, validacao, sigma)
            base.parent.mkdir(parents=True, exist_ok=True)
            modelo.salvar(estado, base)
            for t in dentro:
                destino = caminho_oof(raiz, nome, sigma, t.jogo, t.numero)
                destino.parent.mkdir(parents=True, exist_ok=True)
                np.save(destino, np.asarray(modelo.prever(estado, t), dtype=np.float32))
            base.with_suffix(".ok").write_text("ok\n")
            print(f"{nome} σ={sigma} dobra {dobra} pronta", flush=True)


def melhor_linha(linhas):
    return max(linhas, key=lambda linha: (linha["f1"], -linha["alertas"]))


def _curvas(raiz, nome, sigma, tempos):
    return [suavizar(np.load(caminho_oof(raiz, nome, sigma, t.jogo, t.numero)), PASSO_S) for t in tempos]


def escolher_operacao(nome, tempos, sigmas=SIGMAS, percentis=PERCENTIS, ds=DS, raiz=RAIZ, saida=SAIDA):
    linhas = []
    for sigma in sigmas:
        curvas = _curvas(raiz, nome, sigma, tempos)
        for percentil in percentis:
            for d in ds:
                previsoes = [(t.jogo, t.numero, t.eventos, picos(p, PASSO_S, percentil, d))
                             for t, p in zip(tempos, curvas)]
                metricas, *_ = measure(previsoes)
                linhas.append(dict(sigma=sigma, percentil=percentil, d=d, **metricas))
    escolhida = melhor_linha(linhas)
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    with (saida / f"calibracao_{nome}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*linhas[0], "selecionado"])
        writer.writeheader()
        for linha in linhas:
            writer.writerow(dict(linha, selecionado=linha is escolhida))
    config = {chave: escolhida[chave] for chave in ("sigma", "percentil", "d", "f1", "alertas")}
    (saida / f"config_{nome}.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    return config


def tabela_mesmo_k(nome, tempos, k_por_tempo, sigma, d, raiz=RAIZ):
    curvas = _curvas(raiz, nome, sigma, tempos)
    previsoes = [(t.jogo, t.numero, t.eventos, top_k(p, PASSO_S, k_por_tempo[(t.jogo, t.numero)], d))
                 for t, p in zip(tempos, curvas)]
    metricas, *_ = measure(previsoes)
    return metricas
