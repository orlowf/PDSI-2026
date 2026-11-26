"""Modelo 1 do Checkpoint 3: LightGBM sobre atributos acústicos manuais."""

import argparse
import csv
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dct

from cp3.conjunto import QUADROS_POR_PASSO, PASSO_S
from cp3.extrair import CHAVES_QUADRO
from cp3.rotulos import alvo_gaussiano

JANELA_QUADROS = 100  # 2 s a 50 quadros/s
N_MFCC = 20
CONTEXTO = ("rms", "e_voz", "e_alta", "fluxo", "centroide")
ANTES_PASSOS = 60  # 30 s
DEPOIS_PASSOS = 20  # 10 s
FRACAO_NEGATIVOS = 0.25
PARAMETROS = dict(objective="binary", learning_rate=0.05, num_leaves=63, bagging_fraction=0.8,
                  bagging_freq=1, feature_fraction=0.8, verbose=-1, seed=2026)


def _por_quadro(t):
    mel = np.asarray(t.logmel, dtype=np.float32)
    mfcc = dct(mel, type=2, norm="ortho", axis=1)[:, :N_MFCC]
    delta = np.gradient(mfcc, axis=0)
    quadros = np.stack([np.asarray(t.quadros[k], dtype=np.float32) for k in CHAVES_QUADRO], axis=1)
    nomes = ([f"mel{i}" for i in range(mel.shape[1])] + [f"mfcc{i}" for i in range(N_MFCC)]
             + [f"dmfcc{i}" for i in range(N_MFCC)] + list(CHAVES_QUADRO))
    return np.concatenate([mel, mfcc, delta, quadros], axis=1), nomes


def _media_desvio_janela(x, n_passos):
    acumulado = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)])
    acumulado2 = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x.astype(np.float64) ** 2, axis=0)])
    centros = np.arange(n_passos) * QUADROS_POR_PASSO + QUADROS_POR_PASSO // 2
    a = np.clip(centros - JANELA_QUADROS // 2, 0, len(x))
    b = np.clip(centros + JANELA_QUADROS // 2, 0, len(x))
    n = (b - a)[:, None]
    media = (acumulado[b] - acumulado[a]) / n
    desvio = np.sqrt(np.maximum((acumulado2[b] - acumulado2[a]) / n - media**2, 0))
    return media, desvio


def _mediana_vizinha(s, antes, depois, funcao):
    """Para cada passo, aplica `funcao` aos `antes` passos anteriores ou aos `depois` seguintes."""
    if antes:
        preenchido = np.concatenate([np.full(antes, np.nan), s])
        janelas = sliding_window_view(preenchido, antes)[: len(s)]
    else:
        preenchido = np.concatenate([s[1:], np.full(depois, np.nan)])
        janelas = sliding_window_view(preenchido, depois)[: len(s)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(funcao(janelas, axis=1), nan=0.0)


def atributos_janela(t):
    n_passos = t.n_passos
    x, nomes_quadro = _por_quadro(t)
    x = x[: n_passos * QUADROS_POR_PASSO]
    media, desvio = _media_desvio_janela(x, n_passos)
    colunas = [media, desvio]
    nomes = [f"media_{n}" for n in nomes_quadro] + [f"desvio_{n}" for n in nomes_quadro]
    blocos = x.reshape(n_passos, QUADROS_POR_PASSO, -1).mean(axis=1)
    for chave in CONTEXTO:
        j = nomes_quadro.index(chave)
        serie = blocos[:, j]
        colunas.append((media[:, j] - _mediana_vizinha(serie, ANTES_PASSOS, 0, np.nanmedian))[:, None])
        colunas.append((media[:, j] - _mediana_vizinha(serie, 0, DEPOIS_PASSOS, np.nanmedian))[:, None])
        nomes += [f"subida30_{chave}", f"diferenca10_{chave}"]
    rms = blocos[:, nomes_quadro.index("rms")]
    colunas.append(_mediana_vizinha(rms, 0, DEPOIS_PASSOS, np.nanmax)[:, None])
    nomes.append("max10_rms")
    return np.nan_to_num(np.concatenate(colunas, axis=1)).astype(np.float32), nomes


class Classico:
    def __init__(self, cache=None, num_rodadas=2000, parada=100):
        self.cache = Path(cache) if cache else None
        self.num_rodadas = num_rodadas
        self.parada = parada
        self.nomes = None

    def _atributos(self, t):
        if self.cache is None:
            x, self.nomes = atributos_janela(t)
            return x
        destino = self.cache / t.jogo / f"{t.numero}.npy"
        if destino.is_file():
            if self.nomes is None:
                self.nomes = atributos_janela(t)[1]
            return np.load(destino)
        x, self.nomes = atributos_janela(t)
        destino.parent.mkdir(parents=True, exist_ok=True)
        np.save(destino, x)
        return x

    def _conjunto(self, tempos, sigma_s, amostrar):
        rng = np.random.default_rng(2026)
        xs, ys = [], []
        for t in tempos:
            x = self._atributos(t)
            y = alvo_gaussiano([e for e, _ in t.eventos], len(x), PASSO_S, sigma_s) > 0.5
            if amostrar:
                manter = y | (rng.random(len(y)) < FRACAO_NEGATIVOS)
                x, y = x[manter], y[manter]
            xs.append(x)
            ys.append(y)
        return np.concatenate(xs), np.concatenate(ys).astype(np.float32)

    def treinar(self, treino, validacao, sigma_s):
        x, y = self._conjunto(treino, sigma_s, amostrar=True)
        xv, yv = self._conjunto(validacao, sigma_s, amostrar=True)
        positivos = max(float(y.sum()), 1.0)
        parametros = dict(PARAMETROS, scale_pos_weight=(len(y) - positivos) / positivos)
        dados = lgb.Dataset(x, y, feature_name=self.nomes, free_raw_data=True)
        dados_val = lgb.Dataset(xv, yv, reference=dados)
        return lgb.train(parametros, dados, self.num_rodadas, valid_sets=[dados_val],
                         callbacks=[lgb.early_stopping(self.parada, verbose=False)])

    def prever(self, estado, t):
        return estado.predict(self._atributos(t), num_iteration=estado.best_iteration).astype(np.float32)

    def salvar(self, estado, base):
        estado.save_model(str(base.with_suffix(".txt")), num_iteration=estado.best_iteration)

    def carregar(self, base):
        return lgb.Booster(model_file=str(base.with_suffix(".txt")))


def importancia(raiz, sigma, saida):
    """Ganho médio de cada atributo nos 5 modelos das dobras."""
    from cp3.cv import N_DOBRAS, base_modelo

    total = None
    for dobra in range(N_DOBRAS):
        booster = lgb.Booster(model_file=str(base_modelo(raiz, "classico", sigma, dobra).with_suffix(".txt")))
        ganho = dict(zip(booster.feature_name(), booster.feature_importance("gain")))
        total = ganho if total is None else {k: total[k] + v for k, v in ganho.items()}
    linhas = sorted(((k, v / N_DOBRAS) for k, v in total.items()), key=lambda kv: -kv[1])
    with Path(saida).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["atributo", "ganho_medio"])
        writer.writerows(linhas)
    return linhas


def main(argv=None):
    from cp3.conjunto import carregar_tempo, tempos_prontos
    from cp3.cv import RAIZ, SAIDA, escolher_operacao, rodar_cv
    from cp3.divisao import ler_divisao

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    divisao = ler_divisao()
    tempos = [carregar_tempo(j, n) for j, n in tempos_prontos(divisao["treino"])]
    print(f"{len(tempos)} tempos de treino prontos", flush=True)
    modelo = Classico(cache=RAIZ / "classico")
    rodar_cv("classico", modelo, tempos, divisao["dobras"])
    escolha = escolher_operacao("classico", tempos)
    importancia(RAIZ, escolha["sigma"], SAIDA / "importancia_classico.csv")
    print(json.dumps(escolha), flush=True)


if __name__ == "__main__":
    main()
