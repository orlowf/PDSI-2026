"""Modelo 1 do Checkpoint 3: gradient boosting (scikit-learn) sobre atributos acústicos manuais.

LightGBM foi descartado: a libomp dele e a do PyTorch no mesmo processo derrubam o macOS.
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dct
from sklearn.ensemble import HistGradientBoostingClassifier

from cp3.conjunto import QUADROS_POR_PASSO, PASSO_S
from cp3.extrair import CHAVES_QUADRO
from cp3.rotulos import alvo_gaussiano

JANELA_QUADROS = 100  # 2 s a 50 quadros/s
N_MFCC = 20
CONTEXTO = ("rms", "e_voz", "e_alta", "fluxo", "centroide")
ANTES_PASSOS = 60  # 30 s
DEPOIS_PASSOS = 20  # 10 s
FRACAO_NEGATIVOS = 0.25
PARAMETROS = dict(learning_rate=0.05, max_leaf_nodes=63, max_features=0.8, l2_regularization=1.0,
                  early_stopping=True, scoring="loss", random_state=2026)


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
    def __init__(self, cache=None, num_rodadas=1000, parada=30):
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
        return np.concatenate(xs), np.concatenate(ys)

    def treinar(self, treino, validacao, sigma_s):
        """Parada antecipada pela perda nos jogos de validação (separados por jogo, não por linha).

        Sem pesos por classe: com pesos, a discretização do scikit-learn usa percentis ponderados e
        fica lenta demais; a amostragem de negativos já reduz o desbalanceamento.
        """
        x, y = self._conjunto(treino, sigma_s, amostrar=True)
        xv, yv = self._conjunto(validacao, sigma_s, amostrar=True)
        modelo = HistGradientBoostingClassifier(max_iter=self.num_rodadas, n_iter_no_change=self.parada, **PARAMETROS)
        modelo.fit(x, y, X_val=xv, y_val=yv)
        modelo.nomes_atributos_ = self.nomes
        return modelo

    def prever(self, estado, t):
        return estado.predict_proba(self._atributos(t))[:, 1].astype(np.float32)

    def salvar(self, estado, base):
        joblib.dump(estado, base.with_suffix(".joblib"))

    def carregar(self, base):
        return joblib.load(base.with_suffix(".joblib"))


def importancia(modelo, estado, tempos, sigma_s, saida, linhas_amostra=20000, repeticoes=3):
    """Importância por permutação (queda da precisão média) em passos de tempos fora da dobra do modelo."""
    from sklearn.inspection import permutation_importance

    x, y = modelo._conjunto(tempos, sigma_s, amostrar=True)
    escolhidos = np.random.default_rng(2026).permutation(len(y))[:linhas_amostra]
    resultado = permutation_importance(estado, x[escolhidos], y[escolhidos], scoring="average_precision",
                                       n_repeats=repeticoes, random_state=2026)
    linhas = sorted(zip(estado.nomes_atributos_, resultado.importances_mean), key=lambda kv: -kv[1])
    with Path(saida).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["atributo", "queda_precisao_media"])
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
    from cp3.cv import base_modelo

    estado = modelo.carregar(base_modelo(RAIZ, "classico", escolha["sigma"], 0))
    fora_da_dobra = [t for t in tempos if divisao["dobras"][t.jogo] == 0]
    importancia(modelo, estado, fora_da_dobra, escolha["sigma"], SAIDA / "importancia_classico.csv")
    print(json.dumps(escolha), flush=True)


if __name__ == "__main__":
    main()
