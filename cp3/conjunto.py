"""Leitura dos tempos de jogo extraídos para os modelos do Checkpoint 3."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from cp3.extrair import CHAVES_QUADRO, RAIZ_ATRIBUTOS, tempo_pronto
from cp3.rotulos import eventos_alvo

QUADROS_POR_PASSO = 25  # 0,5 s a 50 quadros/s
PASSO_S = 0.5


@dataclass
class Tempo:
    jogo: str
    numero: int
    logmel: np.ndarray
    quadros: dict
    eventos: list

    @property
    def n_passos(self):
        return self.logmel.shape[0] // QUADROS_POR_PASSO


@lru_cache(maxsize=1)
def anotacoes_soccernet():
    from avaliar_checkpoint2_integral import read_source_annotations
    from ouvindo_o_gol import ANNOTATIONS_DIR

    return read_source_annotations(ANNOTATIONS_DIR, {"train", "valid", "test"})


def _origem(anotacoes, jogo, numero):
    chave = f"{jogo}/{numero}_224p.mkv"
    for origem, tempos in anotacoes.items():
        if chave in tempos:
            return origem
    raise KeyError(f"Tempo sem anotação: {chave}")


def carregar_tempo(jogo, numero, raiz=RAIZ_ATRIBUTOS, anotacoes=None):
    anotacoes = anotacoes if anotacoes is not None else anotacoes_soccernet()
    destino = Path(raiz) / jogo / str(numero)
    logmel = np.load(destino / "logmel.npy", mmap_mode="r")  # 393 jogos não cabem na RAM
    with np.load(destino / "quadros.npz") as arquivo:
        quadros = {k: arquivo[k] for k in CHAVES_QUADRO}
    eventos = eventos_alvo(anotacoes, _origem(anotacoes, jogo, numero), jogo, numero)
    return Tempo(jogo, numero, logmel, quadros, eventos)


def tempos_prontos(jogos, raiz=RAIZ_ATRIBUTOS):
    return [(jogo, n) for jogo in jogos for n in (1, 2) if tempo_pronto(Path(raiz) / jogo / str(n))]


def normalizar_robusto(x):
    mediana = np.median(x, axis=0)
    mad = np.median(np.abs(x - mediana), axis=0) * 1.4826 + 1e-6
    return ((x - mediana) / mad).astype(np.float32)


def reduzido(t, normalizar=True):
    """Média e máximo de cada banda em passos de 0,5 s, normalizados no próprio tempo."""
    n = t.n_passos * QUADROS_POR_PASSO
    blocos = t.logmel[:n].astype(np.float32).reshape(t.n_passos, QUADROS_POR_PASSO, -1)
    x = np.concatenate([blocos.mean(axis=1), blocos.max(axis=1)], axis=1)
    return normalizar_robusto(x) if normalizar else x
