"""Lances-alvo por tempo de jogo e alvo gaussiano para o treino."""

import numpy as np

from ouvindo_o_gol import KEY_EVENTS, dedupe_events


def eventos_alvo(anotacoes, divisao_origem, jogo, tempo):
    """Lances-alvo deduplicados como no CP2; anotações além do fim do áudio continuam na lista."""
    brutos = anotacoes[divisao_origem][f"{jogo}/{tempo}_224p.mkv"]
    eventos = dedupe_events([(e["position_ms"] / 1000, e["label"]) for e in brutos])
    return [(t, rotulo) for t, rotulo in eventos if rotulo in KEY_EVENTS]


def alvo_gaussiano(eventos_s, n_passos, passo_s, sigma_s):
    """Máximo das gaussianas dos eventos, avaliado no centro de cada passo."""
    centros = (np.arange(n_passos) + 0.5) * passo_s
    alvo = np.zeros(n_passos, dtype=np.float32)
    for t in eventos_s:
        if t <= n_passos * passo_s:
            np.maximum(alvo, np.exp(-0.5 * ((centros - t) / sigma_s) ** 2).astype(np.float32), out=alvo)
    return alvo
