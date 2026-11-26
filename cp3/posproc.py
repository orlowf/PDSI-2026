"""Transforma a curva de probabilidade de um tempo de jogo em alertas."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def suavizar(p, passo_s, janela_s=2.0):
    return uniform_filter1d(np.asarray(p, dtype=np.float32), max(1, round(janela_s / passo_s)))


def _suprimir(p, passo_s, indices, d_s, limite=None):
    """Supressão gulosa: mantém o pico mais forte e descarta os que estão a menos de d_s dele."""
    escolhidos = []
    for i in sorted(indices, key=lambda i: -p[i]):
        if all(abs(int(i) - j) * passo_s >= d_s for j in escolhidos):
            escolhidos.append(int(i))
            if limite is not None and len(escolhidos) == limite:
                break
    return [((i + 0.5) * passo_s, float(p[i])) for i in escolhidos]


def _maximos(p):
    p = np.asarray(p)
    indices, _ = find_peaks(np.concatenate([[-np.inf], p, [-np.inf]]))
    return indices - 1


def picos(p, passo_s, percentil, d_s):
    limiar = float(np.percentile(p, percentil))
    return _suprimir(p, passo_s, [i for i in _maximos(p) if p[i] > limiar], d_s)


def top_k(p, passo_s, k, d_s):
    return _suprimir(p, passo_s, _maximos(p), d_s, limite=k)
