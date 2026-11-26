import numpy as np

from cp3.posproc import picos, suavizar, top_k

PASSO = 0.5


def serie_com_picos(alturas_por_segundo, n=1000):
    p = np.zeros(n, dtype=np.float32)
    for segundo, altura in alturas_por_segundo.items():
        p[int(segundo / PASSO)] = altura
    return p


def test_picos_supressao():
    p = serie_com_picos({100: 0.9, 105: 0.8})
    assert [round(t) for t, _ in picos(p, PASSO, 99, 10)] == [100]
    resultado = picos(p, PASSO, 99, 3)
    assert [round(t) for t, _ in resultado] == [100, 105]
    assert resultado[0][1] > resultado[1][1]


def test_picos_serie_constante():
    assert picos(np.full(500, 0.3, dtype=np.float32), PASSO, 95, 10) == []


def test_top_k_exato():
    p = serie_com_picos({50: 0.5, 100: 0.9, 150: 0.7, 200: 0.6, 250: 0.8})
    assert [round(t) for t, _ in top_k(p, PASSO, 3, 10)] == [100, 250, 150]
    assert len(top_k(p, PASSO, 50, 10)) == 5


def test_horario_do_passo():
    p = np.zeros(100, dtype=np.float32)
    p[40] = 1.0
    assert picos(p, PASSO, 50, 10)[0][0] == (40 + 0.5) * PASSO


def test_suavizar_preserva_tamanho_e_media():
    p = serie_com_picos({100: 1.0})
    s = suavizar(p, PASSO, 2.0)
    assert s.shape == p.shape
    assert abs(s.sum() - p.sum()) < 1e-5
    assert s.max() < 1.0
