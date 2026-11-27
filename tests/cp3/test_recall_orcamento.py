import numpy as np

from cp3.recall_orcamento import cobertura, fracao_lances_proximos, linhas_orcamento
from tests.cp3.test_teste import tempo


def test_cobertura_um_alerta_cobre_dois_lances_proximos():
    eventos = [(100.0, "Penalty"), (105.0, "Goal"), (300.0, "Shots on target")]
    previsoes = [("j", 1, eventos, [(102.0, 0.9), (200.0, 0.5)])]
    assert cobertura(previsoes) == (2, 3)


def test_cobertura_respeita_tolerancia_de_15_s():
    previsoes = [("j", 1, [(100.0, "Goal")], [(115.0, 1.0)]), ("j", 2, [(100.0, "Goal")], [(115.5, 1.0)])]
    assert cobertura(previsoes) == (1, 2)


def test_fracao_lances_proximos():
    tempos = [tempo("j", 1, [(10.0, "Goal"), (15.0, "Penalty"), (100.0, "Goal")])]
    assert fracao_lances_proximos(tempos, 10.0) == 0.5


def test_linhas_orcamento_recall_cresce_com_o_numero_de_alertas():
    tempos = [tempo("liga/2016/A", 1, [(50.0, "Goal"), (150.0, "Shots on target")]),
              tempo("liga/2016/A", 2, [(120.0, "Shots off target")])]
    curvas = [np.random.default_rng(i).random(t.n_passos).astype(np.float32) for i, t in enumerate(tempos)]
    linhas = linhas_orcamento("classico", curvas, tempos, orcamentos=(1, 5, 20), d_s=10)
    assert [l["alertas_por_tempo"] for l in linhas] == [1, 5, 20]
    assert {"recall_um_a_um", "recall_cobertura", "recall_gol", "recall_finalizacao para fora"} <= set(linhas[0])
    assert linhas[0]["recall_um_a_um"] <= linhas[-1]["recall_um_a_um"]
    assert all(l["recall_um_a_um"] <= l["recall_cobertura"] for l in linhas)
