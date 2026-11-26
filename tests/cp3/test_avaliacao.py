import csv
from pathlib import Path

import pytest

from cp3.avaliacao import RAIZ_CP2, alertas_da_fusao, avaliar, linhas_de_base, por_jogo

PREVISOES = [
    ("liga/2016/A", 1, [(100.0, "Goal")], [(110.0, 0.9), (400.0, 0.8)]),
    ("liga/2016/A", 2, [(500.0, "Shots on target")], [(120.0, 0.5)]),
]


def test_avaliar_caso_conhecido():
    r = avaliar(PREVISOES)
    assert (r["acertos"], r["alertas"], r["anotados"]) == (1, 3, 2)
    assert r["precisao"] == pytest.approx(1 / 3)
    assert r["revocacao"] == pytest.approx(1 / 2)
    assert r["por_classe"]["gol"] == (1, 1)
    assert r["por_classe"]["finalizacao no alvo"] == (0, 1)
    assert len(r["detalhes"]) == 3


def test_avaliar_tempo_sem_evento():
    r = avaliar([("liga/2016/B", 1, [], [(10.0, 1.0), (50.0, 0.9), (90.0, 0.8), (130.0, 0.7)])])
    assert (r["acertos"], r["alertas"], r["anotados"]) == (0, 4, 0)
    assert r["f1"] == 0.0


def test_por_jogo():
    linhas = por_jogo(PREVISOES + [("liga/2016/B", 1, [(10.0, "Goal")], [(12.0, 1.0)])])
    por_nome = {l["jogo"]: l for l in linhas}
    assert (por_nome["liga/2016/A"]["acertos"], por_nome["liga/2016/A"]["alertas"]) == (1, 3)
    assert por_nome["liga/2016/B"]["f1"] == 1.0


def test_alertas_da_fusao():
    assert alertas_da_fusao(PREVISOES) == {("liga/2016/A", 1): 2, ("liga/2016/A", 2): 1}


@pytest.mark.lento
@pytest.mark.skipif(not RAIZ_CP2.is_dir(), reason="escores do CP2 ausentes")
def test_linhas_de_base_batem_com_cp2():
    with Path("outputs/checkpoint-2-integral/por_jogo.csv").open(newline="") as stream:
        referencia = [l for l in csv.DictReader(stream) if l["lote"] == "avaliacao"]
    jogos = sorted({l["jogo"] for l in referencia})[:5]
    bases = linhas_de_base(jogos)
    for metodo, previsoes in bases.items():
        calculado = {l["jogo"]: l for l in por_jogo(previsoes)}
        for l in referencia:
            if l["metodo"] == metodo and l["jogo"] in jogos:
                obtido = calculado[l["jogo"]]
                assert (obtido["acertos"], obtido["alertas"], obtido["anotados"]) == (
                    int(l["acertos"]), int(l["alertas"]), int(l["anotados"])), (metodo, l["jogo"])
