import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from cp3.divisao import montar_divisao

TAMANHOS_AVALIACAO = {"liga_a": 60, "liga_b": 70, "liga_c": 80, "liga_d": 90, "liga_e": 50, "liga_f": 51}
PENDENTES = ["liga_a/2016/jogo 0", "liga_c/2016/jogo 5", "liga_f/2016/ajuste 1"]


@pytest.fixture
def arquivos(tmp_path):
    manifesto = tmp_path / "manifesto.csv"
    pendencias = tmp_path / "pendencias.csv"
    with manifesto.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["jogo", "divisao_origem", "lote"])
        for liga, n in TAMANHOS_AVALIACAO.items():
            for i in range(n):
                writer.writerow([f"{liga}/2016/jogo {i}", "valid", "avaliacao"])
            for i in range(15):
                writer.writerow([f"{liga}/2016/ajuste {i}", "train", "ajuste"])
    with pendencias.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["jogo", "divisao_origem", "lote", "categoria", "motivo"])
        for jogo in PENDENTES:
            writer.writerow([jogo, "valid", "avaliacao", "x", "y"])
    return manifesto, pendencias


def total_jogos():
    return sum(TAMANHOS_AVALIACAO.values()) + 15 * len(TAMANHOS_AVALIACAO)


def test_teste_vem_so_da_avaliacao(arquivos):
    divisao = montar_divisao(*arquivos)
    assert len(divisao["teste"]) == 100
    assert all("/jogo " in jogo for jogo in divisao["teste"])


def test_sem_sobreposicao_e_sem_pendentes(arquivos):
    divisao = montar_divisao(*arquivos)
    teste, treino = set(divisao["teste"]), set(divisao["treino"])
    assert not teste & treino
    assert not (teste | treino) & set(PENDENTES)
    assert len(teste) + len(treino) == total_jogos() - len(PENDENTES)


def test_estratificado(arquivos):
    divisao = montar_divisao(*arquivos)
    elegiveis = Counter(liga for liga, n in TAMANHOS_AVALIACAO.items() for _ in range(n))
    elegiveis["liga_a"] -= 1
    elegiveis["liga_c"] -= 1
    soma = sum(elegiveis.values())
    contagem = Counter(jogo.split("/")[0] for jogo in divisao["teste"])
    for liga, n in elegiveis.items():
        assert abs(contagem[liga] - 100 * n / soma) <= 1


def test_dobras_equilibradas(arquivos):
    divisao = montar_divisao(*arquivos)
    dobras = divisao["dobras"]
    assert set(dobras) == set(divisao["treino"])
    tamanhos = Counter(dobras.values())
    assert set(tamanhos) == {0, 1, 2, 3, 4}
    n = len(divisao["treino"])
    assert set(tamanhos.values()) <= {n // 5, n // 5 + 1}
    for liga in TAMANHOS_AVALIACAO:
        por_dobra = Counter(d for jogo, d in dobras.items() if jogo.startswith(liga + "/"))
        valores = [por_dobra[d] for d in range(5)]
        assert max(valores) - min(valores) <= 1


def test_deterministico(arquivos):
    assert montar_divisao(*arquivos) == montar_divisao(*arquivos)
    assert montar_divisao(*arquivos, semente=7)["teste"] != montar_divisao(*arquivos)["teste"]


def test_cli_nao_sobrescreve(arquivos, tmp_path):
    saida = tmp_path / "divisao.json"
    comando = [sys.executable, "-m", "cp3.divisao", "--manifesto", str(arquivos[0]),
               "--pendencias", str(arquivos[1]), "--saida", str(saida)]
    raiz = Path(__file__).resolve().parents[2]
    primeira = subprocess.run(comando, cwd=raiz, capture_output=True, text=True)
    assert primeira.returncode == 0, primeira.stderr
    conteudo = saida.read_text()
    assert len(json.loads(conteudo)["teste"]) == 100
    segunda = subprocess.run(comando, cwd=raiz, capture_output=True, text=True)
    assert segunda.returncode != 0
    assert saida.read_text() == conteudo
