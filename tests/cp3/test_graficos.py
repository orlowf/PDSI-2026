import csv

import numpy as np

from cp3.graficos import NOMES, alertas_do_tempo, figuras_teste, linhas_por_classe, main
from cp3.teste import _gravar, linhas_teste
from tests.cp3.test_teste import tempo


def cenario():
    tempos = [tempo("liga/2016/A", 1, [(50.0, "Goal"), (150.0, "Shots on target")]),
              tempo("liga/2016/A", 2, [(120.0, "Penalty")])]
    rng = np.random.default_rng(0)
    curvas = {m: [rng.random(t.n_passos).astype(np.float32) for t in tempos] for m in ("classico", "tcn", "cnn2d")}
    configs = {m: {"sigma": 4, "percentil": 98, "d": 20} for m in curvas}
    fusao = [("liga/2016/A", 1, tempos[0].eventos, [(52.0, 1.0)]), ("liga/2016/A", 2, tempos[1].eventos, [(30.0, 0.9)])]
    bases = {"vol+bonus tom": fusao, "wav2vec 4 s": fusao, "fusao": fusao}
    return tempos, curvas, linhas_teste(curvas, tempos, configs, bases)


def test_figuras_teste_gera_pngs(tmp_path):
    tempos, curvas, (metricas, por_classe, _, alertas) = cenario()
    figuras_teste(tmp_path, metricas, por_classe, tempos, curvas, alertas)
    nomes = sorted(p.name for p in tmp_path.glob("*.png"))
    assert nomes == ["comparacao.png", "comparacao_mesmo_k.png", "linha_do_tempo.png", "por_classe.png"]
    assert all(p.stat().st_size > 10_000 for p in tmp_path.glob("*.png"))


def test_por_classe_compara_com_o_mesmo_numero_de_alertas():
    _, _, (_, por_classe, _, _) = cenario()
    pares = {(l["metodo"], l["ponto"]) for l in linhas_por_classe(por_classe)}
    assert pares == {("classico", "mesmo_k"), ("tcn", "mesmo_k"), ("cnn2d", "mesmo_k"), ("fusao", "cp2")}


def test_alertas_do_tempo_filtra_metodo_ponto_e_tempo():
    _, _, (_, _, _, alertas) = cenario()
    fusao = alertas_do_tempo(alertas, "fusao", "cp2", "liga/2016/A", 2)
    assert fusao == [30.0]
    assert all(isinstance(s, float) for s in alertas_do_tempo(alertas, "classico", "f1", "liga/2016/A", 1))


def test_nome_do_modelo_classico_nao_e_lightgbm():
    assert NOMES["classico"] == "Gradient boosting"


def test_main_refaz_figuras_a_partir_dos_csvs(tmp_path):
    tempos, curvas, (metricas, por_classe, _, alertas) = cenario()
    for nome, linhas in (("metricas", metricas), ("por_classe", por_classe), ("alertas", alertas)):
        _gravar(tmp_path / f"{nome}.csv", linhas)
    with (tmp_path / "alertas.csv").open(newline="", encoding="utf-8") as stream:
        assert all(isinstance(l["pico_s"], str) for l in csv.DictReader(stream))
    main(destino=tmp_path, tempos=tempos, curvas=curvas)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "comparacao.png", "comparacao_mesmo_k.png", "linha_do_tempo.png", "por_classe.png"]
