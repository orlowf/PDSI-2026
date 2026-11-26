import numpy as np

from cp3.graficos import figuras_teste
from cp3.teste import linhas_teste
from tests.cp3.test_teste import tempo


def test_figuras_teste_gera_pngs(tmp_path):
    tempos = [tempo("liga/2016/A", 1, [(50.0, "Goal"), (150.0, "Shots on target")]),
              tempo("liga/2016/A", 2, [(120.0, "Penalty")])]
    rng = np.random.default_rng(0)
    curvas = {m: [rng.random(t.n_passos).astype(np.float32) for t in tempos] for m in ("classico", "tcn", "cnn2d")}
    configs = {m: {"sigma": 4, "percentil": 98, "d": 20} for m in curvas}
    fusao = [("liga/2016/A", 1, tempos[0].eventos, [(52.0, 1.0)]), ("liga/2016/A", 2, tempos[1].eventos, [(30.0, 0.9)])]
    bases = {"vol+bonus tom": fusao, "wav2vec 4 s": fusao, "fusao": fusao}
    metricas, por_classe, *_ = linhas_teste(curvas, tempos, configs, bases)
    figuras_teste(tmp_path, metricas, por_classe, tempos, curvas, bases)
    nomes = sorted(p.name for p in tmp_path.glob("*.png"))
    assert nomes == ["comparacao.png", "comparacao_mesmo_k.png", "linha_do_tempo.png", "por_classe.png"]
    assert all(p.stat().st_size > 10_000 for p in tmp_path.glob("*.png"))
