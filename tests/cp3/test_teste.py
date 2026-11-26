import json

import numpy as np
import pytest

from cp3.conjunto import Tempo
from cp3.teste import curva_media, hash_configuracao, linhas_teste, verificar_trava


def preparar_configs(tmp_path, percentil=98):
    (tmp_path / "divisao.json").write_text(json.dumps({"teste": ["a"]}))
    (tmp_path / "config_classico.json").write_text(json.dumps({"sigma": 4, "percentil": percentil, "d": 20}))
    return hash_configuracao(tmp_path, ["classico"])


def test_trava_recusa_config_alterada(tmp_path):
    destino = tmp_path / "teste"
    verificar_trava(destino, preparar_configs(tmp_path))
    with pytest.raises(SystemExit):
        verificar_trava(destino, preparar_configs(tmp_path, percentil=99))
    assert sorted(p.name for p in destino.iterdir()) == ["trava.json"]


def test_trava_aceita_mesma_config(tmp_path):
    destino = tmp_path / "teste"
    verificar_trava(destino, preparar_configs(tmp_path))
    verificar_trava(destino, preparar_configs(tmp_path))
    assert json.loads((destino / "trava.json").read_text())["hash"] == preparar_configs(tmp_path)


class Constante:
    def prever(self, estado, t):
        return np.full(t.n_passos, estado, dtype=np.float32)


def tempo(jogo, numero, eventos, n_passos=400):
    return Tempo(jogo, numero, np.zeros((n_passos * 25, 64), np.float16), {}, eventos)


def test_media_das_dobras():
    curva = curva_media(Constante(), [0.1, 0.2, 0.3, 0.4, 0.5], tempo("a", 1, []))
    assert np.allclose(curva, 0.3)


def test_mesmo_k_iguala_fusao():
    tempos = [tempo("liga/2016/A", 1, [(50.0, "Goal")]), tempo("liga/2016/A", 2, [(120.0, "Goal")])]
    rng = np.random.default_rng(0)
    curvas = {"classico": [rng.random(t.n_passos).astype(np.float32) for t in tempos]}
    configs = {"classico": {"sigma": 4, "percentil": 98, "d": 20}}
    fusao = [("liga/2016/A", 1, tempos[0].eventos, [(52.0, 1.0), (100.0, 0.5), (150.0, 0.2)]),
             ("liga/2016/A", 2, tempos[1].eventos, [(30.0, 0.9)])]
    metricas, por_classe, por_jogo, alertas = linhas_teste(curvas, tempos, configs, {"fusao": fusao})
    linha = {(l["metodo"], l["ponto"]): l for l in metricas}
    assert linha[("classico", "mesmo_k")]["alertas"] == linha[("fusao", "cp2")]["alertas"] == 4
    assert ("classico", "f1") in linha
    assert {l["metodo"] for l in por_classe} == {"classico", "fusao"}
    assert all(l["jogo"] == "liga/2016/A" for l in por_jogo)
    assert alertas
