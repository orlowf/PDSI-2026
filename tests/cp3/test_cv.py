import json

import numpy as np

from cp3.conjunto import Tempo
from cp3.cv import caminho_oof, escolher_operacao, melhor_linha, rodar_cv, tabela_mesmo_k
from cp3.rotulos import alvo_gaussiano


def tempos_sinteticos(n_jogos=10, n_passos=600):
    rng = np.random.default_rng(1)
    tempos = []
    for j in range(n_jogos):
        for numero in (1, 2):
            eventos = [(float(s), "Goal") for s in rng.uniform(20, n_passos * 0.5 - 20, 3)]
            tempos.append(Tempo(f"liga/2016/J{j}", numero, np.zeros((n_passos * 25, 64), np.float16), {}, eventos))
    dobras = {f"liga/2016/J{j}": j % 5 for j in range(n_jogos)}
    return tempos, dobras


class ModeloFalso:
    def __init__(self):
        self.vistos = []
        self.treinos = 0

    def treinar(self, treino, validacao, sigma_s):
        self.treinos += 1
        self.vistos.append({t.jogo for t in treino} | {t.jogo for t in validacao})
        return {"sigma": sigma_s, "jogos": sorted({t.jogo for t in treino})}

    def prever(self, estado, t):
        ruido = np.random.default_rng(len(t.jogo)).uniform(0, 0.05, t.n_passos)
        return alvo_gaussiano([e for e, _ in t.eventos], t.n_passos, 0.5, 2.0) + ruido

    def salvar(self, estado, base):
        base.with_suffix(".json").write_text(json.dumps(estado))

    def carregar(self, base):
        return json.loads(base.with_suffix(".json").read_text())


def test_oof_sem_vazamento(tmp_path):
    tempos, dobras = tempos_sinteticos()
    modelo = ModeloFalso()
    rodar_cv("falso", modelo, tempos, dobras, sigmas=(2,), raiz=tmp_path)
    for k in range(5):
        estado = modelo.carregar(tmp_path / "modelos" / "falso" / "sigma2" / f"dobra{k}")
        assert all(dobras[j] != k for j in estado["jogos"])
    assert all(len({dobras[j] for j in vistos}) == 4 for vistos in modelo.vistos)


def test_oof_cobre_todo_treino(tmp_path):
    tempos, dobras = tempos_sinteticos()
    rodar_cv("falso", ModeloFalso(), tempos, dobras, sigmas=(2, 4), raiz=tmp_path)
    for sigma in (2, 4):
        for t in tempos:
            assert caminho_oof(tmp_path, "falso", sigma, t.jogo, t.numero).is_file()


def test_retoma_dobras(tmp_path):
    tempos, dobras = tempos_sinteticos()
    rodar_cv("falso", ModeloFalso(), tempos, dobras, sigmas=(2,), raiz=tmp_path)
    segundo = ModeloFalso()
    rodar_cv("falso", segundo, tempos, dobras, sigmas=(2,), raiz=tmp_path)
    assert segundo.treinos == 0


def test_melhor_linha_desempata_por_menos_alertas():
    linhas = [dict(f1=0.5, alertas=30, p=1), dict(f1=0.5, alertas=20, p=2), dict(f1=0.4, alertas=5, p=3)]
    assert melhor_linha(linhas)["p"] == 2


def test_escolher_operacao_grava_escolha(tmp_path):
    tempos, dobras = tempos_sinteticos()
    rodar_cv("falso", ModeloFalso(), tempos, dobras, sigmas=(2,), raiz=tmp_path)
    escolha = escolher_operacao("falso", tempos, sigmas=(2,), raiz=tmp_path, saida=tmp_path / "out")
    assert escolha["f1"] > 0.9
    assert {"sigma", "percentil", "d", "f1", "alertas"} <= set(escolha)
    assert json.loads((tmp_path / "out" / "config_falso.json").read_text())["percentil"] == escolha["percentil"]
    assert (tmp_path / "out" / "calibracao_falso.csv").is_file()


def test_tabela_mesmo_k(tmp_path):
    tempos, dobras = tempos_sinteticos()
    rodar_cv("falso", ModeloFalso(), tempos, dobras, sigmas=(2,), raiz=tmp_path)
    k = {(t.jogo, t.numero): 2 for t in tempos}
    r = tabela_mesmo_k("falso", tempos, k, sigma=2, d=10, raiz=tmp_path)
    assert r["alertas"] == 2 * len(tempos)
