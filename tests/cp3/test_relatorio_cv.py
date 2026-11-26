import csv

from cp3.cv import escolher_operacao, rodar_cv
from cp3.relatorio_cv import resumo_cv
from tests.cp3.test_cv import ModeloFalso, tempos_sinteticos


def test_resumo_tem_dois_pontos(tmp_path):
    tempos, dobras = tempos_sinteticos()
    rodar_cv("falso", ModeloFalso(), tempos, dobras, sigmas=(2,), raiz=tmp_path)
    escolher_operacao("falso", tempos, sigmas=(2,), raiz=tmp_path, saida=tmp_path / "out")
    k = {(t.jogo, t.numero): 3 for t in tempos}
    linhas = resumo_cv(["falso"], tempos, k, raiz=tmp_path, saida=tmp_path / "out")
    assert [(l["modelo"], l["ponto"]) for l in linhas] == [("falso", "f1"), ("falso", "mesmo_k")]
    assert linhas[1]["alertas"] == 3 * len(tempos)
    with (tmp_path / "out" / "cv_resumo.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
