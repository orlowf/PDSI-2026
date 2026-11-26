import numpy as np

from cp3.rotulos import alvo_gaussiano, eventos_alvo


def test_alvo_pico_no_evento():
    alvo = alvo_gaussiano([100.0], 400, 0.5, 4.0)
    assert alvo.dtype == np.float32
    assert int(np.argmax(alvo)) == 199
    assert abs(alvo[199] - np.exp(-0.5 * (0.25 / 4) ** 2)) < 1e-6
    assert abs(alvo[207] - np.exp(-0.5 * (3.75 / 4) ** 2)) < 1e-6  # centro do passo 207 = 103,75 s


def test_alvo_dois_eventos_maximo():
    alvo = alvo_gaussiano([100.0, 101.0], 400, 0.5, 4.0)
    assert alvo.max() <= 1.0


def test_alvo_sem_evento():
    assert not alvo_gaussiano([], 100, 0.5, 4.0).any()


def test_alvo_evento_apos_fim():
    alvo = alvo_gaussiano([1000.0], 100, 0.5, 2.0)
    assert not alvo.any()


def anotacoes(eventos):
    chave = "liga/2016/A 1 - 0 B/1_224p.mkv"
    return {"train": {chave: [{"position_ms": int(t * 1000), "label": rotulo} for t, rotulo in eventos]}}


def test_eventos_alvo_filtra_e_deduplica():
    anot = anotacoes([(10.0, "Goal"), (11.0, "Goal"), (20.0, "Throw-in"), (30.0, "Shots on target")])
    assert eventos_alvo(anot, "train", "liga/2016/A 1 - 0 B", 1) == [(10.0, "Goal"), (30.0, "Shots on target")]


def test_eventos_alvo_mantem_evento_apos_fim():
    anot = anotacoes([(99999.0, "Goal")])
    assert eventos_alvo(anot, "train", "liga/2016/A 1 - 0 B", 1) == [(99999.0, "Goal")]
