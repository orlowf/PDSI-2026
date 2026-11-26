import numpy as np

from cp3.conjunto import Tempo, carregar_tempo, reduzido, tempos_prontos
from cp3.extrair import CHAVES_QUADRO


def tempo_sintetico(n_quadros, seed=0):
    rng = np.random.default_rng(seed)
    mel = rng.normal(-40, 10, (n_quadros, 64)).astype(np.float16)
    quadros = {k: rng.normal(0, 1, n_quadros).astype(np.float16) for k in CHAVES_QUADRO}
    return Tempo("liga/2016/A", 1, mel, quadros, [])


def test_reduzido_forma():
    t = tempo_sintetico(1000)
    t.logmel[:25] = 3.0
    r = reduzido(t, normalizar=False)
    assert r.shape == (40, 128)
    assert r.dtype == np.float32
    assert t.n_passos == 40
    assert np.allclose(r[0, :64], r[0, 64:])
    assert not np.allclose(r[1, :64], r[1, 64:])


def test_reduzido_normaliza():
    r = reduzido(tempo_sintetico(5000))
    assert np.abs(np.median(r, axis=0)).max() < 1e-3


def gravar(raiz, jogo, numero, n=100, completo=True):
    destino = raiz / jogo / str(numero)
    destino.mkdir(parents=True)
    t = tempo_sintetico(n)
    np.save(destino / "logmel.npy", t.logmel)
    if completo:
        np.savez(destino / "quadros.npz", **t.quadros)


def test_tempos_prontos_ignora_incompletos(tmp_path):
    gravar(tmp_path, "liga/2016/A", 1)
    gravar(tmp_path, "liga/2016/A", 2, completo=False)
    gravar(tmp_path, "liga/2016/B", 1)
    gravar(tmp_path, "liga/2016/B", 2)
    assert tempos_prontos(["liga/2016/A", "liga/2016/B", "liga/2016/C"], tmp_path) == [
        ("liga/2016/A", 1), ("liga/2016/B", 1), ("liga/2016/B", 2)]


def test_carregar_tempo_com_eventos(tmp_path):
    gravar(tmp_path, "liga/2016/A", 1, n=500)
    anotacoes = {"valid": {"liga/2016/A/1_224p.mkv": [{"position_ms": 5000, "label": "Goal"},
                                                      {"position_ms": 6000, "label": "Foul"}]}}
    t = carregar_tempo("liga/2016/A", 1, tmp_path, anotacoes)
    assert t.logmel.shape == (500, 64)
    assert set(t.quadros) == set(CHAVES_QUADRO)
    assert t.eventos == [(5.0, "Goal")]


def test_carregar_tempo_usa_memmap(tmp_path):
    gravar(tmp_path, "liga/2016/A", 1, n=500)
    anotacoes = {"valid": {"liga/2016/A/1_224p.mkv": []}}
    t = carregar_tempo("liga/2016/A", 1, tmp_path, anotacoes)
    assert isinstance(t.logmel, np.memmap)
