import numpy as np
from sklearn.metrics import roc_auc_score

from cp3.classico import Classico, atributos_janela
from cp3.conjunto import Tempo
from cp3.extrair import CHAVES_QUADRO
from cp3.rotulos import alvo_gaussiano


def tempo_com_rajadas(seed, n_passos=1200, eventos_s=None):
    rng = np.random.default_rng(seed)
    n = n_passos * 25
    mel = rng.normal(-40, 3, (n, 64))
    quadros = {k: rng.normal(0, 1, n) for k in CHAVES_QUADRO}
    quadros["rms"] = np.abs(rng.normal(0.05, 0.01, n))
    if eventos_s is None:
        eventos_s = sorted(rng.uniform(30, n_passos * 0.5 - 30, 4))
    for s in eventos_s:
        a, b = int(s * 50), int((s + 6) * 50)
        mel[a:b] += 25
        quadros["rms"][a:b] += 0.5
        quadros["e_voz"][a:b] += 10
    quadros = {k: v.astype(np.float16) for k, v in quadros.items()}
    return Tempo(f"liga/2016/J{seed}", 1, mel.astype(np.float16), quadros, [(float(s), "Goal") for s in eventos_s])


def test_atributos_forma_e_nomes():
    t = tempo_com_rajadas(0)
    t.logmel[:5000] = -100
    for chave in t.quadros:
        t.quadros[chave][:5000] = 0
    x, nomes = atributos_janela(t)
    assert x.shape == (t.n_passos, len(nomes))
    assert len(set(nomes)) == len(nomes)
    assert np.isfinite(x).all()


def test_contexto_detecta_subida():
    t = tempo_com_rajadas(1, eventos_s=[])
    t.quadros["rms"][60 * 50:] += np.float16(1.0)
    x, nomes = atributos_janela(t)
    coluna = x[:, nomes.index("subida30_rms")]
    inicio = int(np.argmax(coluna > 0.5)) * 0.5  # primeiro passo em que a subida aparece
    assert abs(inicio - 60) <= 1
    assert coluna[int(90 / 0.5)] < 0.5  # 30 s depois, o novo nível já é a mediana


def test_classico_aprende_sintetico():
    treino = [tempo_com_rajadas(s) for s in range(8)]
    validacao = [tempo_com_rajadas(100)]
    modelo = Classico()
    estado = modelo.treinar(treino, validacao, 4.0)
    rotulos, escores = [], []
    for t in (tempo_com_rajadas(200), tempo_com_rajadas(201)):
        p = modelo.prever(estado, t)
        assert p.shape == (t.n_passos,)
        rotulos.append(alvo_gaussiano([e for e, _ in t.eventos], t.n_passos, 0.5, 4.0) > 0.5)
        escores.append(p)
    assert roc_auc_score(np.concatenate(rotulos), np.concatenate(escores)) > 0.9


def test_salvar_e_carregar(tmp_path):
    treino = [tempo_com_rajadas(s) for s in range(4)]
    modelo = Classico()
    estado = modelo.treinar(treino, [tempo_com_rajadas(50)], 4.0)
    modelo.salvar(estado, tmp_path / "dobra0")
    t = tempo_com_rajadas(60)
    assert np.allclose(modelo.prever(estado, t), modelo.prever(modelo.carregar(tmp_path / "dobra0"), t))


def test_classico_convive_com_torch():
    """Duas libomp no mesmo processo (LightGBM + torch) derrubam o macOS; o modelo 1 precisa conviver com o torch."""
    import subprocess
    import sys
    from pathlib import Path

    codigo = ("import torch; from tests.cp3.test_classico import tempo_com_rajadas; from cp3.classico import Classico; "
              "m = Classico(); e = m.treinar([tempo_com_rajadas(s) for s in range(3)], [tempo_com_rajadas(9)], 4.0); "
              "m.prever(e, tempo_com_rajadas(10)); torch.nn.Conv1d(4, 4, 3)(torch.randn(2, 4, 50)).sum().backward()")
    raiz = Path(__file__).resolve().parents[2]
    resultado = subprocess.run([sys.executable, "-c", codigo], cwd=raiz, capture_output=True, timeout=240)
    assert resultado.returncode == 0, resultado.stderr.decode()[-500:]


def test_importancia_grava_todos_os_atributos(tmp_path):
    from cp3.classico import importancia

    modelo = Classico()
    estado = modelo.treinar([tempo_com_rajadas(s) for s in range(4)], [tempo_com_rajadas(50)], 4.0)
    linhas = importancia(modelo, estado, [tempo_com_rajadas(70)], 4.0, tmp_path / "imp.csv", linhas_amostra=500, repeticoes=1)
    assert len(linhas) == len(estado.nomes_atributos_)
    assert linhas[0][1] >= linhas[-1][1]
    assert (tmp_path / "imp.csv").read_text().startswith("atributo,queda_precisao_media")
