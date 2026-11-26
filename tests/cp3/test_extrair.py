import numpy as np
import librosa
import soundfile as sf

from cp3.extrair import N_MELS, SR, atributos_quadro, extrair_tempo, logmel, tempo_pronto

CHAVES = {"rms", "centroide", "rolloff", "fluxo", "zcr", "e_baixa", "e_voz", "e_alta"}


def seno(freq=1000.0, segundos=10.0):
    t = np.arange(int(SR * segundos)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_logmel_forma_e_tipo():
    m = logmel(seno(), SR)
    assert m.shape == (501, N_MELS)
    assert m.dtype == np.float16
    assert np.isfinite(m).all()


def test_logmel_banda_do_seno():
    m = logmel(seno(), SR).astype(np.float32)
    bordas = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0.0, fmax=SR / 2)
    banda = int(np.argmax(m.mean(axis=0)))
    assert bordas[banda] <= 1000 <= bordas[banda + 2]


def test_atributos_seno():
    a = atributos_quadro(seno(), SR)
    assert set(a) == CHAVES
    assert all(v.shape == (501,) and v.dtype == np.float16 for v in a.values())
    assert abs(np.median(a["centroide"].astype(np.float32)) - 1000) < 100
    assert np.median(a["e_voz"]) > np.median(a["e_baixa"])
    assert np.median(a["e_voz"]) > np.median(a["e_alta"])
    assert abs(np.median(a["zcr"].astype(np.float32)) - 2000 / SR) < 0.02


def test_silencio_finito():
    y = np.zeros(SR * 5, dtype=np.float32)
    assert np.isfinite(logmel(y, SR)).all()
    for valor in atributos_quadro(y, SR).values():
        assert np.isfinite(valor).all()


def test_extrair_tempo_atomico(tmp_path):
    wav = tmp_path / "1.wav"
    sf.write(wav, seno(segundos=3.0), SR)
    destino = tmp_path / "saida"
    n = extrair_tempo(wav, destino)
    assert n == 151
    assert tempo_pronto(destino)
    assert sorted(p.name for p in destino.iterdir()) == ["logmel.npy", "quadros.npz"]
    with (destino / "logmel.npy").open("r+b") as stream:
        stream.truncate(200)
    assert not tempo_pronto(destino)
    assert not tempo_pronto(tmp_path / "inexistente")
