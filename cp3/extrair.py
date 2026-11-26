"""Extrai log-mel e atributos de quadro de cada tempo de jogo do Checkpoint 3."""

import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16000
N_FFT = 1024
HOP = 320
N_MELS = 64
FPS = SR // HOP
FAIXAS_HZ = {"e_baixa": (0, 300), "e_voz": (300, 3400), "e_alta": (3400, 8000)}
CHAVES_QUADRO = ("rms", "centroide", "rolloff", "fluxo", "zcr", *FAIXAS_HZ)
AMIN = 1e-10


def _magnitude(y):
    return np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True))


def _logmel_de(S, sr):
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_fft=N_FFT, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=1.0, amin=AMIN).T.astype(np.float16)


def _atributos_de(y, S, sr):
    potencia = S**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    diferenca = np.diff(S, axis=1, prepend=S[:, :1])
    atributos = {
        "rms": librosa.feature.rms(S=S, frame_length=N_FFT)[0],
        "centroide": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0],
        # log1p mantém o fluxo de trechos muito altos dentro da faixa do float16
        "fluxo": np.log1p(np.sqrt(np.sum(np.maximum(diferenca, 0) ** 2, axis=0))),
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP, center=True)[0],
    }
    for nome, (baixo, alto) in FAIXAS_HZ.items():
        faixa = (freqs >= baixo) & (freqs < alto)
        atributos[nome] = 10 * np.log10(potencia[faixa].sum(axis=0) + AMIN)
    n = S.shape[1]
    return {k: np.nan_to_num(v[:n], nan=0.0).astype(np.float16) for k, v in atributos.items()}


def logmel(y, sr):
    return _logmel_de(_magnitude(y), sr)


def atributos_quadro(y, sr):
    return _atributos_de(y, _magnitude(y), sr)


def _gravar_atomico(destino, nome, escrever):
    with tempfile.NamedTemporaryFile(dir=destino, prefix=f".{nome}.", delete=False) as stream:
        temporario = Path(stream.name)
        try:
            escrever(stream)
        except BaseException:
            stream.close()
            temporario.unlink(missing_ok=True)
            raise
    os.replace(temporario, destino / nome)


def extrair_tempo(wav, destino):
    y, sr = sf.read(wav, dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"Taxa de amostragem {sr} Hz em {wav}; esperado {SR} Hz.")
    y = y.mean(axis=1)
    S = _magnitude(y)
    mel = _logmel_de(S, sr)
    quadros = _atributos_de(y, S, sr)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    _gravar_atomico(destino, "quadros.npz", lambda f: np.savez(f, **quadros))
    _gravar_atomico(destino, "logmel.npy", lambda f: np.save(f, mel))
    return mel.shape[0]


def tempo_pronto(destino):
    destino = Path(destino)
    try:
        mel = np.load(destino / "logmel.npy")
        with np.load(destino / "quadros.npz") as quadros:
            series = {k: quadros[k] for k in CHAVES_QUADRO}
    except (OSError, ValueError, KeyError, EOFError):
        return False
    if mel.ndim != 2 or mel.shape[1] != N_MELS or not np.isfinite(mel).all():
        return False
    return all(s.shape == (mel.shape[0],) and np.isfinite(s).all() for s in series.values())
