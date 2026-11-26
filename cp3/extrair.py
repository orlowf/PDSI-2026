"""Extrai log-mel e atributos de quadro de cada tempo de jogo do Checkpoint 3."""

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
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


RAIZ_ATRIBUTOS = Path(".cache/checkpoint-3/atributos")
RAIZ_MIDIA = Path(".cache/checkpoint-3/midia")
REGISTRO = Path("outputs/checkpoint-3/extracao.csv")
REVISAO_CP2 = Path(".cache/checkpoint-2-integral/revisao_midia.txt")
LIVRE_MINIMO = 10 * 10**9
CAMPOS_REGISTRO = ("jogo", "estado", "segundos", "motivo")


def _adquirir_padrao(*args, **kwargs):
    from adquirir_midias_checkpoint2_integral import acquire_game

    return acquire_game(*args, **kwargs)


def _reaproveitar_padrao(jogo):
    from executar_checkpoint2_integral import cached_wavs

    return cached_wavs(jogo)


def processar_jogo(jogo, raiz_atributos, raiz_midia, revisao, faixas, adquirir=None, reaproveitar=None):
    """Garante log-mel e atributos dos dois tempos; apaga só o áudio que este passo baixou."""
    destinos = {t: Path(raiz_atributos) / jogo.path / str(t) for t in (1, 2)}
    faltam = [t for t, destino in destinos.items() if not tempo_pronto(destino)]
    if not faltam:
        return "ja_existia"
    existentes = (reaproveitar or _reaproveitar_padrao)(jogo)
    if existentes:
        for t in faltam:
            extrair_tempo(existentes[t - 1], destinos[t])
        return "reaproveitado"
    pasta_baixada = Path(raiz_midia) / "audio" / jogo.path
    try:
        wavs = (adquirir or _adquirir_padrao)(jogo.path, jogo.source, jogo.annotations,
                                              Path(raiz_midia), revisao, faixas)
        for t in faltam:
            extrair_tempo(wavs[t - 1], destinos[t])
    finally:
        shutil.rmtree(pasta_baixada, ignore_errors=True)
    return "pronto"


def _registrar(registro, **linha):
    registro = Path(registro)
    registro.parent.mkdir(parents=True, exist_ok=True)
    novo = not registro.exists()
    with registro.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CAMPOS_REGISTRO)
        if novo:
            writer.writeheader()
        writer.writerow({campo: linha.get(campo, "") for campo in CAMPOS_REGISTRO})


def executar(jogos, raiz_atributos, raiz_midia, revisao, faixas, registro,
             adquirir=None, reaproveitar=None, uso_disco=shutil.disk_usage):
    for jogo in jogos:
        Path(raiz_atributos).mkdir(parents=True, exist_ok=True)
        livre = uso_disco(raiz_atributos).free
        if livre < LIVRE_MINIMO:
            _registrar(registro, jogo=jogo.path, estado="parado",
                       motivo=f"disco com {livre / 10**9:.1f} GB livres (mínimo 10 GB)")
            return
        inicio = time.monotonic()
        try:
            estado = processar_jogo(jogo, raiz_atributos, raiz_midia, revisao, faixas, adquirir, reaproveitar)
            motivo = ""
        except Exception as exc:  # um jogo ruim não interrompe a fila
            estado, motivo = "falha", f"{type(exc).__name__}: {str(exc)[:300]}"
            if type(exc).__name__ == "AccessError":
                _registrar(registro, jogo=jogo.path, estado="parado", motivo=motivo)
                return
        _registrar(registro, jogo=jogo.path, estado=estado,
                   segundos=f"{time.monotonic() - inicio:.1f}", motivo=motivo)
        print(f"{estado:13s} {jogo.path}", flush=True)


def _jogos_da_divisao():
    from cp3.divisao import ler_divisao
    from executar_checkpoint2_integral import games_from_annotations
    from ouvindo_o_gol import ANNOTATIONS_DIR

    divisao = ler_divisao()
    catalogo = {jogo.path: jogo for jogo in games_from_annotations(ANNOTATIONS_DIR)}
    return [catalogo[j] for j in divisao["treino"]] + [catalogo[j] for j in divisao["teste"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faixas", type=int, default=8, help="conexões HTTP paralelas por mídia")
    parser.add_argument("--limite", type=int, help="processa só os N primeiros jogos")
    args = parser.parse_args(argv)
    jogos = _jogos_da_divisao()[:args.limite]
    if REVISAO_CP2.is_file():
        revisao = REVISAO_CP2.read_text().strip()
    else:
        from adquirir_midias_checkpoint2_integral import resolve_revision

        revisao = resolve_revision()
    executar(jogos, RAIZ_ATRIBUTOS, RAIZ_MIDIA, revisao, args.faixas, REGISTRO)


if __name__ == "__main__":
    sys.exit(main())
