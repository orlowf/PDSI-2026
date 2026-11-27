import csv
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cp3.extrair import SR, executar, processar_jogo, tempo_pronto

Jogo = namedtuple("Jogo", "path source annotations")
JOGO = Jogo("liga/2016/A 1 - 0 B", "train", {1: [], 2: []})


def escrever_wav(path, segundos=2.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.random.default_rng(0).normal(0, 0.1, int(SR * segundos)).astype(np.float32), SR)
    return path


class AdquirirFalso:
    def __init__(self, falhar=False):
        self.chamadas = 0
        self.falhar = falhar

    def __call__(self, game, source, annotations, job_root, revision, ranges):
        self.chamadas += 1
        pasta = Path(job_root) / "audio" / game
        wavs = [escrever_wav(pasta / f"{t}.wav") for t in (1, 2)]
        if self.falhar:
            raise RuntimeError("conexão caiu")
        return wavs


def sem_cache(_jogo):
    return None


def wavs_em(raiz):
    return list(Path(raiz).rglob("*.wav"))


def test_apaga_wav_baixado(tmp_path):
    adquirir = AdquirirFalso()
    estado = processar_jogo(JOGO, tmp_path / "atr", tmp_path / "midia", "rev", 1,
                            adquirir=adquirir, reaproveitar=sem_cache)
    assert estado == "pronto"
    assert all(tempo_pronto(tmp_path / "atr" / JOGO.path / str(t)) for t in (1, 2))
    assert wavs_em(tmp_path / "midia") == []


def test_nao_apaga_wav_preexistente(tmp_path):
    existentes = [escrever_wav(tmp_path / "cache" / f"{t}.wav") for t in (1, 2)]
    adquirir = AdquirirFalso()
    estado = processar_jogo(JOGO, tmp_path / "atr", tmp_path / "midia", "rev", 1,
                            adquirir=adquirir, reaproveitar=lambda _j: existentes)
    assert estado == "reaproveitado"
    assert adquirir.chamadas == 0
    assert all(p.is_file() for p in existentes)


def test_retoma(tmp_path):
    atr = tmp_path / "atr"
    processar_jogo(JOGO, atr, tmp_path / "midia", "rev", 1, adquirir=AdquirirFalso(), reaproveitar=sem_cache)
    pronto = atr / JOGO.path / "1" / "logmel.npy"
    instante = pronto.stat().st_mtime_ns
    with (atr / JOGO.path / "2" / "logmel.npy").open("r+b") as stream:
        stream.truncate(100)
    estado = processar_jogo(JOGO, atr, tmp_path / "midia", "rev", 1, adquirir=AdquirirFalso(), reaproveitar=sem_cache)
    assert estado == "pronto"
    assert pronto.stat().st_mtime_ns == instante
    assert tempo_pronto(atr / JOGO.path / "2")


def test_falha_nao_deixa_wav(tmp_path):
    with pytest.raises(RuntimeError):
        processar_jogo(JOGO, tmp_path / "atr", tmp_path / "midia", "rev", 1,
                       adquirir=AdquirirFalso(falhar=True), reaproveitar=sem_cache)
    assert wavs_em(tmp_path / "midia") == []


def test_ja_existia(tmp_path):
    atr = tmp_path / "atr"
    processar_jogo(JOGO, atr, tmp_path / "midia", "rev", 1, adquirir=AdquirirFalso(), reaproveitar=sem_cache)
    adquirir = AdquirirFalso()
    estado = processar_jogo(JOGO, atr, tmp_path / "midia", "rev", 1, adquirir=adquirir, reaproveitar=sem_cache)
    assert estado == "ja_existia"
    assert adquirir.chamadas == 0


def test_para_com_disco_cheio(tmp_path):
    Uso = namedtuple("Uso", "total used free")
    adquirir = AdquirirFalso()
    registro = tmp_path / "extracao.csv"
    executar([JOGO], tmp_path / "atr", tmp_path / "midia", "rev", 1, registro,
             adquirir=adquirir, reaproveitar=sem_cache, uso_disco=lambda _p: Uso(1, 1, 5 * 10**9))
    assert adquirir.chamadas == 0
    linhas = list(csv.DictReader(registro.open()))
    assert linhas[-1]["estado"] == "parado"
    assert "disco" in linhas[-1]["motivo"]


def test_falha_de_um_jogo_segue_para_o_proximo(tmp_path):
    outro = Jogo("liga/2016/C 0 - 0 D", "train", {1: [], 2: []})
    chamadas = []

    def adquirir(game, *args):
        chamadas.append(game)
        if game == JOGO.path:
            raise RuntimeError("falhou")
        return AdquirirFalso()(game, *args)

    registro = tmp_path / "extracao.csv"
    executar([JOGO, outro], tmp_path / "atr", tmp_path / "midia", "rev", 1, registro,
             adquirir=adquirir, reaproveitar=sem_cache)
    estados = {l["jogo"]: l["estado"] for l in csv.DictReader(registro.open())}
    assert estados == {JOGO.path: "falha", outro.path: "pronto"}


def test_executar_limpa_transferencias_de_execucao_morta(tmp_path):
    sobra = tmp_path / "midia" / "transferencias" / "soccer-abc" / "HQ" / "media.mkv"
    sobra.parent.mkdir(parents=True)
    sobra.write_bytes(b"parcial")
    escrever_wav(tmp_path / "midia" / "transferencias" / "soccer-abc" / "224p" / "audio.wav")
    executar([JOGO], tmp_path / "atr", tmp_path / "midia", "rev", 1, tmp_path / "extracao.csv",
             adquirir=AdquirirFalso(), reaproveitar=sem_cache)
    assert not (tmp_path / "midia" / "transferencias").exists()
    assert wavs_em(tmp_path / "midia") == []


def test_main_recuperar_passa_recovery_para_acquire_game(tmp_path, monkeypatch):
    import cp3.extrair as extrair

    recebidos = []
    monkeypatch.setattr(extrair, "_jogos_da_divisao", lambda: [JOGO])
    monkeypatch.setattr(extrair, "REVISAO_CP2", tmp_path / "revisao.txt")
    (tmp_path / "revisao.txt").write_text("a" * 40)

    def executar_falso(jogos, *args, adquirir=None, **kwargs):
        recebidos.append(adquirir)

    monkeypatch.setattr(extrair, "executar", executar_falso)
    extrair.main(["--recuperar"])
    extrair.main([])
    assert recebidos[0].keywords == {"recovery": True}
    assert recebidos[1] is None
