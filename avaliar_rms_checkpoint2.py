"""RMS control with the same windows and calibration as the wav2vec experiment."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from avaliar_arousal import TOL_S, arousal_detections, choose_percentile, read_games, write_csv
from avaliar_lote import prf
from ouvindo_o_gol import KEY_EVENTS, load_events, match_detections, robust_z
from pontuar_arousal import HOP_S, SAMPLE_RATE, WINDOW_S


def rms_windows(wav):
    waveform, rate = sf.read(wav, dtype="float32")
    if rate != SAMPLE_RATE or waveform.ndim != 1:
        raise ValueError(f"Esperado WAV mono a 16 kHz: {wav}")
    hop_samples = HOP_S * rate
    span = WINDOW_S // HOP_S
    if len(waveform) < span * hop_samples:
        raise ValueError(f"Áudio menor que uma janela completa: {wav}")
    usable = len(waveform) // hop_samples * hop_samples
    blocks = waveform[:usable].reshape(-1, hop_samples)
    power = np.einsum("ij,ij->i", blocks, blocks, dtype="float64") / hop_samples
    window_power = np.convolve(power, np.full(span, 1 / span), mode="valid")
    return robust_z(10 * np.log10(np.maximum(window_power, 1e-20)))


def load_rms(games):
    for game in games:
        for half in (1, 2):
            wav = Path("data/audio") / game / f"{half}.wav"
            _, annotations = load_events(game, half)
            events = [(t, label) for t, label in annotations if label in KEY_EVENTS]
            yield game, half, wav, events, rms_windows(wav)


def measure(halves, percentile):
    detected = [(arousal_detections(scores, percentile), events) for _, _, _, events, scores in halves]
    for selection in ("todos", "top3"):
        for target in ("lances", "gols"):
            hits, alerts, annotated = 0, 0, 0
            for detections, events in detected:
                chosen = detections if selection == "todos" else detections[:3]
                truth = [t for t, label in events if target == "lances" or label == "Goal"]
                hits += len(match_detections(chosen, truth, TOL_S))
                annotated += len(truth)
                alerts += len(chosen)
            precision, recall, f1 = prf(hits, alerts, annotated)
            yield dict(selecao=selection, alvo=target, acertos=hits, alertas=alerts,
                       anotados=annotated, precisao=precision, revocacao=recall, f1=f1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ajuste", type=Path, default=Path("data/games_valid.txt"))
    parser.add_argument("--avaliacao", type=Path, default=Path("data/games_valid_holdout.txt"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/checkpoint-2/controle-rms"))
    args = parser.parse_args()
    adjustment = read_games(args.ajuste)
    evaluation = read_games(args.avaliacao)
    if set(adjustment) & set(evaluation):
        parser.error("As listas de ajuste e avaliação contêm jogos em comum.")
    fit = list(load_rms(adjustment))
    percentile, tuning = choose_percentile(fit)
    args.saida.mkdir(parents=True, exist_ok=True)
    write_csv(args.saida / "calibracao.csv", [
        dict(row, selecionado=(row["percentil"] == percentile)) for row in tuning
    ])
    print(f"Controle RMS 4 s/2 s | percentil escolhido no ajuste: {percentile}", flush=True)
    for split, halves in (("ajuste", fit), ("avaliacao", load_rms(evaluation))):
        write_csv(args.saida / f"{split}_metricas.csv", list(measure(halves, percentile)))


if __name__ == "__main__":
    main()
