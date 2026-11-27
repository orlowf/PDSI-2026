"""Compare arousal and the Checkpoint 1 rules on separate SoccerNet game lists."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from avaliar_lote import VARIANTS, prf
from ouvindo_o_gol import (
    KEY_EVENTS,
    detect,
    load_events,
    match_detections,
    normalized_features,
    robust_z,
    score_and_detect,
)
from pontuar_arousal import HOP_S, SAMPLE_RATE, WINDOW_S

PERCENTIS = (95, 97, 98, 99)
TOL_S = 15
METHODS = ("so volume", "vol+bonus tom", "wav2vec 2.0")
FAMILIES = {
    "Goal": "gol",
    "Shots on target": "finalizacao no alvo",
    "Shots off target": "finalizacao para fora",
    "Penalty": "penalti",
    "Red card": "expulsao",
    "Yellow->red card": "expulsao",
}


def read_games(path):
    games = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not games or len(games) != len(set(games)):
        raise ValueError(f"Lista de jogos vazia ou com duplicatas: {path}")
    return games


def load_halves(games, scores_dir, *, window_s=WINDOW_S):
    for game in games:
        for half in (1, 2):
            wav = Path("data/audio") / game / f"{half}.wav"
            score_file = scores_dir / game / f"{half}.csv"
            with score_file.open(newline="") as file:
                rows = list(csv.DictReader(file))
            audio = sf.info(wav)
            expected = max(0, (audio.frames - window_s * SAMPLE_RATE) // (HOP_S * SAMPLE_RATE) + 1)
            times = np.array([float(row["centro_s"]) for row in rows])
            values = np.array([float(row["arousal"]) for row in rows])
            starts = HOP_S * np.arange(expected)
            if (
                (audio.samplerate, audio.channels) != (SAMPLE_RATE, 1)
                or not np.isfinite(values).all()
                or expected == 0
                or not np.array_equal(times, starts + window_s / 2)
                or not np.array_equal([float(row["inicio_s"]) for row in rows], starts)
                or not np.array_equal([float(row["fim_s"]) for row in rows], starts + window_s)
            ):
                raise ValueError(f"Escores ausentes ou fora da grade do áudio: {score_file}")
            _, annotations = load_events(game, half)
            events = [(s, label) for s, label in annotations if label in KEY_EVENTS]
            yield game, half, wav, events, robust_z(values)


def arousal_detections(scores, percentile, *, window_s=WINDOW_S):
    return detect(
        scores, float(np.percentile(scores, percentile)), HOP_S,
        hop_s=HOP_S, first_s=window_s / 2,
    )


def choose_percentile(halves):
    rows = []
    for percentile in PERCENTIS:
        hits, alerts, truth_count = 0, 0, 0
        for _, _, _, events, scores in halves:
            detections = arousal_detections(scores, percentile)
            hits += len(match_detections(detections, [s for s, _ in events], TOL_S))
            alerts += len(detections)
            truth_count += len(events)
        precision, recall, f1 = prf(hits, alerts, truth_count)
        rows.append(dict(percentil=percentile, acertos=hits, alertas=alerts,
                         anotados=truth_count, precisao=precision, revocacao=recall, f1=f1))
    selected = max(rows, key=lambda row: (row["f1"], -row["alertas"]))
    return selected["percentil"], rows


def compare(halves, split, percentile):
    totals = Counter()
    breakdown = Counter()
    details = []
    for game, half, wav, events, scores in halves:
        volume, pitch, _ = normalized_features(wav)
        methods = {method: score_and_detect(volume, pitch, VARIANTS[method])[2] for method in METHODS[:2]}
        methods[METHODS[-1]] = arousal_detections(scores, percentile)
        counts = Counter(FAMILIES[label] for _, label in events)
        for method, detections in methods.items():
            for n_picks in ("todos", "top3"):
                picks = detections if n_picks == "todos" else detections[:3]
                for target in ("lances", "gols"):
                    truth = events if target == "lances" else [(s, label) for s, label in events if label == "Goal"]
                    matches = match_detections(picks, [s for s, _ in truth], TOL_S)
                    key = (method, n_picks, target)
                    totals[key + ("acertos",)] += len(matches)
                    totals[key + ("alertas",)] += len(picks)
                    totals[key + ("anotados",)] += len(truth)
                    if target != "lances":
                        continue
                    for family, count in counts.items():
                        breakdown[method, n_picks, family, "anotados"] += count
                    for event_index in matches.values():
                        breakdown[method, n_picks, FAMILIES[truth[event_index][1]], "acertos"] += 1
                    if n_picks == "todos":
                        for i, (_, peak, _, _, strength) in enumerate(picks):
                            when, label = truth[matches[i]] if i in matches else (None, "")
                            details.append(dict(lote=split, jogo=game, tempo=half, metodo=method,
                                                pico_s=peak, intensidade=strength, evento_s=when, rotulo=label))
        print(f"{split} | {game} | tempo {half}", flush=True)

    metrics, families = [], []
    for method in METHODS:
        for n_picks in ("todos", "top3"):
            for target in ("lances", "gols"):
                key = (method, n_picks, target)
                hits, alerts, truth_count = (totals[key + (name,)] for name in ("acertos", "alertas", "anotados"))
                precision, recall, f1 = prf(hits, alerts, truth_count)
                metrics.append(dict(lote=split, metodo=method, selecao=n_picks, alvo=target,
                                    acertos=hits, alertas=alerts, anotados=truth_count,
                                    precisao=precision, revocacao=recall, f1=f1))
            for family in dict.fromkeys(FAMILIES.values()):
                hits = breakdown[method, n_picks, family, "acertos"]
                count = breakdown[method, n_picks, family, "anotados"]
                families.append(dict(lote=split, metodo=method, selecao=n_picks, classe=family,
                                     acertos=hits, anotados=count, revocacao=hits / count if count else None))
    return metrics, families, details


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ajuste", type=Path, required=True)
    parser.add_argument("--avaliacao", type=Path)
    parser.add_argument("--escores", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()
    adjustment = read_games(args.ajuste)
    evaluation = read_games(args.avaliacao) if args.avaliacao else []
    if set(adjustment) & set(evaluation):
        parser.error("As listas de ajuste e avaliação contêm jogos em comum.")

    adjustment_halves = list(load_halves(adjustment, args.escores))
    percentile, calibration = choose_percentile(adjustment_halves)
    args.saida.mkdir(parents=True, exist_ok=True)
    write_csv(args.saida / "calibracao.csv", [
        dict(row, selecionado=(row["percentil"] == percentile)) for row in calibration
    ])
    print(f"Percentil escolhido no ajuste: {percentile}", flush=True)
    for split, halves in (("ajuste", adjustment_halves), ("avaliacao", load_halves(evaluation, args.escores))):
        if split == "avaliacao" and not evaluation:
            continue
        for name, rows in zip(("metricas", "por_classe", "alertas"), compare(halves, split, percentile)):
            write_csv(args.saida / f"{split}_{name}.csv", rows)


if __name__ == "__main__":
    main()
