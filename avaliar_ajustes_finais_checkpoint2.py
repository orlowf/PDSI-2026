"""Evaluate mixed rankings and 2-second windows with checkpoint audio only."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from avaliar_alternativas_checkpoint2 import AROUSAL_RISE, FUSION, MERGE_S, TOP_K, measure
from avaliar_arousal import FAMILIES, arousal_detections, load_halves, read_games, write_csv
from ouvindo_o_gol import KEY_EVENTS, load_events

PERCENTILE = 98
RATIOS = (2, 1)
DURATIONS = (4, 2)


def read_rankings(path, split):
    ranked = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["lote"] == split and row["metodo"] in (FUSION, AROUSAL_RISE):
                key = row["jogo"], int(row["tempo"]), row["metodo"]
                if int(row["ordem"]) != len(ranked[key]) + 1:
                    raise ValueError(f"Ordem de alertas inválida: {key}")
                ranked[key].append((float(row["pico_s"]), float(row["escore"])))
    return ranked


def mixed_candidates(fusion, rise, fusion_quota):
    order = [FUSION] * fusion_quota + [AROUSAL_RISE] * (TOP_K - fusion_quota)
    streams = {FUSION: iter(fusion), AROUSAL_RISE: iter(rise)}
    other = {FUSION: AROUSAL_RISE, AROUSAL_RISE: FUSION}
    chosen = []
    for first in order:
        for source in (first, other[first]):
            candidate = next((
                (t, s) for t, s in streams[source] if all(abs(t - peak) > MERGE_S for peak, _ in chosen)
            ), None)
            if candidate is not None:
                chosen.append(candidate)
                break
    return chosen


def ranking_halves(games, rankings):
    halves = []
    for game in games:
        for half in (1, 2):
            _, annotations = load_events(game, half)
            truth = [(t, label) for t, label in annotations if label in KEY_EVENTS]
            halves.append((game, half, truth, rankings[game, half, FUSION], rankings[game, half, AROUSAL_RISE]))
    return halves


def mixture(halves, quota):
    return [(game, half, truth, mixed_candidates(fusion, rise, quota))
            for game, half, truth, fusion, rise in halves]


def audio_peaks(games, scores_dir, window_s):
    for game, half, _, truth, scores in load_halves(games, scores_dir, window_s=window_s):
        peaks = [(peak, strength) for _, peak, _, _, strength
                 in arousal_detections(scores, PERCENTILE, window_s=window_s)]
        yield game, half, truth, peaks


def report(split, method, predictions, selections):
    metrics, classes, details = [], [], []
    for selection in selections:
        chosen = predictions if selection == "todos" else [
            (game, half, truth, ranks[:TOP_K]) for game, half, truth, ranks in predictions
        ]
        group = dict(lote=split, metodo=method, selecao=selection)
        for target in ("lances", "gols"):
            metric, hits, annotated, alerts = measure(chosen, target=target)
            metrics.append(dict(group, alvo=target, **metric))
            if target == "lances":
                for label in dict.fromkeys(FAMILIES.values()):
                    count = annotated[label]
                    classes.append(dict(group, classe=label, acertos=hits[label], anotados=count,
                                        revocacao=hits[label] / count if count else None))
                details.extend(dict(group, **row) for row in alerts)
    return metrics, classes, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("outputs/checkpoint-2")
    parser.add_argument("--ajuste", type=Path, default=Path("data/games_valid.txt"))
    parser.add_argument("--avaliacao", type=Path, default=Path("data/games_valid_holdout.txt"))
    parser.add_argument("--alertas", type=Path, default=root / "alternativas/alertas.csv")
    parser.add_argument("--escores-4s", type=Path, default=root / "escores")
    parser.add_argument("--escores-2s", type=Path, default=root / "escores-2s")
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--somente-mistura", action="store_true")
    args = parser.parse_args()
    adjustment, evaluation = read_games(args.ajuste), read_games(args.avaliacao)
    if set(adjustment) & set(evaluation):
        parser.error("As listas de ajuste e avaliação contêm jogos em comum.")
    output = args.saida or root / ("mistura" if args.somente_mistura else "ajustes-finais")

    fit = ranking_halves(adjustment, read_rankings(args.alertas, "ajuste"))
    calibration = [
        dict(vagas_fusao=quota, vagas_subida=TOP_K - quota, **measure(mixture(fit, quota), TOP_K)[0])
        for quota in RATIOS
    ]
    # Keep the first quota in RATIOS when the adjustment F1 ties.
    selected = max(calibration, key=lambda row: row["f1"])
    quota = selected["vagas_fusao"]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "calibracao_mistura.csv",
              [dict(row, selecionado=row is selected) for row in calibration])
    print(f"Mistura escolhida no ajuste: {quota} da fusão e {TOP_K - quota} da subida", flush=True)

    test = ranking_halves(evaluation, read_rankings(args.alertas, "avaliacao"))
    combined = [[], [], []]
    for split, games, halves in (("ajuste", adjustment, fit), ("avaliacao", evaluation, test)):
        runs = [("mistura", mixture(halves, quota), ("top3",))]
        if not args.somente_mistura:
            for window_s, scores in zip(DURATIONS, (args.escores_4s, args.escores_2s)):
                runs.append((f"wav2vec {window_s} s", list(audio_peaks(games, scores, window_s)), ("top3", "todos")))
        for name, rows, selections in runs:
            for table, new_rows in zip(combined, report(split, name, rows, selections)):
                table.extend(new_rows)
    for name, rows in zip(("metricas", "por_classe", "alertas"), combined):
        write_csv(output / f"{name}.csv", rows)


if __name__ == "__main__":
    main()
