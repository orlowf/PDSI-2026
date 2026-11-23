"""Test rank fusion and local audio changes with the Checkpoint 2 protocol."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from avaliar_arousal import FAMILIES, TOL_S, load_halves, read_games, write_csv
from avaliar_lote import prf
from avaliar_rms_checkpoint2 import rms_windows
from ouvindo_o_gol import detect, match_detections
from pontuar_arousal import HOP_S, WINDOW_S

BASELINE = "vol+bonus tom"
AROUSAL = "wav2vec 2.0"
FUSION = "fusao"
AROUSAL_RISE = "subida arousal"
RMS_RISE = "subida RMS"
TOP_K = 3
MERGE_S = 4
PERCENTILE = 98
WEIGHTS = (0.50, 0.25, 0.75)
CONTEXTS_S = (20, 12, 30)


def read_candidates(path):
    candidates = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["metodo"] in (BASELINE, AROUSAL):
                key = (row["jogo"], int(row["tempo"]), row["metodo"])
                candidates[key].append((float(row["pico_s"]), float(row["intensidade"])))
    return candidates


def fuse_candidates(baseline, arousal, arousal_weight):
    """Merge nearby candidate peaks and sum per-method reciprocal rank scores."""
    points = sorted(
        (time, method, weight / (rank + TOP_K))
        for method, weight, ranked in (
            (BASELINE, 1 - arousal_weight, baseline),
            (AROUSAL, arousal_weight, arousal),
        )
        for rank, (time, _) in enumerate(ranked, start=1)
    )
    groups = []
    for point in points:
        if not groups or point[0] - groups[-1][0][0] > MERGE_S:
            groups.append([])
        groups[-1].append(point)
    ranked = []
    for group in groups:
        best_by_method = {}
        for time, method, support in group:
            if method not in best_by_method or support > best_by_method[method][0]:
                best_by_method[method] = (support, time)
        _, peak = max(best_by_method.values(), key=lambda candidate: (candidate[0], -candidate[1]))
        ranked.append((peak, sum(support for support, _ in best_by_method.values())))
    return sorted(ranked, key=lambda candidate: (-candidate[1], candidate[0]))


def rise_candidates(values, context_s):
    """Compare each window with the recent non-overlapping windows."""
    offset = context_s // HOP_S
    previous_count = offset - WINDOW_S // HOP_S + 1
    if len(values) <= offset:
        return []
    history = np.lib.stride_tricks.sliding_window_view(values, previous_count)
    rise = values[offset:] - np.median(history[:len(values) - offset], axis=1)
    threshold = max(0.0, float(np.percentile(rise, PERCENTILE)))
    detections = detect(rise, threshold, HOP_S, hop_s=HOP_S, first_s=WINDOW_S / 2 + offset * HOP_S)
    return [(peak, strength) for _, peak, _, _, strength in detections]


def load_split(games, scores, candidates_file):
    candidates = read_candidates(candidates_file)
    halves = {}
    for game, half, wav, truth, arousal_scores in load_halves(games, scores):
        rms_scores = rms_windows(wav)
        if len(rms_scores) != len(arousal_scores):
            raise ValueError(f"Volume e ativação em grades diferentes: {wav}")
        halves[game, half] = (truth, {
            BASELINE: candidates[game, half, BASELINE],
            AROUSAL: candidates[game, half, AROUSAL],
            AROUSAL_RISE: arousal_scores,
            RMS_RISE: rms_scores,
        })
    return halves


def predictions(halves, method, setting=None):
    for (game, half), (events, features) in halves.items():
        if method == FUSION:
            ranked = fuse_candidates(features[BASELINE], features[AROUSAL], setting)
        elif method in (AROUSAL_RISE, RMS_RISE):
            ranked = rise_candidates(features[method], setting)
        else:
            ranked = features[method]
        yield game, half, events, ranked


def measure(predictions_by_half, top_k=None, target="lances"):
    hits, alerts, annotations = 0, 0, 0
    kind_hits, kind_annotations = Counter(), Counter()
    details = []
    for game, half, events, ranked in predictions_by_half:
        truth = events if target == "lances" else [(t, label) for t, label in events if label == "Goal"]
        ranked = ranked[:top_k]
        pairs = match_detections([(0, peak) for peak, _ in ranked], [t for t, _ in truth], TOL_S)
        hits += len(pairs)
        alerts += len(ranked)
        annotations += len(truth)
        kind_annotations.update(FAMILIES[label] for _, label in truth)
        kind_hits.update(FAMILIES[truth[i][1]] for i in pairs.values())
        if target == "lances" and top_k is None:
            for i, (peak, score) in enumerate(ranked):
                event_time, label = truth[pairs[i]] if i in pairs else (None, "")
                details.append(dict(jogo=game, tempo=half, ordem=i+1, pico_s=peak,
                                    escore=score, evento_s=event_time, rotulo=label))
    precision, recall, f1 = prf(hits, alerts, annotations)
    return (dict(acertos=hits, alertas=alerts, anotados=annotations, precisao=precision,
                 revocacao=recall, f1=f1), kind_hits, kind_annotations, details)


def tune(halves):
    settings = {}
    calibration = []
    for method, candidates in ((FUSION, WEIGHTS), (AROUSAL_RISE, CONTEXTS_S), (RMS_RISE, CONTEXTS_S)):
        rows = []
        for setting in candidates:
            metric, *_ = measure(predictions(halves, method, setting), TOP_K)
            rows.append(dict(metodo=method, parametro=setting, **metric))
        winner = max(rows, key=lambda row: (row["f1"], row["acertos"]))
        settings[method] = winner["parametro"]
        calibration.extend(dict(row, selecionado=row is winner) for row in rows)
    return settings, calibration


def compare(split, halves, settings):
    metrics, classes, alerts = [], [], []
    for method in (BASELINE, AROUSAL, *settings):
        ranked = list(predictions(halves, method, settings.get(method)))
        for selection, top_k in (("top3", TOP_K), ("todos", None)):
            for target in ("lances", "gols"):
                summary, hit_kind, annotated_kind, details = measure(ranked, top_k, target)
                group = dict(lote=split, metodo=method, selecao=selection)
                metrics.append(dict(group, alvo=target, **summary))
                if target == "lances":
                    for kind in dict.fromkeys(FAMILIES.values()):
                        count = annotated_kind[kind]
                        hits = hit_kind[kind]
                        classes.append(dict(group, classe=kind, acertos=hits, anotados=count,
                                            revocacao=hits / count if count else None))
                alerts.extend(dict(lote=split, metodo=method, **row) for row in details)
    return metrics, classes, alerts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("outputs/checkpoint-2")
    parser.add_argument("--ajuste", type=Path, default=Path("data/games_valid.txt"))
    parser.add_argument("--avaliacao", type=Path, default=Path("data/games_valid_holdout.txt"))
    parser.add_argument("--escores", type=Path, default=base / "escores")
    parser.add_argument("--alertas-originais", type=Path, default=base / "avaliacao")
    parser.add_argument("--saida", type=Path, default=base / "alternativas")
    args = parser.parse_args()
    adjustment, evaluation = read_games(args.ajuste), read_games(args.avaliacao)
    if set(adjustment) & set(evaluation):
        parser.error("As listas de ajuste e avaliação contêm jogos em comum.")

    fit = load_split(adjustment, args.escores, args.alertas_originais / "ajuste_alertas.csv")
    settings, calibration = tune(fit)
    args.saida.mkdir(parents=True, exist_ok=True)
    write_csv(args.saida / "calibracao.csv", calibration)
    print(f"Parâmetros congelados no ajuste: {settings}", flush=True)

    holdout = load_split(evaluation, args.escores, args.alertas_originais / "avaliacao_alertas.csv")
    for name, first, second in zip(
        ("metricas", "por_classe", "alertas"),
        compare("ajuste", fit, settings),
        compare("avaliacao", holdout, settings),
    ):
        write_csv(args.saida / f"{name}.csv", first + second)


if __name__ == "__main__":
    main()
