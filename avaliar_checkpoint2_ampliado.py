"""Calibra e avalia o Checkpoint 2 em lotes independentes do SoccerNet."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np
import soundfile as sf

from avaliar_ajustes_finais_checkpoint2 import RATIOS, mixed_candidates
from avaliar_alternativas_checkpoint2 import (
    AROUSAL, AROUSAL_RISE, BASELINE, CONTEXTS_S, FUSION, PERCENTILE, RMS_RISE, TOP_K, WEIGHTS,
    measure, predictions,
)
from avaliar_arousal import FAMILIES, TOL_S, arousal_detections, choose_percentile, read_games, write_csv
from avaliar_lote import VARIANTS, prf
from avaliar_rms_checkpoint2 import rms_windows
from ouvindo_o_gol import ANNOTATIONS_DIR, KEY_EVENTS, dedupe_events, normalized_features, robust_z, score_and_detect
from pontuar_arousal import HOP_S, MODEL, REVISION, SAMPLE_RATE, WINDOW_S

VOLUME = "so volume"
RMS = "rms 4 s"
AROUSAL_4 = "wav2vec 4 s"
AROUSAL_2 = "wav2vec 2 s"
MIXTURE = "mistura"
METHODS = (VOLUME, BASELINE, RMS, AROUSAL_4, AROUSAL_2, FUSION, AROUSAL_RISE, RMS_RISE, MIXTURE)
END_TOLERANCE_S = 1.0


@dataclass
class Half:
    game: str
    number: int
    events: list
    volume: np.ndarray
    pitch: np.ndarray
    rms: np.ndarray
    arousal_4: np.ndarray
    arousal_2: np.ndarray


def game_path(game):
    path = Path(game)
    if path.is_absolute() or len(path.parts) != 3 or ".." in path.parts:
        raise ValueError(f"Caminho de partida inválido: {game}")
    return path


def complete_windows(frames, window):
    return max(0, (frames - window * SAMPLE_RATE) // (HOP_S * SAMPLE_RATE) + 1)


def classic_features(wav, cache):
    status = wav.stat()
    info = sf.info(wav)
    if (info.samplerate, info.channels) != (SAMPLE_RATE, 1):
        raise ValueError(f"Esperado WAV mono a {SAMPLE_RATE} Hz: {wav}")
    expected = complete_windows(info.frames, WINDOW_S)
    if not expected:
        raise ValueError(f"Áudio menor que uma janela completa: {wav}")
    signature = np.array([status.st_size, status.st_mtime_ns])
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as previous:
                if np.array_equal(previous["wav"], signature):
                    volume, pitch, rms = (previous[key] for key in ("volume", "pitch", "rms"))
                    if (
                        volume.ndim == 1 and len(volume) and volume.shape == pitch.shape
                        and rms.shape == (expected,)
                        and all(np.isfinite(v).all() for v in (volume, pitch, rms))
                    ):
                        return info, volume, pitch, rms
        except (OSError, ValueError, KeyError, BadZipFile):
            pass

    volume, pitch, _ = normalized_features(wav)
    rms = rms_windows(wav)
    if rms.shape != (expected,) or not all(np.isfinite(v).all() for v in (volume, pitch, rms)):
        raise ValueError(f"Atributos inválidos: {wav}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache.parent, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, wav=signature, volume=volume, pitch=pitch, rms=rms)
            handle.close()
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Atributos clássicos preparados: {wav.parent.name} | tempo {wav.stem}", flush=True)
    return info, volume, pitch, rms


def read_scores(path, frames, window):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    starts = HOP_S * np.arange(complete_windows(frames, window))
    expected = np.column_stack((starts, starts + window / 2, starts + window))
    timestamps = np.array([tuple(float(row[k]) for k in ("inicio_s", "centro_s", "fim_s")) for row in rows])
    scores = np.array([float(row["arousal"]) for row in rows])
    if not len(starts) or not np.array_equal(timestamps, expected) or not np.isfinite(scores).all():
        raise ValueError(f"Escores ausentes ou fora da grade de {window} s: {path}")
    return robust_z(scores)


def read_annotations(division):
    rows = json.loads((ANNOTATIONS_DIR / f"{division}.json").read_text())["data"]
    result = {row["id"]: row["events"] for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Tempos duplicados na divisão {division}.")
    return result


def load_halves(games, division, audio, features, scores_4, scores_2):
    annotations = read_annotations(division)
    halves = []
    for game in games:
        path = game_path(game)
        for number in (1, 2):
            key = f"{game}/{number}_224p.mkv"
            if key not in annotations:
                raise ValueError(f"{key} está fora da divisão {division}.")
            info, volume, pitch, rms = classic_features(
                audio / path / f"{number}.wav", features / path / f"{number}.npz",
            )
            raw = [(event["position_ms"] / 1000, event["label"]) for event in annotations[key]]
            # SoccerNet annotations are on the video clock, which can outlast the audio stream slightly.
            if any(not 0 <= timestamp <= info.duration + END_TOLERANCE_S for timestamp, _ in raw):
                raise ValueError(f"Anotações fora da duração do áudio: {key} ({info.duration:.3f} s).")
            events = [(t, label) for t, label in dedupe_events(raw) if label in KEY_EVENTS]
            arousal_4 = read_scores(scores_4 / path / f"{number}.csv", info.frames, 4)
            arousal_2 = read_scores(scores_2 / path / f"{number}.csv", info.frames, 2)
            halves.append(Half(game, number, events, volume, pitch, rms, arousal_4, arousal_2))
    return halves


def as_ranking(detections):
    return [(peak, strength) for _, peak, _, _, strength in detections]


def percentile_fit(halves, attribute):
    inputs = [(h.game, h.number, None, h.events, getattr(h, attribute)) for h in halves]
    return choose_percentile(inputs)


def base_rankings(halves, arousal_percentile):
    result = {}
    for h in halves:
        primary = as_ranking(score_and_detect(h.volume, h.pitch, VARIANTS[BASELINE])[2])
        activation = as_ranking(arousal_detections(h.arousal_4, arousal_percentile))
        result[h.game, h.number] = (h.events, {
            BASELINE: primary, AROUSAL: activation, AROUSAL_RISE: h.arousal_4, RMS_RISE: h.rms,
        })
    return result


def calibrate(halves):
    activation_p, activation_table = percentile_fit(halves, "arousal_4")
    rms_p, rms_table = percentile_fit(halves, "rms")
    rows = []
    for method, winner, table in ((AROUSAL_4, activation_p, activation_table), (RMS, rms_p, rms_table)):
        for item in table:
            parameter = item["percentil"]
            metric = {key: value for key, value in item.items() if key != "percentil"}
            rows.append(dict(metodo=method, parametro=parameter, selecionado=parameter == winner, **metric))

    base = base_rankings(halves, activation_p)
    settings = {}
    for method, values in ((FUSION, WEIGHTS), (AROUSAL_RISE, CONTEXTS_S), (RMS_RISE, CONTEXTS_S)):
        candidates = [(value, measure(predictions(base, method, value), TOP_K)[0]) for value in values]
        winner = max(candidates, key=lambda item: item[1]["f1"])[0]
        settings[method] = winner
        rows.extend(dict(metodo=method, parametro=value, selecionado=value == winner, **metric)
                    for value, metric in candidates)

    fusion = list(predictions(base, FUSION, settings[FUSION]))
    rise = list(predictions(base, AROUSAL_RISE, settings[AROUSAL_RISE]))
    quotas = []
    for quota in RATIOS:
        mixed = [(g, number, events, mixed_candidates(f, r, quota))
                 for (g, number, events, f), (_, _, _, r) in zip(fusion, rise, strict=True)]
        quotas.append((quota, measure(mixed, TOP_K)[0]))
    quota = max(quotas, key=lambda item: item[1]["f1"])[0]
    rows.extend(dict(metodo=MIXTURE, parametro=value, selecionado=value == quota, **metric)
                for value, metric in quotas)
    return {
        "percentil_ativacao": activation_p,
        "percentil_rms": rms_p,
        "peso_ativacao_fusao": settings[FUSION],
        "contexto_subida_ativacao_s": settings[AROUSAL_RISE],
        "contexto_subida_rms_s": settings[RMS_RISE],
        "vagas_fusao_mistura": quota,
        "percentil_subida": PERCENTILE,
        "duracoes_ativacao_s": [4, 2],
        "passo_ativacao_s": HOP_S,
        "modelo_ativacao": MODEL,
        "revisao_ativacao": REVISION,
        "tolerancia_fim_audio_s": END_TOLERANCE_S,
        "top_k": TOP_K,
        "tolerancia_s": TOL_S,
    }, rows, base


def all_rankings(halves, config, base):
    ranks = {
        VOLUME: [(h.game, h.number, h.events, as_ranking(
            score_and_detect(h.volume, h.pitch, VARIANTS[VOLUME])[2])) for h in halves],
        BASELINE: list(predictions(base, BASELINE)),
        RMS: [(h.game, h.number, h.events, as_ranking(
            arousal_detections(h.rms, config["percentil_rms"]))) for h in halves],
        AROUSAL_4: list(predictions(base, AROUSAL)),
        AROUSAL_2: [(h.game, h.number, h.events, as_ranking(
            arousal_detections(h.arousal_2, config["percentil_ativacao"], window_s=2))) for h in halves],
        FUSION: list(predictions(base, FUSION, config["peso_ativacao_fusao"])),
        AROUSAL_RISE: list(predictions(base, AROUSAL_RISE, config["contexto_subida_ativacao_s"])),
        RMS_RISE: list(predictions(base, RMS_RISE, config["contexto_subida_rms_s"])),
    }
    quota = config["vagas_fusao_mistura"]
    ranks[MIXTURE] = [(g, number, events, mixed_candidates(f, r, quota))
                      for (g, number, events, f), (_, _, _, r) in
                      zip(ranks[FUSION], ranks[AROUSAL_RISE], strict=True)]
    return ranks


def aggregate(split, halves, config, base):
    metrics, classes, by_game, alerts = [], [], [], []
    ranks = all_rankings(halves, config, base)
    for method in METHODS:
        for selection in (("top3",) if method == MIXTURE else ("top3", "todos")):
            group = dict(lote=split, metodo=method, selecao=selection)
            games = defaultdict(list)
            for g, n, events, ranking in ranks[method]:
                candidates = ranking[:TOP_K] if selection == "top3" else ranking
                games[g].append((g, n, events, candidates))
            for target in ("lances", "gols"):
                hits, annotated = Counter(), Counter()
                numbers = Counter()
                for game, parts in games.items():
                    metric, game_hits, game_annotated, details = measure(parts, target=target)
                    by_game.append(dict(group, alvo=target, jogo=game, **metric))
                    numbers.update({key: metric[key] for key in ("acertos", "alertas", "anotados")})
                    hits.update(game_hits)
                    annotated.update(game_annotated)
                    if target == "lances":
                        alerts.extend(dict(group, **row) for row in details)
                precision, recall, f1 = prf(numbers["acertos"], numbers["alertas"], numbers["anotados"])
                metrics.append(dict(group, alvo=target, **numbers, precisao=precision, revocacao=recall, f1=f1))
                if target == "lances":
                    for label in dict.fromkeys(FAMILIES.values()):
                        count = annotated[label]
                        classes.append(dict(group, classe=label, acertos=hits[label], anotados=count,
                                            revocacao=hits[label] / count if count else None))
    return metrics, classes, by_game, alerts


def paired_list(by_game):
    metrics = {
        (r["jogo"], r["metodo"]): r for r in by_game
        if (r["lote"], r["alvo"], r["selecao"]) == ("avaliacao", "lances", "top3")
    }
    for game in dict.fromkeys(g for g, method in metrics if method == BASELINE):
        baseline, mixed = metrics[game, BASELINE], metrics[game, MIXTURE]
        yield dict(jogo=game, anotados=baseline["anotados"],
                   acertos_regra=baseline["acertos"], alertas_regra=baseline["alertas"],
                   acertos_mistura=mixed["acertos"], alertas_mistura=mixed["alertas"],
                   delta_acertos=mixed["acertos"] - baseline["acertos"])


def write_json(path, payload):
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, default=Path(".cache/checkpoint-2-ampliado"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--ajuste", type=Path)
    parser.add_argument("--avaliacao", type=Path)
    parser.add_argument("--escores-4s", type=Path)
    parser.add_argument("--escores-2s", type=Path)
    parser.add_argument("--saida", type=Path, default=Path("outputs/checkpoint-2-ampliado"))
    parser.add_argument("--divisao-ajuste", choices=("train", "valid"), default="train")
    parser.add_argument("--divisao-avaliacao", choices=("train", "valid"), default="valid")
    parser.add_argument("--permitir-jogos-antigos", action="store_true", help="Usado apenas para conferir o lote histórico.")
    prep = parser.add_mutually_exclusive_group()
    prep.add_argument("--preparar-jogo", action="append")
    prep.add_argument("--preparar-lista", type=Path)
    prep.add_argument("--somente-ajuste", action="store_true", help="Congela os parâmetros com o lote de ajuste.")
    args = parser.parse_args()
    root = args.raiz
    audio = args.audio or root / "audio"
    features = root / "classicos"

    if args.preparar_jogo or args.preparar_lista:
        games = args.preparar_jogo or read_games(args.preparar_lista)
        for game in dict.fromkeys(games):
            path = game_path(game)
            for number in (1, 2):
                classic_features(audio / path / f"{number}.wav", features / path / f"{number}.npz")
        return

    adjustment = read_games(args.ajuste or root / "amostra_ajuste.txt")
    evaluation = [] if args.somente_ajuste else read_games(args.avaliacao or root / "amostra_avaliacao.txt")
    if set(adjustment) & set(evaluation):
        parser.error("Ajuste e avaliação contêm partidas em comum.")
    if not args.permitir_jogos_antigos:
        old = set(read_games(Path("data/games_valid.txt")) + read_games(Path("data/games_valid_holdout.txt")))
        if old & (set(adjustment) | set(evaluation)):
            parser.error("Há partidas já usadas na rodada anterior.")

    scores_4 = args.escores_4s or root / "escores-4s"
    scores_2 = args.escores_2s or root / "escores-2s"
    fit = load_halves(adjustment, args.divisao_ajuste, audio, features, scores_4, scores_2)
    config, calibration, base = calibrate(fit)
    config.update(divisao_ajuste=args.divisao_ajuste, jogos_ajuste=adjustment)
    args.saida.mkdir(parents=True, exist_ok=True)
    frozen_path = args.saida / "configuracao_ajuste.json"
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text())
        if config != frozen:
            raise ValueError(f"O ajuste diverge da configuração já congelada em {frozen_path}.")
        config = frozen
    else:
        write_json(frozen_path, config)
    write_csv(args.saida / "calibracao.csv", calibration)
    print(f"Configuração de ajuste conferida em {frozen_path}", flush=True)
    if args.somente_ajuste:
        return

    config.update(divisao_avaliacao=args.divisao_avaliacao, jogos_avaliacao=evaluation)
    write_json(args.saida / "configuracao.json", config)

    fit_results = aggregate("ajuste", fit, config, base)
    holdout = load_halves(evaluation, args.divisao_avaliacao, audio, features, scores_4, scores_2)
    eval_results = aggregate("avaliacao", holdout, config, base_rankings(holdout, config["percentil_ativacao"]))
    names = ("metricas.csv", "por_classe.csv", "por_jogo.csv", "alertas.csv")
    for name, first, second in zip(names, fit_results, eval_results, strict=True):
        write_csv(args.saida / name, first + second)
    paired = list(paired_list(eval_results[2]))
    write_csv(args.saida / "comparacao_lista_curta.csv", paired)
    gains = sum(row["delta_acertos"] > 0 for row in paired)
    losses = sum(row["delta_acertos"] < 0 for row in paired)
    print(f"Avaliação concluída em {len(evaluation)} jogos. Mistura vs. regra: "
          f"{gains} ganhos, {len(paired) - gains - losses} empates, {losses} perdas.", flush=True)


if __name__ == "__main__":
    main()
