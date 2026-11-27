"""Ajusta e avalia os três métodos do Checkpoint 2, sem limite de alertas."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from avaliar_alternativas_checkpoint2 import BASELINE, FUSION, MERGE_S, WEIGHTS, fuse_candidates, measure
from avaliar_arousal import FAMILIES, PERCENTIS, TOL_S, arousal_detections
from avaliar_checkpoint2_ampliado import AROUSAL_4, END_TOLERANCE_S, as_ranking, game_path, read_scores, write_json
from avaliar_lote import VARIANTS, prf
from ouvindo_o_gol import ANNOTATIONS_DIR, KEY_EVENTS, dedupe_events, score_and_detect
from pontuar_arousal import HOP_S, MODEL, REVISION, SAMPLE_RATE, WINDOW_S

METHODS = (BASELINE, AROUSAL_4, FUSION)
LOTS = ("ajuste", "avaliacao")
SOURCE_SPLITS = ("train", "valid", "test")
PENDING_FIELDS = ("jogo", "divisao_origem", "lote", "categoria", "motivo")


@dataclass
class Half:
    game: str
    number: int
    events: list
    volume: np.ndarray
    pitch: np.ndarray
    arousal: np.ndarray


def read_manifest(path):
    games = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = ("jogo", "divisao_origem", "lote")
        if not set(fields).issubset(reader.fieldnames or ()):
            raise ValueError(f"O manifesto deve conter {', '.join(fields)}: {path}")
        for row in reader:
            game, source, lot = (row[field] for field in fields)
            if not game or game in games or source not in SOURCE_SPLITS or lot not in LOTS:
                raise ValueError(f"Linha {reader.line_num} inválida no manifesto: {path}")
            game_path(game)
            games[game] = (source, lot)
    if {lot for _, lot in games.values()} != set(LOTS):
        raise ValueError("O manifesto deve conter os lotes de ajuste e avaliação.")
    return games


def read_ready(path, games):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ready = set(lines)
    if len(ready) != len(lines) or not ready.issubset(games):
        raise ValueError(f"Lista de partidas processadas duplicada ou fora do manifesto: {path}")
    return ready


def read_source_annotations(root, sources):
    annotations = {}
    for source in SOURCE_SPLITS:
        if source not in sources:
            continue
        path = root / f"{source}.json"
        events = {}
        for row in json.loads(path.read_text(encoding="utf-8"))["data"]:
            key = row["id"]
            if key in events:
                raise ValueError(f"Tempo duplicado no gabarito: {source}/{key}")
            events[key] = row["events"]
        annotations[source] = events
    return annotations


def load_half(root, annotations, game, source, number):
    path = game_path(game)
    features = root / "classicos" / path / f"{number}.npz"
    with np.load(features, allow_pickle=False) as cache:
        frames = cache["frames"]
        volume, pitch = cache["volume"], cache["pitch"]
        if (
            frames.size != 1 or not np.issubdtype(frames.dtype, np.integer)
            or int(frames.item()) < WINDOW_S * SAMPLE_RATE
            or volume.ndim != 1 or not len(volume) or volume.shape != pitch.shape
            or not all(np.isfinite(value).all() for value in (volume, pitch))
        ):
            raise ValueError(f"Atributos clássicos inválidos: {features}")
        frames = int(frames.item())

    key = f"{game}/{number}_224p.mkv"
    try:
        raw_annotations = annotations[source][key]
    except KeyError as error:
        raise ValueError(f"Tempo ausente no gabarito de {source}: {key}") from error
    raw_events = [(row["position_ms"] / 1000, row["label"]) for row in raw_annotations]
    duration = frames / SAMPLE_RATE
    if any(not 0 <= second <= duration + END_TOLERANCE_S for second, _ in raw_events):
        raise ValueError(f"Anotação fora do áudio: {source}/{key} ({duration:.3f} s).")
    events = [(second, label) for second, label in dedupe_events(raw_events) if label in KEY_EVENTS]
    arousal = read_scores(root / "escores-4s" / path / f"{number}.csv", frames, WINDOW_S)
    return Half(game, number, events, volume, pitch, arousal)


def rank_rule(halves, percentile):
    for half in halves:
        _, _, detections = score_and_detect(half.volume, half.pitch, VARIANTS[BASELINE], percentile=percentile)
        yield half.game, half.number, half.events, as_ranking(detections)


def rank_arousal(halves, percentile):
    for half in halves:
        detections = arousal_detections(half.arousal, percentile)
        yield half.game, half.number, half.events, as_ranking(detections)


def rank_fusion(rule, arousal, weight):
    for (game, number, events, ranked_rule), (_, _, _, ranked_arousal) in zip(rule, arousal, strict=True):
        yield game, number, events, fuse_candidates(ranked_rule, ranked_arousal, weight)


def select_parameter(method, options, reference, rankings):
    rows = []
    for value in options:
        metrics, *_ = measure(rankings(value))
        rows.append(dict(metodo=method, parametro=value, **metrics))
    winner = max(rows, key=lambda row: (row["f1"], -abs(row["parametro"] - reference), -row["alertas"]))
    return winner["parametro"], [dict(row, selecionado=row is winner) for row in rows]


def calibrate(halves):
    rule_percentile, rule_rows = select_parameter(BASELINE, PERCENTIS, 98, lambda p: rank_rule(halves, p))
    arousal_percentile, arousal_rows = select_parameter(AROUSAL_4, PERCENTIS, 98, lambda p: rank_arousal(halves, p))
    rule = list(rank_rule(halves, rule_percentile))
    arousal = list(rank_arousal(halves, arousal_percentile))
    weight, fusion_rows = select_parameter(FUSION, WEIGHTS, 0.5, lambda w: rank_fusion(rule, arousal, w))
    parameters = dict(percentil_regra=rule_percentile, percentil_ativacao=arousal_percentile,
                      peso_ativacao_fusao=weight)
    return parameters, rule_rows + arousal_rows + fusion_rows


def predictions(halves, parameters):
    rule = list(rank_rule(halves, parameters["percentil_regra"]))
    arousal = list(rank_arousal(halves, parameters["percentil_ativacao"]))
    fusion = rank_fusion(rule, arousal, parameters["peso_ativacao_fusao"])
    return {BASELINE: rule, AROUSAL_4: arousal, FUSION: fusion}


def collect_results(lot, halves, parameters, games):
    individual, breakdown, alerts = [], [], []
    for method, predictions_by_half in predictions(halves, parameters).items():
        by_game = defaultdict(list)
        for row in predictions_by_half:
            by_game[row[0]].append(row)
        for game, parts in by_game.items():
            source, _ = games[game]
            metrics, hits, truth, details = measure(parts)
            group = dict(lote=lot, metodo=method, selecao="todos", alvo="lances")
            individual.append(dict(group, jogo=game, divisao_origem=source, **metrics))
            breakdown.append((method, hits, truth))
            alerts.extend(dict(group, divisao_origem=source, **row) for row in details)
    return individual, breakdown, alerts


def summarize(lot, individual, breakdown):
    metrics, classes = [], []
    for method in METHODS:
        group = dict(lote=lot, metodo=method, selecao="todos", alvo="lances")
        matching = [row for row in individual if row["metodo"] == method]
        totals = {key: sum(row[key] for row in matching) for key in ("acertos", "alertas", "anotados")}
        precision, recall, f1 = prf(totals["acertos"], totals["alertas"], totals["anotados"])
        metrics.append(dict(group, jogos=len(matching), tempos=2 * len(matching), **totals,
                            precisao=precision, revocacao=recall, f1=f1))
        hits, truth = Counter(), Counter()
        for selected, game_hits, game_truth in breakdown:
            if selected == method:
                hits.update(game_hits)
                truth.update(game_truth)
        for family in dict.fromkeys(FAMILIES.values()):
            classes.append(dict(group, classe=family, acertos=hits[family], anotados=truth[family],
                                revocacao=hits[family] / truth[family] if truth[family] else None))
    return metrics, classes


def pending_category(reason):
    text = reason.casefold()
    if not text:
        return "Sem motivo registrado"
    if any(term in text for term in ("connectionerror", "readtimeout", "connecttimeout", "timeouterror")):
        return "Conexão ou transferência"
    if "lacuna" in text or "descontinu" in text:
        return "Lacunas no áudio"
    if any(term in text for term in ("sincron", "relógio", "cronologia", "anotação fora", "duração")):
        return "Sincronização sem comprovação"
    if any(term in text for term in ("decodifica", "silêncio", "sha256", "hash", "integridade", "corrompido")):
        return "Qualidade da mídia"
    if "credencial" in text:
        return "Acesso à mídia"
    return "Outra falha registrada"


def pending_games(root, games, ready):
    status_path = root / "andamento.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    failures = status.get("falhas", {})
    rows = []
    for game, (source, lot) in games.items():
        if game in ready:
            continue
        reason = failures.get(game, {}).get("motivo", "")
        rows.append(dict(jogo=game, divisao_origem=source, lote=lot,
                         categoria=pending_category(reason), motivo=reason))
    return rows


def write_csv(path, rows, *, fields=None):
    with NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", newline="", suffix=".part", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def summary_markdown(config, metrics, pending):
    labels = {BASELINE: "Regra de volume e tom", AROUSAL_4: "wav2vec · 4 s", FUSION: "Fusão"}
    predicted = config["jogos_previstos"]
    processed = config["jogos_processados"]
    lines = [
        "# Checkpoint 2 · Rodada ampliada",
        "",
        f"Foram processadas **{processed} de {predicted} partidas elegíveis** do manifesto. "
        f"A lista de {predicted - processed} partidas sem resultado e seus motivos está em "
        "[pendencias.csv](pendencias.csv).",
        "",
        "| Lote | Partidas processadas | Partidas elegíveis |",
        "| --- | ---: | ---: |",
    ]
    for lot, title in (("ajuste", "Ajuste"), ("avaliacao", "Avaliação")):
        count = len(config["lotes"][lot]["jogos"])
        eligible = count + len(config["lotes"][lot]["ausentes"])
        lines.append(f"| {title} | {count} | {eligible} |")

    if pending:
        lines.extend([
            "",
            "## Partidas sem resultado nesta rodada",
            "",
            "Cada partida continua no manifesto. A tabela agrupa os motivos registrados. "
            "O arquivo por partida traz o texto registrado quando disponível. "
            "[Partidas e motivos](pendencias.csv)",
            "",
            "| Categoria | Divisão de origem | Lote | Partidas |",
            "| --- | --- | --- | ---: |",
        ])
        counts = Counter((row["categoria"], row["divisao_origem"], row["lote"]) for row in pending)
        for (category, source, lot), count in sorted(counts.items()):
            lines.append(f"| {category} | {source} | {lot} | {count} |")

    weight = f"{config['peso_ativacao_fusao']:.0%}"
    lines.extend([
        "",
        f"No ajuste, o F1 selecionou o percentil **{config['percentil_regra']}** para a regra, "
        f"o percentil **{config['percentil_ativacao']}** para o wav2vec e o peso de **{weight}** "
        "do wav2vec na fusão. Esses parâmetros foram fixados antes de calcular as métricas de avaliação.",
        "",
        f"As contagens consideram todos os alertas. Cada anotação pode receber um acerto, "
        f"com tolerância de {config['tolerancia_s']} segundos. "
        "[Tabela de calibração](calibracao.csv)",
    ])
    for lot, title, description in (
        ("avaliacao", "Avaliação separada", "Partidas do lote reservado para avaliar os métodos."),
        ("geral", "Conjunto geral", "Reúne as partidas processadas dos lotes de ajuste e avaliação."),
    ):
        lines.extend([
            "",
            f"## {title}",
            "",
            description,
            "",
            "| Método | Acertos | Alertas | Anotações | Precisão | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in metrics:
            if row["lote"] != lot:
                continue
            counts = [f"{row[key]:,}".replace(",", ".") for key in ("acertos", "alertas", "anotados")]
            rates = [f"{row[key]:.1%}".replace(".", ",") for key in ("precisao", "revocacao")]
            f1 = f"{row['f1']:.3f}".replace(".", ",")
            values = [labels[row["metodo"]], *counts, *rates, f1]
            lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "[Métricas completas](metricas.csv) · [Métricas por classe](por_classe.csv)", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifesto", type=Path, required=True)
    parser.add_argument("--raiz", type=Path, default=Path(".cache/checkpoint-2-integral"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/checkpoint-2-integral"))
    parser.add_argument("--anotacoes", type=Path, default=ANNOTATIONS_DIR)
    args = parser.parse_args()

    games = read_manifest(args.manifesto)
    ready = read_ready(args.raiz / "prontos.txt", games)
    pending = pending_games(args.raiz, games, ready)
    selected = {lot: [game for game, (_, name) in games.items() if name == lot and game in ready] for lot in LOTS}
    if not all(selected.values()):
        parser.error("A avaliação requer pelo menos uma partida processada em cada lote.")
    annotations = read_source_annotations(args.anotacoes, {source for source, _ in games.values()})

    fit = [load_half(args.raiz, annotations, game, games[game][0], number)
           for game in selected["ajuste"] for number in (1, 2)]
    parameters, calibration = calibrate(fit)
    print(f"Parâmetros ajustados em {len(selected['ajuste'])} partidas: {parameters}", flush=True)

    holdout = [load_half(args.raiz, annotations, game, games[game][0], number)
               for game in selected["avaliacao"] for number in (1, 2)]
    all_metrics, all_classes, individual, breakdown, alerts = [], [], [], [], []
    for lot, halves in (("ajuste", fit), ("avaliacao", holdout)):
        records, families, details = collect_results(lot, halves, parameters, games)
        metrics, classes = summarize(lot, records, families)
        all_metrics.extend(metrics)
        all_classes.extend(classes)
        individual.extend(records)
        breakdown.extend(families)
        alerts.extend(details)
    metrics, classes = summarize("geral", individual, breakdown)
    all_metrics.extend(metrics)
    all_classes.extend(classes)

    config = dict(
        versao_protocolo=1,
        manifesto_sha256=sha256(args.manifesto.read_bytes()).hexdigest(),
        metodos=METHODS,
        selecao="todos",
        criterio_ajuste="f1",
        **parameters,
        duracao_janela_ativacao_s=WINDOW_S,
        passo_ativacao_s=HOP_S,
        modelo_ativacao=MODEL,
        revisao_ativacao=REVISION,
        distancia_fusao_s=MERGE_S,
        tolerancia_fim_audio_s=END_TOLERANCE_S,
        tolerancia_s=TOL_S,
        jogos_previstos=len(games),
        jogos_processados=len(ready),
        universo_processado=len(ready) == len(games),
        lotes={
            lot: dict(jogos=selected[lot], ausentes=[
                game for game, (_, name) in games.items() if name == lot and game not in ready
            ])
            for lot in LOTS
        },
    )
    args.saida.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("calibracao.csv", calibration), ("metricas.csv", all_metrics), ("por_classe.csv", all_classes),
        ("por_jogo.csv", individual), ("alertas.csv", alerts),
    ):
        write_csv(args.saida / name, rows)
    write_json(args.saida / "configuracao.json", config)
    write_csv(args.saida / "pendencias.csv", pending, fields=PENDING_FIELDS)
    (args.saida / "resumo.md").write_text(summary_markdown(config, all_metrics, pending), encoding="utf-8")
    print(f"Resultados salvos em {args.saida}: {len(ready)}/{len(games)} partidas, "
          f"{len(selected['avaliacao'])} na avaliação separada.", flush=True)


if __name__ == "__main__":
    main()
