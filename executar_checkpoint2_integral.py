"""Recalibra o Checkpoint 2 em uma divisão própria das partidas anotadas."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from shutil import copy2
import subprocess
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf
import torch

from avaliar_checkpoint2_ampliado import classic_features, game_path, read_scores
from pontuar_arousal import MODEL, REVISION, load_model, score_audio
from pontuar_checkpoint2_ampliado import ModelWithInputDtype
from preparar_midias_checkpoint2_ampliado import MediaError, atomic_text, check_audio

SEED = "pdsi-cp2-integral-2026-v1"
SOURCE_SPLITS = ("train", "valid", "test")
ORIGINAL = Path(".cache/checkpoint-2-ampliado")
CLASSICS = "classicos"
SCORES = "escores-4s"
SPLIT_FIELDS = ("jogo", "divisao_origem", "lote")


@dataclass(frozen=True)
class Game:
    path: str
    source: str
    subset: str
    annotations: dict


def digest(path):
    return sha256(f"{SEED}|{path}".encode()).hexdigest()


def games_from_annotations(annotation_root):
    catalog = {}
    for source in SOURCE_SPLITS:
        rows = json.loads((annotation_root / f"{source}.json").read_text())["data"]
        groups = defaultdict(dict)
        for row in rows:
            path, filename = row["id"].rsplit("/", 1)
            if filename not in ("1_224p.mkv", "2_224p.mkv"):
                raise ValueError(f"{source}: formato de anotação inesperado: {row['id']}")
            game_path(path)
            half = int(filename[0])
            if half in groups[path]:
                raise ValueError(f"Tempo duplicado em {source}: {row['id']}")
            groups[path][half] = row["events"]
        for path, halves in groups.items():
            if halves.keys() != {1, 2} or path in catalog:
                raise ValueError(f"Partida repetida ou sem os dois tempos: {path}")
            catalog[path] = (source, halves)

    competitions = defaultdict(list)
    for path in catalog:
        competitions[path.split("/")[0]].append(path)
    quotas = {name: len(games) // 5 for name, games in competitions.items()}
    missing = (len(catalog) + 2) // 5 - sum(quotas.values())
    by_remainder = sorted(competitions, key=lambda name: (-(len(competitions[name]) % 5), name))
    for name in by_remainder[:missing]:
        quotas[name] += 1
    adjust = {
        path for name, games in competitions.items()
        for path in sorted(games, key=digest)[:quotas[name]]
    }
    return [
        Game(path, source, "ajuste" if path in adjust else "avaliacao", halves)
        for path, (source, halves) in sorted(catalog.items(), key=lambda item: digest(item[0]))
    ]


def write_plan(games, root, output):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=SPLIT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(dict(jogo=g.path, divisao_origem=g.source, lote=g.subset) for g in games)
    text = stream.getvalue()
    manifest = root / "manifesto.csv"
    if manifest.is_file() and manifest.read_text() != text:
        raise ValueError("O manifesto existente diverge da divisão calculada. Use outra pasta de execução.")
    atomic_text(manifest, text)
    atomic_text(output / "manifesto.csv", text)
    by_competition = defaultdict(Counter)
    for game in games:
        by_competition[game.path.split("/")[0]][game.subset] += 1
    plan = {
        "semente": SEED,
        "fracao_ajuste": 0.2,
        "divisao_por": "partida, estratificada por competição",
        "origem": dict(Counter(g.source for g in games)),
        "lotes": dict(Counter(g.subset for g in games)),
        "competicoes": dict(by_competition),
        "model": MODEL,
        "revisao_modelo": REVISION,
        "janela_s": 4,
        "passo_s": 2,
    }
    atomic_text(output / "protocolo.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    return manifest


def checked_json(path, status):
    atomic_text(path, json.dumps(status, ensure_ascii=False, indent=2) + "\n")


def record_status(root, output, games, ready, failures, state):
    atomic_text(root / "prontos.txt", "".join(f"{g.path}\n" for g in games if g.path in ready))
    status = {
        "estado": state,
        "total": len(games),
        "prontos": len(ready),
        "prontos_por_lote": dict(Counter(g.subset for g in games if g.path in ready)),
        "falhas": failures,
    }
    checked_json(root / "andamento.json", status)
    checked_json(output / "andamento.json", status)


def feature_path(root, game, half):
    return root / CLASSICS / game.path / f"{half}.npz"


def score_path(root, game, half):
    return root / SCORES / game.path / f"{half}.csv"


def save_features(cache, info, volume, pitch, rms, wav):
    cache.parent.mkdir(parents=True, exist_ok=True)
    sig = wav.stat()
    temp = None
    try:
        with NamedTemporaryFile(dir=cache.parent, suffix=".part", delete=False) as handle:
            temp = Path(handle.name)
            np.savez_compressed(handle, frames=np.int64(info.frames), volume=volume,
                                pitch=pitch, rms=rms, wav=[sig.st_size, sig.st_mtime_ns])
        temp.replace(cache)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def reuse_expanded(game, root, sample_games):
    if game.path not in sample_games:
        return False
    for half in (1, 2):
        wav = ORIGINAL / "audio" / game.path / f"{half}.wav"
        info = sf.info(wav)
        previous = feature_path(ORIGINAL, game, half)
        score = score_path(ORIGINAL, game, half)
        read_scores(score, info.frames, 4)
        with np.load(previous, allow_pickle=False) as old:
            sig = wav.stat()
            if list(old["wav"]) != [sig.st_size, sig.st_mtime_ns]:
                return False
            values = [old[name] for name in ("volume", "pitch", "rms")]
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError(f"Cache ampliado inválido para {game.path}/{half}")
        save_features(feature_path(root, game, half), info, *values, wav)
        dest = score_path(root, game, half)
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy2(score, dest)
    return True


def cached_wavs(game):
    for source in (Path("data/audio"), ORIGINAL / "audio"):
        paths = [source / game.path / f"{half}.wav" for half in (1, 2)]
        if all(path.is_file() for path in paths):
            for half, wav in enumerate(paths, 1):
                try:
                    check_audio(wav, sf.info(wav).duration, game.annotations[half])
                except (MediaError, OSError, ValueError, RuntimeError):
                    break
            else:
                return paths
    return None


def prepare(game, root, media_revision, ranges, recovery=False):
    from adquirir_midias_checkpoint2_integral import acquire_game

    wavs = cached_wavs(game)
    created = wavs is None
    if created:
        options = {'recovery': True} if recovery else {}
        wavs = acquire_game(game.path, game.source, game.annotations, root, media_revision, ranges, **options)
    for half, wav in enumerate(wavs, 1):
        path = feature_path(root, game, half)
        status = wav.stat()
        if path.is_file():
            with np.load(path, allow_pickle=False) as old:
                if "frames" in old and list(old["wav"]) == [status.st_size, status.st_mtime_ns]:
                    continue
        info, volume, pitch, rms = classic_features(wav, path)
        save_features(path, info, volume, pitch, rms, wav)
    return wavs, created


def process_pool(args, games, ready, failures):
    from adquirir_midias_checkpoint2_integral import AccessError, resolve_revision

    root = args.raiz
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    former_profile = json.loads((ORIGINAL / "perfil_escores.json").read_text())
    reusable = (device == former_profile["dispositivo"] and
                str(dtype).split(".")[-1] == former_profile["precisao"] and
                MODEL == former_profile["modelo"] and REVISION == former_profile["revisao"] and
                torch.__version__ == former_profile["torch"])
    profile_path = root / "perfil.json"
    profile = {
        "device": device, "precision": str(dtype), "torch": torch.__version__,
        "model": MODEL, "model_revision": REVISION, "window_s": 4, "hop_s": 2,
    }
    if profile_path.is_file() and json.loads(profile_path.read_text()) != profile:
        raise ValueError("O perfil de pontuação mudou. Use outra pasta de execução.")
    checked_json(profile_path, profile)
    checked_json(args.saida / "perfil.json", profile)
    sample_games = set()
    if reusable:
        for manifest in ("amostra_ajuste.txt", "amostra_avaliacao.txt"):
            sample_games.update((ORIGINAL / manifest).read_text().splitlines())
    reused = 0
    for game in games:
        if game.path not in ready and reuse_expanded(game, root, sample_games):
            ready.add(game.path)
            failures.pop(game.path, None)
            reused += 1
    if reused:
        print(f"Reaproveitadas {reused} partidas da execução anterior de 4 s.", flush=True)
    record_status(root, args.saida, games, ready, failures, "executando")
    pending_games = sorted(
        (game for game in games if game.path not in ready),
        key=lambda game: game.path in failures,
    )
    if args.retentar_conexao:
        pending_games = [
            game for game in pending_games
            if any(
                error in failures.get(game.path, {}).get("motivo", "")
                for error in ("ConnectionError", "ReadTimeout", "TimeoutError")
            )
        ]
        print(f"Partidas separadas para nova tentativa de conexão: {len(pending_games)}", flush=True)
    if not pending_games:
        return
    if args.limite is not None:
        pending_games = pending_games[:args.limite]
    if not pending_games:
        return
    revision_path = root / "revisao_midia.txt"
    media_revision = revision_path.read_text().strip() if revision_path.is_file() else resolve_revision()
    if not re.fullmatch(r"[0-9a-f]{40}", media_revision):
        raise ValueError("Revisão de mídia inválida.")
    atomic_text(revision_path, media_revision + "\n")
    processor, model, index = load_model(args.modelo_local, device)
    model = ModelWithInputDtype(model.to(dtype=dtype), dtype)
    # Prepara alguns jogos em paralelo, sem acumular WAVs além da janela de trabalho.
    iterator = iter(pending_games)
    with ThreadPoolExecutor(max_workers=args.concorrencia) as pool:
        active = {}

        def refill():
            while len(active) < args.concorrencia and (game := next(iterator, None)) is not None:
                future = pool.submit(prepare, game, root, media_revision, args.faixas, args.recuperar_audio)
                active[future] = game

        refill()
        while active:
            complete, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in complete:
                game = active.pop(future)
                try:
                    wavs, created = future.result()
                    refill()
                    for half, wav in enumerate(wavs, 1):
                        dest = score_path(root, game, half)
                        score_audio(wav, dest, processor, model, index, args.lote, device, 0, None, window_s=4)
                        read_scores(dest, sf.info(wav).frames, 4)
                except AccessError:
                    raise
                except (MediaError, OSError, ValueError) as exc:
                    failures[game.path] = {"lote": game.subset, "motivo": str(exc)}
                    print(f"Falha registrada: {game.path} | {type(exc).__name__}", flush=True)
                else:
                    ready.add(game.path)
                    failures.pop(game.path, None)
                    record_status(root, args.saida, games, ready, failures, "executando")
                    if created:
                        for wav in wavs:
                            wav.unlink()
                    print(f"{len(ready)}/{len(games)} partidas concluídas | {game.subset} | {game.path}", flush=True)
                refill()
                record_status(root, args.saida, games, ready, failures, "executando")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, default=Path(".cache/checkpoint-2-integral"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/checkpoint-2-integral"))
    parser.add_argument("--anotacoes", type=Path, default=Path("data/soccernet/videos-224p"))
    parser.add_argument("--modelo-local", type=Path, default=Path(".cache/checkpoint-2/model"))
    parser.add_argument("--concorrencia", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--faixas", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--lote", type=int, default=24)
    parser.add_argument("--limite", type=int, help="Executa apenas essa quantidade de partidas pendentes.")
    parser.add_argument("--retentar-conexao", action="store_true",
                        help="Retenta apenas jogos pendentes com falha registrada de conexão.")
    parser.add_argument("--recuperar-audio", action="store_true",
                        help="Busca referências de sincronização ao longo dos tempos pendentes.")
    parser.add_argument("--somente-planejar", action="store_true")
    args = parser.parse_args()
    if args.lote < 1 or (args.limite is not None and args.limite < 1):
        parser.error("Informe valores positivos.")
    args.raiz.mkdir(parents=True, exist_ok=True)
    args.saida.mkdir(parents=True, exist_ok=True)
    with (args.raiz / "execucao.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Já há uma execução nesta pasta.")
        games = games_from_annotations(args.anotacoes)
        manifest = write_plan(games, args.raiz, args.saida)
        print(f"Partidas planejadas: {len(games)} | {dict(Counter(g.subset for g in games))}", flush=True)
        if args.somente_planejar:
            return
        status_path = args.raiz / "andamento.json"
        ready_path = args.raiz / "prontos.txt"
        ready = set(ready_path.read_text().splitlines()) if ready_path.is_file() else set()
        failures = json.loads(status_path.read_text())["falhas"] if status_path.is_file() else {}
        try:
            process_pool(args, games, ready, failures)
            if len({game.subset for game in games if game.path in ready}) == 2:
                subprocess.run(
                    [sys.executable, "avaliar_checkpoint2_integral.py", "--manifesto", str(manifest),
                     "--raiz", str(args.raiz), "--saida", str(args.saida), "--anotacoes", str(args.anotacoes)],
                    check=True,
                )
        except BaseException:
            record_status(args.raiz, args.saida, games, ready, failures, "interrompido")
            raise
        state = "completo" if len(ready) == len(games) else "parcial"
        record_status(args.raiz, args.saida, games, ready, failures, state)


if __name__ == "__main__":
    main()
