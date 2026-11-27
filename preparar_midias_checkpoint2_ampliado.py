"""Prepare a reproducible, annotated audio sample from authorized SoccerNet media."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from threading import Event, Lock
from time import sleep
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers
import requests
import soundfile as sf


REPO = "SoccerNet/SoccerNet_raw_HQ"
BRANCH = "videos-224p"
SEED = "pdsi-cp2-v2"
REPARO_DTS = "aresample=16000:async=1:first_pts=0"
SPLITS = {"train": ("ajuste", 2), "valid": ("avaliacao", 5)}
HEADER_SIZE = 128 * 1024
BLOCK_SIZE = 8 * 1024 * 1024
TIMEOUT = (12, 25)
CANDIDATE_FIELDS = ("divisao", "lote", "competicao", "ordem", "jogo", "sha256", "situacao", "motivo")
MEDIA_FIELDS = ("divisao", "lote", "competicao", "ordem", "jogo", "tempo",
                "wav", "bytes_mkv", "codec", "canais_originais", "hz_original",
                "duracao_mkv_s", "duracao_wav_s", "quadros_wav", "conversao", "revisao")
RANGE_HEADER = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


class MediaError(Exception):
    """A candidate file could not be validated."""


class AccessError(Exception):
    """Authentication failed at the source repository."""


def read_groups(data):
    old_games = set()
    for name in ("games_valid.txt", "games_valid_holdout.txt"):
        old_games.update(line.strip() for line in (data / name).read_text().splitlines() if line.strip())
    groups, annotations = {}, {}
    for split, (label, quota) in SPLITS.items():
        annotation_file = data / "soccernet" / BRANCH / f"{split}.json"
        games = {}
        for row in json.loads(annotation_file.read_text())["data"]:
            path, file = row["id"].rsplit("/", 1)
            if path not in old_games and file in ("1_224p.mkv", "2_224p.mkv"):
                half = int(file[0])
                games.setdefault(path, set()).add(half)
                annotations[path, half] = row["events"]
        competitions = sorted({path.split("/")[0] for path in games})
        if len(competitions) != 6:
            raise ValueError(f"{split}: esperadas seis competições")
        for competition in competitions:
            ordered = sorted(
                (sha256(f"{SEED}|{split}|{path}".encode()).hexdigest(), path)
                for path in games if path.split("/")[0] == competition
            )
            groups[split, competition] = [
                dict(divisao=split, lote=label, competicao=competition, ordem=rank, jogo=path,
                     sha256=digest, situacao="reserva", motivo="")
                for rank, (digest, path) in enumerate(ordered, start=1)
            ]
            if len(ordered) < quota:
                raise ValueError(f"{split}/{competition}: candidatos insuficientes")
    return groups, annotations


def atomic_text(path, content):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def csv_text(rows, fields):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


class Sample:
    def __init__(self, out, groups, revision):
        self.out = out
        self.groups = groups
        self.revision = revision
        self.selected = []
        self.media = []
        self.state = "executando"
        self.lock = Lock()
        self.reason = ""
        self.restore()
        self.save()

    def restore(self):
        audit_path = self.out / "amostra_candidatos.csv"
        media_path = self.out / "amostra_midias.csv"
        if not audit_path.exists() or not media_path.exists():
            return
        saved = self.read_rows(audit_path)
        saved_media = {}
        for row in self.read_rows(media_path):
            if row["revisao"] != self.revision:
                raise ValueError("Revisão incompatível com o manifesto de mídia existente.")
            saved_media.setdefault(row["jogo"], []).append(row)
        key = lambda row: (row["divisao"], row["competicao"], str(row["ordem"]), row["jogo"], row["sha256"])
        previous = {key(row): row for row in saved}
        current = {key(row): row for rows in self.groups.values() for row in rows}
        if previous.keys() != current.keys():
            raise ValueError("A ordem dos candidatos difere da execução existente.")
        state_path = self.out / "amostra_estado.json"
        old_state = json.loads(state_path.read_text()) if state_path.exists() else {}
        retry_ranks = {}
        if old_state.get("reparo_dts") != REPARO_DTS:
            for row in saved:
                group = row["divisao"], row["competicao"]
                if row["situacao"] == "rejeitado" and row["motivo"].startswith(
                    ("FFmpeg falhou", "Diferença de duração")
                ):
                    retry_ranks.setdefault(group, int(row["ordem"]))
        for index, row in current.items():
            status = previous[index]["situacao"]
            retry_rank = retry_ranks.get((row["divisao"], row["competicao"]))
            if retry_rank is not None and row["ordem"] >= retry_rank:
                continue
            if status == "rejeitado":
                row.update(situacao=status, motivo=previous[index]["motivo"])
            if status == "selecionado":
                halves = saved_media.get(row["jogo"], ())
                if (len(halves) == 2 and {half["tempo"] for half in halves} == {"1", "2"}
                        and all(self.restorable_wav(row["jogo"], half) for half in halves)):
                    row.update(situacao=status)
                    self.selected.append(row)
                    self.media.extend(halves)

    def restorable_wav(self, game, row):
        path = self.out / "audio" / game / f"{row['tempo']}.wav"
        try:
            info = sf.info(path)
        except (OSError, sf.LibsndfileError):
            return False
        return ((info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
                and info.frames == int(row["quadros_wav"]))

    @staticmethod
    def read_rows(path):
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def save(self):
        ordered = sorted(self.selected, key=lambda r: (r["competicao"], r["ordem"]))
        media = sorted(self.media, key=lambda r: (r["divisao"], r["competicao"], int(r["ordem"]), int(r["tempo"])))
        atomic_text(self.out / "amostra_midias.csv", csv_text(media, MEDIA_FIELDS))
        all_candidates = (row for rows in self.groups.values() for row in rows)
        atomic_text(self.out / "amostra_candidatos.csv", csv_text(all_candidates, CANDIDATE_FIELDS))
        for label, _ in SPLITS.values():
            games = [r["jogo"] for r in ordered if r["lote"] == label]
            atomic_text(self.out / f"amostra_{label}.txt", "".join(game + "\n" for game in games))
        counts = Counter((r["divisao"], r["competicao"]) for r in self.selected)
        state = dict(
            estado=self.state, motivo=self.reason, revisao=self.revision, semente=SEED,
            reparo_dts=REPARO_DTS,
            ajuste_completo=all(counts[split, league] == SPLITS[split][1]
                                for split, league in self.groups if split == "train"),
            cotas=[dict(divisao=split, competicao=league, obtidos=counts[split, league],
                        previstos=SPLITS[split][1]) for split, league in self.groups],
        )
        atomic_text(self.out / "amostra_estado.json", json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    def record(self, row, reason="", media=None):
        with self.lock:
            if media is None:
                row.update(situacao="rejeitado", motivo=reason)
                message = f"REJEITADO {row['lote']} / {row['competicao']} / posição {row['ordem']}: {reason}"
            else:
                row.update(situacao="selecionado", motivo="")
                self.selected.append(row)
                self.media.extend(media)
                message = f"PRONTO {row['lote']} / {len(self.selected)} jogos no total / {row['jogo']}"
            self.save()
            print(message, flush=True)


def response_range(response, start, end=None, total=None):
    if response.status_code in (401, 403) and urlparse(response.url).hostname == "huggingface.co":
        raise AccessError("O Hugging Face recusou a credencial para a mídia.")
    if response.status_code != 206:
        raise MediaError(f"Resposta HTTP {response.status_code} na mídia.")
    match = RANGE_HEADER.fullmatch(response.headers.get("Content-Range", ""))
    if not match:
        raise MediaError("Resposta sem Content-Range válido.")
    lo, hi, size = map(int, match.groups())
    if (lo != start or hi < lo or hi >= size
            or (end is not None and hi != min(end, size - 1))
            or (total is not None and size != total)):
        raise MediaError("Intervalo de bytes da mídia incompatível.")
    return hi + 1, size


def consume(session, url, start, end, total, file):
    """Write only bytes from a verified range, leaving the offset for retries."""
    with session.get(url, headers={"Range": f"bytes={start}-{end or ''}"},
                     stream=True, timeout=TIMEOUT) as response:
        next_offset, size = response_range(response, start, end, total)
        expected = min(end + 1, size) if end is not None else size
        if next_offset != expected:
            raise MediaError("Resposta não cobre o intervalo solicitado.")
        for chunk in response.iter_content(chunk_size=256 * 1024):
            if file.tell() + len(chunk) > next_offset:
                raise MediaError("A mídia excede o tamanho esperado.")
            file.write(chunk)
        if file.tell() != next_offset:
            raise MediaError("Recebimento incompleto da mídia.")
        return size


def transfer_header(session, url):
    for attempt in range(3):
        try:
            data = io.BytesIO()
            size = consume(session, url, 0, HEADER_SIZE - 1, None, data)
            return data.getvalue(), size
        except (requests.RequestException, MediaError) as exc:
            if attempt == 2:
                if isinstance(exc, MediaError):
                    raise
                raise MediaError(f"Falha no cabeçalho após tentativas ({type(exc).__name__}).") from None
            sleep(attempt + 1)


def download_remaining(session, url, size, header, media_file):
    with media_file.open("wb") as file:
        file.write(header)
        try:
            consume(session, url, file.tell(), None, size, file)
        except (requests.RequestException, MediaError):
            while file.tell() < size:
                end = min(file.tell() + BLOCK_SIZE, size) - 1
                for attempt in range(4):
                    if file.tell() > end:
                        break
                    try:
                        consume(session, url, file.tell(), end, size, file)
                        break
                    except (requests.RequestException, MediaError) as exc:
                        if attempt == 3:
                            if isinstance(exc, MediaError):
                                raise
                            raise MediaError(f"Transferência interrompida ({type(exc).__name__}).") from None
                        sleep(attempt + 1)
    if media_file.stat().st_size != size:
        raise MediaError("O tamanho final da mídia diverge do inventário.")


def transfer_block(url, size, start, end, fd, headers):
    for attempt in range(4):
        try:
            with requests.get(url, headers={**headers, "Range": f"bytes={start}-{end}"},
                              stream=True, timeout=TIMEOUT) as response:
                next_offset, _ = response_range(response, start, end, size)
                for chunk in response.iter_content(chunk_size=256 * 1024):
                    if start + len(chunk) > next_offset:
                        raise MediaError("A mídia excede o intervalo esperado.")
                    if os.pwrite(fd, chunk, start) != len(chunk):
                        raise MediaError("Falha de escrita de bloco.")
                    start += len(chunk)
                if start != next_offset:
                    raise MediaError("Transferência de bloco incompleta.")
            return
        except (requests.RequestException, MediaError) as exc:
            if start > end:
                return
            if attempt == 3:
                if isinstance(exc, MediaError):
                    raise
                raise MediaError(f"Transferência interrompida ({type(exc).__name__}).") from None
            sleep(attempt + 1)


def restore_partial(media_file, header, size, previous):
    """Adopt a sequential prefix only after checking the authenticated header."""
    if not previous.exists():
        return
    for candidate in previous.rglob(media_file.name):
        try:
            if not len(header) < candidate.stat().st_size <= size:
                continue
            with candidate.open("rb") as file:
                if file.read(len(header)) != header:
                    continue
            candidate.replace(media_file)
            return
        except FileNotFoundError:
            continue


def download_concurrent(url, size, header, media_file, headers, blocks):
    start = media_file.stat().st_size if media_file.exists() else len(header)
    offsets = range(start, size, BLOCK_SIZE)
    mode = "r+b" if media_file.exists() else "wb"
    with media_file.open(mode) as file:
        file.truncate(size)
        if os.pwrite(file.fileno(), header, 0) != len(header):
            raise MediaError("Falha de escrita do cabeçalho.")
        with ThreadPoolExecutor(blocks) as pool:
            jobs = [
                pool.submit(transfer_block, url, size, offset, min(offset + BLOCK_SIZE, size) - 1,
                            file.fileno(), headers)
                for offset in offsets
            ]
            for job in jobs:
                job.result()


def inspect_prefix(header):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels:format=duration", "-of", "json", "pipe:0"],
        input=header, capture_output=True, timeout=10,
    )
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        duration = float(data["format"]["duration"])
        if not 600 < duration < 5000:
            raise ValueError
        return stream, duration
    except (ValueError, KeyError, IndexError):
        raise MediaError("O cabeçalho não confirmou uma faixa de áudio completa.") from None


def check_audio(wav, duration, events):
    try:
        info = sf.info(wav)
        if (info.samplerate, info.channels, info.subtype) != (16000, 1, "PCM_16"):
            raise MediaError("Esperado WAV mono, PCM16, 16 kHz.")
        if abs(info.duration - duration) > 3:
            raise MediaError("Diferença de duração maior que 3 s entre mídia e áudio.")
        if any(not 0 <= event["position_ms"] <= 1000 * (info.duration + 0.25) for event in events):
            raise MediaError("Há anotação fora da duração do áudio.")
        if not any(block.any() for block in sf.blocks(wav, blocksize=160000, dtype="int16")):
            raise MediaError("Faixa completamente silenciosa.")
        return info
    except sf.LibsndfileError:
        raise MediaError("WAV inválido.") from None


def convert_audio(mkv, wav, duration, events):
    reasons = []
    for method, params in (("pcm16", []), ("pcm16_reparo_dts", ["-af", REPARO_DTS])):
        run = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-xerror", "-i", str(mkv), "-map", "0:a:0",
             *params, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
            capture_output=True, timeout=120,
        )
        if run.returncode:
            reasons.append(f"{method}: FFmpeg status {run.returncode}")
            continue
        try:
            check_audio(wav, duration, events)
        except MediaError as exc:
            reasons.append(f"{method}: {exc}")
        else:
            return method
    raise MediaError("; ".join(reasons))


def prepare_game(row, annotations, sample, work_dir, blocks):
    game = row["jogo"]
    with requests.Session() as session, tempfile.TemporaryDirectory(prefix="midia-", dir=work_dir) as tmp:
        session.headers.update(build_hf_headers(token=True))
        session.headers["Accept-Encoding"] = "identity"
        temp_dir = Path(tmp)
        halves = []
        for half in (1, 2):
            if (game, half) not in annotations:
                raise MediaError(f"Sem anotações do {half}º tempo.")
            url = hf_hub_url(REPO, f"{game}/{half}_224p.mkv", repo_type="dataset", revision=sample.revision)
            header, size = transfer_header(session, url)
            details, duration = inspect_prefix(header)
            halves.append((half, url, header, size, details, duration))
        media = []
        for half, url, header, size, details, duration in halves:
            wav = sample.out / "audio" / game / f"{half}.wav"
            events = annotations[game, half]
            conversion = "reaproveitado"
            if not wav.exists():
                mkv = temp_dir / f"{half}.mkv"
                raw_wav = temp_dir / f"{half}.wav"
                if blocks > 1:
                    restore_partial(mkv, header, size, work_dir.parent / "stream-antigo")
                    download_concurrent(url, size, header, mkv, dict(session.headers), blocks)
                else:
                    download_remaining(session, url, size, header, mkv)
                conversion = convert_audio(mkv, raw_wav, duration, events)
                wav.parent.mkdir(parents=True, exist_ok=True)
                raw_wav.replace(wav)
                mkv.unlink()
                print(f"WAV {row['lote']} / {half}º tempo / {game} ({conversion})", flush=True)
            info = check_audio(wav, duration, events)
            media.append({
                **{k: row[k] for k in ("divisao", "lote", "competicao", "ordem", "jogo")},
                "tempo": half, "wav": str(wav), "bytes_mkv": size, "codec": details["codec_name"],
                "canais_originais": details["channels"], "hz_original": details["sample_rate"],
                "duracao_mkv_s": f"{duration:.3f}", "duracao_wav_s": f"{info.duration:.3f}",
                "quadros_wav": info.frames, "conversao": conversion, "revisao": sample.revision,
            })
        return media


def prepare_group(rows, annotations, sample, work_dir, done, blocks):
    selected = sum(row["situacao"] == "selecionado" for row in rows)
    remaining = SPLITS[rows[0]["divisao"]][1] - selected
    for row in rows:
        if remaining == 0 or done.is_set():
            break
        if row["situacao"] != "reserva":
            continue
        try:
            media = prepare_game(row, annotations, sample, work_dir, blocks)
        except (MediaError, subprocess.TimeoutExpired) as exc:
            sample.record(row, str(exc) if isinstance(exc, MediaError) else "Tempo limite de conversão.")
        except Exception:
            done.set()
            raise
        else:
            sample.record(row, media=media)
            remaining -= 1
    return remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path, default=Path(".cache/checkpoint-2-ampliado"))
    parser.add_argument("--dados", type=Path, default=Path("data"))
    parser.add_argument("--concorrencia", type=int, default=4)
    parser.add_argument("--faixas", type=int, default=8,
                        help="Número de faixas HTTP concorrentes por tempo.")
    args = parser.parse_args()
    if not 1 <= args.faixas <= 16:
        parser.error("--faixas deve ficar entre 1 e 16")
    if not 1 <= args.concorrencia <= 8:
        parser.error("--concorrencia deve ficar entre 1 e 8")
    groups, annotations = read_groups(args.dados)
    args.saida.mkdir(parents=True, exist_ok=True)
    with (args.saida / "amostra_execucao.lock").open("a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Já existe uma preparação de mídia no diretório.") from None
        return run(args, groups, annotations)


def run(args, groups, annotations):
    temp = args.saida / "temporarios"
    temp.mkdir(exist_ok=True)
    state_file = args.saida / "amostra_estado.json"
    if state_file.exists():
        revision = json.loads(state_file.read_text())["revisao"]
    else:
        revision = HfApi().dataset_info(REPO, revision=BRANCH, token=True, timeout=20).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("Não foi possível fixar a revisão do acervo.")
    sample = Sample(args.saida, groups, revision)
    done = Event()
    try:
        with ThreadPoolExecutor(args.concorrencia) as workers:
            jobs = [workers.submit(prepare_group, group, annotations, sample, temp, done, args.faixas) for group in groups.values()]
            missing = sum(job.result() for job in jobs)
        sample.state = "incompleto" if missing else "completo"
        if missing:
            sample.reason = f"Faltam {missing} partidas nas cotas."
    except AccessError as exc:
        sample.state, sample.reason = "falha", str(exc)
    except BaseException as exc:
        sample.state, sample.reason = "falha", f"Execução interrompida: {type(exc).__name__}."
    finally:
        sample.save()
    print(f"Amostra {sample.state}: {len(sample.selected)} jogos, {len(sample.media)} tempos. {sample.reason}", flush=True)
    return int(sample.state != "completo")


if __name__ == "__main__":
    raise SystemExit(main())
