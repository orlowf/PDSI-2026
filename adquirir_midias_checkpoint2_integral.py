"""Acquire and verify two SoccerNet audio halves in a dedicated run directory."""

from configparser import ConfigParser, Error as ConfigError
from hashlib import file_digest
import json
from math import isfinite
from pathlib import Path
import re
from shutil import disk_usage
import subprocess
from tempfile import TemporaryDirectory
from typing import NamedTuple

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError
from huggingface_hub.utils import build_hf_headers
import numpy as np
import requests
import soundfile as sf

from preparar_midias_checkpoint2_ampliado import (
    AccessError,
    BRANCH,
    MediaError,
    REPO,
    TIMEOUT,
    atomic_text,
    check_audio,
    convert_audio,
    download_concurrent,
    download_remaining,
    inspect_prefix,
    transfer_header,
)


SYNCHRONIZED_FALLBACK_REVISION = "280e9828acb8f99250eefdeb4c0eae9a1f784739"
HQ_REVISION = "462c8ed7da5784d606c6173efe93aa90cc15fea1"
SPLITS = frozenset({"train", "valid", "test"})
SAMPLE_RATE = 16000
REFERENCE_BYTES = 16 * 1024 * 1024
REFERENCE_INTERVAL_SECONDS = 10
FRAMES = re.compile(rb"pts:(-?\d+)\s+pts_time:[^\s]+[^\n]*?nb_samples:(\d+)")
DECODER_WARNINGS = re.compile(rb"warning|error|invalid|corrupt|overread|backstep", re.I)


class MediaProbe(NamedTuple):
    url: str
    header: bytes
    size: int
    audio: dict
    duration: float


class AudioConversionError(MediaError):
    """The transferred video did not yield synchronized audio."""


def resolve_revision() -> str:
    """Resolve the authorized branch without returning credentials or signed URLs."""
    try:
        revision = HfApi().dataset_info(REPO, revision=BRANCH, token=True, timeout=20).sha
    except LocalTokenNotFoundError:
        raise AccessError("A credencial local do Hugging Face não está disponível.") from None
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code in (401, 403):
            raise AccessError("O Hugging Face recusou a credencial para a mídia.") from None
        raise MediaError("Não foi possível consultar a revisão do acervo.") from None
    except requests.RequestException:
        raise MediaError("Falha de rede ao consultar a revisão do acervo.") from None
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise MediaError("Não foi possível fixar a revisão do acervo.")
    return revision


def _probe(session, game, half, quality, revision):
    url = hf_hub_url(REPO, f"{game}/{half}_{quality}.mkv", repo_type="dataset", revision=revision)
    header, size = transfer_header(session, url)
    try:
        details, duration = inspect_prefix(header)
    except subprocess.TimeoutExpired:
        raise MediaError(f"{half}º tempo: tempo limite ao ler o cabeçalho da mídia.") from None
    return MediaProbe(url, header, size, details, duration)


def _download(session, probe, media, ranges):
    if ranges > 1:
        download_concurrent(probe.url, probe.size, probe.header, media, dict(session.headers), ranges)
    else:
        download_remaining(session, probe.url, probe.size, probe.header, media)


def _decode(session, probe, directory, events, ranges, reference_duration=None):
    directory.mkdir()
    media = directory / "media.mkv"
    wav = directory / "audio.wav"
    _download(session, probe, media, ranges)
    try:
        method = convert_audio(media, wav, probe.duration, events)
        if reference_duration is not None:
            check_audio(wav, reference_duration, events)
        return wav, method
    except MediaError as exc:
        raise AudioConversionError(str(exc)) from None


def _decoded_frames(stderr):
    return [tuple(map(int, row)) for row in FRAMES.findall(stderr)]


def _reference_clips(media):
    """Retain short, time-indexed pieces of the original 224p audio for comparison."""
    with media.open("rb") as file:
        prefix = file.read(REFERENCE_BYTES)
    result = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", "pipe:0", "-map", "0:a:0",
         "-af", f"aresample={SAMPLE_RATE},aformat=sample_fmts=flt:channel_layouts=mono,ashowinfo",
         "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"],
        input=prefix, capture_output=True, timeout=30,
    )
    clips = _clips_from_pcm(result)
    if not _has_separated_references(clips):
        raise AudioConversionError("referência 224p insuficiente para comprovar o relógio HQ.")
    return _spread(clips, 15)


def _clips_from_pcm(result, clock=0.):
    samples = np.frombuffer(result.stdout, dtype="<f4")
    clips, group = [], []
    cursor = first = 0
    last = None

    def retain_group():
        if not group:
            return
        span = np.concatenate(group)
        length = min(len(span), SAMPLE_RATE // 5)
        if length >= 250:
            for index in range(0, len(span) - length + 1, 12 * SAMPLE_RATE):
                signal = span[index:index+length]
                if signal.std() >= .001:
                    clips.append(((first + index) / SAMPLE_RATE - clock, signal.copy()))

    for stamp, length in _decoded_frames(result.stderr):
        if cursor + length > len(samples):
            break
        if last is not None and abs(stamp - last) > .007 * SAMPLE_RATE:
            retain_group()
            group.clear()
        if not group:
            first = stamp
        group.append(samples[cursor:cursor+length])
        cursor += length
        last = stamp + length
    retain_group()
    return sorted(clips, key=lambda item: item[0])


def _has_separated_references(clips):
    return len(clips) >= 2 and clips[-1][0] - clips[0][0] >= 5


def _spread(clips, limit):
    selected = np.linspace(0, len(clips) - 1, min(len(clips), limit), dtype=int)
    return [clips[index] for index in selected]


def _container_clock(media):
    metadata = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of",
         "default=noprint_wrappers=1:nokey=1", str(media)],
        capture_output=True, text=True, check=True, timeout=35,
    )
    return float(metadata.stdout.strip())


def _packets(media, track=0):
    trace = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", f"a:{track}", "-show_entries",
         "packet=pts_time,duration_time", "-of", "csv=p=0", str(media)],
        capture_output=True, text=True, check=True, timeout=180,
    )
    for line in trace.stdout.splitlines():
        values = line.split(",")[:2]
        if len(values) == 2 and all(value not in ("", "N/A") for value in values):
            start, length = map(float, values)
            if isfinite(start) and isfinite(length) and length > 0:
                yield start, length


def _full_clock_references(media, duration):
    """Decode the complete clock and retain bounded PCM excerpts where packets exist."""
    clock = _container_clock(media)
    if not isfinite(clock):
        raise ValueError("Relógio do contêiner inválido.")
    starts = {}
    for stamp, _ in _packets(media):
        timestamp = stamp - clock
        if 0 <= timestamp < duration:
            bucket = int(timestamp // REFERENCE_INTERVAL_SECONDS)
            starts[bucket] = min(starts.get(bucket, stamp), stamp)
    if not starts:
        return []

    excerpts = "+".join(f"between(t,{start - .04:.6f},{start + .48:.6f})" for start in starts.values())
    filtergraph = (f"aresample={SAMPLE_RATE},aformat=sample_fmts=flt:channel_layouts=mono,"
                   f"aselect='{excerpts}',ashowinfo")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-copyts", "-i", str(media), "-map", "0:a:0",
         "-af", filtergraph, "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"],
        capture_output=True, timeout=180,
    )
    return [clip for clip in _clips_from_pcm(result, clock) if 0 <= clip[0] < duration]


def _expanded_references(primary_media, fallback_media, duration):
    references = []
    for quality, media in (("224p", primary_media), ("720p", fallback_media)):
        if media is None or not media.is_file():
            continue
        try:
            clips = _full_clock_references(media, duration)
        except (OSError, ValueError, subprocess.SubprocessError):
            clips = []
        if quality == "224p" and not _has_separated_references(clips):
            try:
                clips = _reference_clips(media)
            except (OSError, ValueError, MediaError, subprocess.SubprocessError):
                pass
        references.extend((stamp, signal, quality) for stamp, signal in _spread(clips, 30))
    references.sort(key=lambda item: item[0])
    if not _has_separated_references(references):
        raise AudioConversionError("referências 224p/720p insuficientes para comprovar o relógio HQ.")
    return references


def _confirm_alignment(wav, references, *, recovery=False):
    """Require independent waveform matches at the same source clock."""
    from scipy.signal import correlate

    matches = []
    with sf.SoundFile(wav) as audio:
        for timestamp, reference, *source in references:
            origin = max(0, round((timestamp - .6) * SAMPLE_RATE))
            end = min(audio.frames, round((timestamp + .6) * SAMPLE_RATE) + len(reference))
            if end - origin < len(reference):
                continue
            audio.seek(origin)
            window = audio.read(end - origin, dtype="float64")
            ref = reference.astype(float) - reference.mean()
            n = len(ref)
            sum_x = np.r_[0., np.cumsum(window)]
            sum_x2 = np.r_[0., np.cumsum(window**2)]
            energy = (sum_x2[n:] - sum_x2[:-n]) - (sum_x[n:] - sum_x[:-n])**2/n
            corr = correlate(window, ref, mode="valid", method="fft")
            corr /= np.linalg.norm(ref) * np.sqrt(np.maximum(energy, 1e-12))
            if recovery:
                offsets = (origin + np.arange(len(corr))) / SAMPLE_RATE - timestamp
                eligible = np.flatnonzero(np.abs(offsets) <= .35)
                if not len(eligible):
                    continue
                peak = int(eligible[np.argmax(corr[eligible])])
            else:
                peak = int(np.argmax(corr))
            adjustment = (origin + peak) / SAMPLE_RATE - timestamp
            if corr[peak] >= .60 and abs(adjustment) <= .35:
                match = dict(reference_seconds=timestamp, offset_seconds=adjustment,
                             correlation=float(corr[peak]), frames=n)
                if source:
                    match["reference_source"] = source[0]
                matches.append(match)
    for first_index, first in enumerate(matches):
        for second in matches[first_index+1:]:
            if (abs(second["reference_seconds"] - first["reference_seconds"]) >= 5
                    and abs(second["offset_seconds"] - first["offset_seconds"]) <= .04):
                return matches
    source_label = "224p/720p" if recovery else "224p"
    raise AudioConversionError(f"cronologia HQ sem duas correspondências independentes com o {source_label}.")


def _hq_offset(session, game, half):
    url = hf_hub_url(REPO, f"{game}/video.ini", repo_type="dataset", revision=HQ_REVISION)
    with session.get(url, stream=True, timeout=TIMEOUT) as response:
        if response.status_code in (401, 403):
            raise AccessError("O Hugging Face recusou a credencial para o video.ini.")
        if not response.ok:
            raise AudioConversionError(f"video.ini indisponível (HTTP {response.status_code}).")
        contents = bytearray()
        for chunk in response.iter_content(chunk_size=2048):
            contents.extend(chunk)
            if len(contents) > 16384:
                raise AudioConversionError("video.ini excede o tamanho de metadados aceito.")
    info = ConfigParser()
    info.read_string(contents.decode("utf-8-sig"))
    offset = info[f"{half}_HQ.mkv"].getfloat("start_time_second")
    if not isfinite(offset) or offset < 0:
        raise AudioConversionError("início anotado inválido no video.ini.")
    return offset


def _verify_hq_packets(media, offset, duration, track=0):
    """Confirm that the selected audio packets cover the configured crop."""
    origin = _container_clock(media) + offset
    end = origin + duration
    intervals = [(max(origin, start), min(end, start + length)) for start, length in _packets(media, track)
                 if start < end and start + length > origin]
    frontier, gap, coverage = origin, 0., 0.
    for start, stop in sorted(intervals):
        gap = max(gap, start-frontier)
        coverage += max(0., stop-max(start,frontier))
        frontier = max(frontier,stop)
    gap = max(gap,end-frontier)
    if not intervals or coverage < duration - .5 or gap > .15:
        raise AudioConversionError("lacunas nos pacotes HQ dentro do tempo anotado.")
    return dict(packets=len(intervals), coverage_seconds=coverage, largest_gap_seconds=gap,
                source_start_seconds=origin, source_end_seconds=end)


def _render_hq(media, wav, offset, duration, track=0):
    # Decode from the beginning: input seek could strand the MP3 bit reservoir.
    filtergraph = (f"atrim=start={offset:.6f}:end={offset+duration:.6f},aresample={SAMPLE_RATE},"
                   "aformat=sample_fmts=s16:channel_layouts=mono,ashowinfo,asetpts=PTS-STARTPTS")
    run = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-y", "-i", str(media),
         "-map", f"0:a:{track}", "-af", filtergraph, "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "pcm_s16le", str(wav)],
        capture_output=True, timeout=180,
    )
    run.check_returncode()
    decoded = _decoded_frames(run.stderr)
    warnings = [line for line in run.stderr.splitlines()
                if b"ashowinfo" not in line and DECODER_WARNINGS.search(line)]
    if not decoded or warnings:
        raise AudioConversionError(f"decodificação HQ não verificável ({len(warnings)} avisos/erros).")
    expected = round(duration*SAMPLE_RATE)
    first = decoded[0][0]
    last = sum(decoded[-1])
    frames = sum(length for _, length in decoded)
    gaps = [after-before-length for (before,length),(after,_) in zip(decoded,decoded[1:])]
    discontinuity = max(map(abs,gaps),default=0)
    if (abs(frames-expected) > 1 or abs(first-round(offset*SAMPLE_RATE)) > .03*SAMPLE_RATE
            or abs(last-round((offset+duration)*SAMPLE_RATE)) > .03*SAMPLE_RATE
            or discontinuity > .05*SAMPLE_RATE or sf.info(wav).frames != frames):
        raise AudioConversionError("WAV HQ não preservou o número de amostras e o relógio.")
    rms_threshold = .0001
    block_rms = [np.sqrt(np.mean(np.square(block, dtype=float)))
                 for block in sf.blocks(wav, blocksize=SAMPLE_RATE, dtype="float32")]
    silent = sum(1 for rms in block_rms if rms < rms_threshold)
    if silent > len(block_rms)*.01:
        raise AudioConversionError("faixa HQ apresenta silêncio excessivo após conversão.")
    return dict(decoded_frames=frames, largest_discontinuity_samples=discontinuity,
                blocks_below_minus_80_dbfs=silent, total_one_second_blocks=len(block_rms))


def _verify_hq_track(media, directory, offset, duration, track, references, digest, *, recovery=False):
    packets = _verify_hq_packets(media, offset, duration, track)
    wav = directory / "audio.wav"
    decoded = _render_hq(media, wav, offset, duration, track)
    matches = _confirm_alignment(wav, references, recovery=recovery)
    evidence = dict(offset_seconds=offset, track=f"0:a:{track}", sha256=digest, packets=packets,
                    pcm=decoded, waveform_matches=matches)
    return wav, evidence


def _verified_hq_candidate(media, directory, source, offset, duration, references, digest, recovery):
    if not recovery:
        wav, evidence = _verify_hq_track(media, directory, offset, duration, 0, references, digest)
        return wav, source, evidence

    track_info = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_name,channels", "-of", "json", str(media)],
        capture_output=True, text=True, check=True, timeout=35,
    )
    tracks = json.loads(track_info.stdout)["streams"]
    failures = []
    for index, track in enumerate(tracks[:2]):
        if track.get("channels") not in (1, 2) or not track.get("codec_name"):
            continue
        try:
            wav, evidence = _verify_hq_track(
                media, directory, offset, duration, index, references, digest, recovery=True,
            )
            return wav, source._replace(audio=track), evidence
        except AudioConversionError as exc:
            failures.append(f"faixa {index + 1}: {exc}")
        except (OSError, ValueError, subprocess.SubprocessError, sf.LibsndfileError) as exc:
            failures.append(f"faixa {index + 1}: validação interrompida ({type(exc).__name__}).")
    raise AudioConversionError(" ".join(failures) or "nenhuma faixa HQ mono/estéreo foi identificada.")


def _recover_hq(session, game, half, reference_media, reference_duration, directory, ranges,
                *, recovery=False, fallback_media=None):
    try:
        offset = _hq_offset(session,game,half)
        references = (_expanded_references(reference_media, fallback_media, reference_duration)
                      if recovery else _reference_clips(reference_media))
        source = _probe(session,game,half,"HQ",HQ_REVISION)
        if (not recovery and source.audio["channels"] != 2) or offset + reference_duration > source.duration:
            raise AudioConversionError("faixa estéreo ou intervalo do video.ini incompatível.")
        directory.mkdir()
        if source.size + round(2*reference_duration*SAMPLE_RATE) > disk_usage(directory).free:
            raise AudioConversionError("espaço insuficiente para este arquivo HQ.")
        files = HfApi().get_paths_info(REPO,[f"{game}/{half}_HQ.mkv"],repo_type="dataset",
                                      revision=HQ_REVISION,token=True)
        file, = files
        if file.lfs is None or not file.lfs.sha256 or source.size != file.size:
            raise AudioConversionError("identidade e tamanho da mídia HQ não confirmados pela revisão.")
        media = directory / "media.mkv"
        _download(session,source,media,min(ranges,4))
        with media.open("rb") as stream:
            digest = file_digest(stream,"sha256").hexdigest()
        if digest != file.lfs.sha256:
            raise AudioConversionError("arquivo HQ não confere com o hash da revisão.")
        return _verified_hq_candidate(
            media, directory, source, offset, reference_duration, references, digest, recovery,
        )
    except AccessError:
        raise
    except MediaError as exc:
        raise AudioConversionError(f"HQ, {half}º tempo: {exc} Revisão manual necessária.") from None
    except (ConfigError, KeyError, TypeError, ValueError, OSError, requests.RequestException,
            subprocess.SubprocessError) as exc:
        # Exception strings from HF/HTTP can contain protected URLs.
        raise AudioConversionError(
            f"HQ, {half}º tempo: não foi possível validar a recuperação ({type(exc).__name__}). "
            "Revisão manual necessária."
        ) from None


def _existing_audio(wav, record, game, half, source_split, revision, events):
    if not record.is_file() or not wav.is_file():
        return False
    try:
        metadata = json.loads(record.read_text(encoding="utf-8"))
        identity = metadata["game"], metadata["half"], metadata["split"], metadata["primary_revision"]
        if identity != (game, half, source_split, revision):
            return False
        info = check_audio(wav, float(metadata["duration_s"]), events)
        return info.frames == metadata["frames"]
    except (KeyError, TypeError, ValueError, OSError, MediaError):
        return False


def acquire_game(game: str, source_split: str, events_by_half: dict[int, list],
                 job_root: Path, revision: str, ranges: int, *, recovery: bool = False) -> list[Path]:
    """Return one 16 kHz mono WAV per half, from the pinned source or a validated fallback."""
    relative = Path(game)
    if relative.is_absolute() or len(relative.parts) != 3 or ".." in relative.parts:
        raise ValueError("Identificador de partida inválido.")
    if source_split not in SPLITS or events_by_half.keys() != {1, 2}:
        raise ValueError("São necessárias anotações dos dois tempos de uma divisão válida.")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not 1 <= ranges <= 16:
        raise ValueError("Revisão ou número de faixas HTTP inválidos.")
    try:
        headers = build_hf_headers(token=True)
    except LocalTokenNotFoundError:
        raise AccessError("A credencial local do Hugging Face não está disponível.") from None

    audio_dir = Path(job_root) / "audio" / relative
    work_dir = Path(job_root) / "transferencias"
    audio_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with requests.Session() as session:
        session.headers.update(headers)
        session.headers["Accept-Encoding"] = "identity"
        for half, events in sorted(events_by_half.items()):
            wav = audio_dir / f"{half}.wav"
            record = audio_dir / f"{half}.json"
            if not _existing_audio(wav, record, game, half, source_split, revision, events):
                primary = _probe(session, game, half, "224p", revision)
                probe, source_revision, quality = primary, revision, "224p"
                evidence = None
                with TemporaryDirectory(prefix="soccer-", dir=work_dir) as temporary:
                    directory = Path(temporary)
                    try:
                        raw_wav, method = _decode(session, primary, directory / "224p", events, ranges)
                    except subprocess.TimeoutExpired:
                        raise MediaError(f"{half}º tempo: tempo limite de conversão.") from None
                    except AudioConversionError:
                        fallback = _probe(session, game, half, "720p", SYNCHRONIZED_FALLBACK_REVISION)
                        probe, source_revision, quality = fallback, SYNCHRONIZED_FALLBACK_REVISION, "720p"
                        try:
                            if abs(fallback.duration - primary.duration) > 3:
                                raise AudioConversionError("Duração 720p diverge da referência 224p.")
                            raw_wav, method = _decode(
                                session, fallback, directory / "720p", events, ranges, primary.duration,
                            )
                        except subprocess.TimeoutExpired:
                            raise MediaError(f"{half}º tempo: tempo limite de conversão em 720p.") from None
                        except AudioConversionError:
                            raw_wav, probe, evidence = _recover_hq(
                                session, game, half, directory / "224p" / "media.mkv",
                                primary.duration, directory / "HQ", ranges, recovery=recovery,
                                fallback_media=(directory / "720p" / "media.mkv" if recovery else None),
                            )
                            source_revision, quality, method = HQ_REVISION, "HQ", "pcm16_atrim_validado"

                    info = check_audio(raw_wav, primary.duration, events)
                    metadata = dict(
                        game=game, half=half, split=source_split, quality=quality,
                        primary_revision=revision, source_revision=source_revision,
                        media_bytes=probe.size, audio_codec=probe.audio["codec_name"],
                        duration_s=primary.duration, source_duration_s=probe.duration,
                        frames=info.frames, conversion=method,
                    )
                    if evidence is not None:
                        metadata["hq_validation"] = evidence
                    serialized = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
                    raw_wav.replace(wav)
                    atomic_text(record, serialized)
            paths.append(wav)
    return paths
