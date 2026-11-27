"""Score SoccerNet audio windows using the audEERING wav2vec 2.0 model."""

import argparse
import csv
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
REVISION = "6eba34a2485ea31cb03600241787c3a5edab8626"
SAMPLE_RATE = 16_000
WINDOW_S = 4
HOP_S = 2
FIELDS = ("inicio_s", "centro_s", "fim_s", "arousal")


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Regression head from the audEERING model card."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = torch.nn.ModuleDict({
            "dense": torch.nn.Linear(config.hidden_size, config.hidden_size),
            "out_proj": torch.nn.Linear(config.hidden_size, config.num_labels),
        })
        self.dropout = torch.nn.Dropout(config.final_dropout)
        self.post_init()

    def forward(self, input_values):
        features = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
        hidden = torch.tanh(self.classifier["dense"](self.dropout(features)))
        return self.classifier["out_proj"](self.dropout(hidden))


def load_model(local_path, device):
    source = local_path or MODEL
    options = {"revision": REVISION} if local_path is None else {"local_files_only": True}
    processor = Wav2Vec2FeatureExtractor.from_pretrained(source, **options)
    model, loading = EmotionModel.from_pretrained(
        source, output_loading_info=True, use_safetensors=True, **options
    )
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(field):
            raise RuntimeError(f"Incompatibilidade nos pesos ({field}): {loading[field]}")
    index = next(k for k, label in model.config.id2label.items() if label.lower() == "arousal")
    return processor, model.to(device).eval(), index


def score_audio(wav, dest, processor, model, index, batch_size, device, start_s, duration_s, *, window_s=WINDOW_S):
    info = sf.info(wav)
    if (info.samplerate, info.channels) != (SAMPLE_RATE, 1):
        raise ValueError(f"Esperado WAV mono a 16 kHz: {wav}")
    offset = start_s * SAMPLE_RATE
    end = info.frames if duration_s is None else min(info.frames, offset + duration_s * SAMPLE_RATE)
    window, hop = window_s * SAMPLE_RATE, HOP_S * SAMPLE_RATE
    starts = range(0, end - offset - window + 1, hop)
    if not starts:
        raise ValueError(f"Áudio insuficiente para uma janela completa: {wav}")
    if dest.is_file():
        with dest.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) == len(starts) and all(
            tuple(float(row[field]) for field in FIELDS[:3])
            == (t, t + window_s / 2, t + window_s)
            and np.isfinite(float(row["arousal"]))
            for row, t in zip(rows, (start_s + s / SAMPLE_RATE for s in starts))
        ):
            print(f"CSV existente conferido: {dest} ({len(rows)} janelas)", flush=True)
            return
        raise ValueError(f"CSV incompatível com o trecho solicitado: {dest}")

    signal, _ = sf.read(wav, start=offset, frames=end - offset, dtype="float32")
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", dir=dest.parent, suffix=".part", delete=False
        ) as file, torch.inference_mode():
            temporary = Path(file.name)
            writer = csv.writer(file)
            writer.writerow(FIELDS)
            for batch_start in range(0, len(starts), batch_size):
                offsets = starts[batch_start:batch_start + batch_size]
                batch = [signal[s:s + window] for s in offsets]
                inputs = processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                scores = model(inputs.input_values.to(device))[:, index].cpu().numpy()
                if not np.isfinite(scores).all():
                    raise ValueError(f"O modelo retornou valores não finitos: {wav}")
                for s, score in zip(offsets, scores):
                    t = start_s + s / SAMPLE_RATE
                    writer.writerow((t, t + window_s / 2, t + window_s, float(score)))
                processed = batch_start + len(offsets)
                if processed % (batch_size * 40) == 0:
                    print(f"{wav.parent.name} | {wav.stem}º tempo | {processed}/{len(starts)} janelas", flush=True)
        temporary.replace(dest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    elapsed = perf_counter() - start
    print(f"{dest}: {len(starts)} janelas em {elapsed:.2f} s ({len(starts) / elapsed:.2f} janelas/s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--lista", type=Path)
    selection.add_argument("--jogo")
    parser.add_argument("--tempo", type=int, choices=(1, 2), default=1)
    parser.add_argument("--inicio", type=int, default=0)
    parser.add_argument("--duracao", type=int)
    parser.add_argument("--janela", type=int, choices=(2, WINDOW_S), default=WINDOW_S, help="Duração da janela em segundos")
    parser.add_argument("--dados", type=Path, default=Path("data"))
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--modelo-local", type=Path)
    parser.add_argument("--dispositivo", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--lote", type=int, default=8)
    args = parser.parse_args()
    if args.lote < 1 or args.inicio < 0 or (args.duracao is not None and args.duracao < args.janela):
        parser.error(f"Informe lote positivo, início não negativo e duração mínima de {args.janela} s.")
    if args.lista:
        if args.inicio or args.duracao is not None:
            parser.error("Início e duração são opções de um jogo por vez.")
        games = [line.strip() for line in args.lista.read_text().splitlines() if line.strip()]
        if not games or len(games) != len(set(games)):
            parser.error("A lista deve conter jogos distintos.")
        tasks = [(game, half) for game in games for half in (1, 2)]
    else:
        tasks = [(args.jogo, args.tempo)]
    device = args.dispositivo
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(6, torch.get_num_threads()))

    print(f"{MODEL}@{REVISION} | torch {torch.__version__} | {device} | lote {args.lote} | janela {args.janela} s", flush=True)
    processor, model, index = load_model(args.modelo_local, device)
    for game, half in tasks:
        wav = args.dados / "audio" / game / f"{half}.wav"
        filename = f"{half}.csv"
        if args.inicio or args.duracao is not None:
            filename = f"{half}_desde-{args.inicio}s_duracao-{args.duracao or 'fim'}.csv"
        dest = args.saida / game / filename
        score_audio(
            wav, dest, processor, model, index, args.lote, device, args.inicio, args.duracao,
            window_s=args.janela,
        )


if __name__ == "__main__":
    main()
