"""Score both audio window sizes for the expanded Checkpoint 2 sample."""

import argparse
import json
from pathlib import Path
from time import sleep

import torch

from avaliar_arousal import read_games
from pontuar_arousal import MODEL, REVISION, load_model, score_audio

WINDOWS = (4, 2)
MANIFESTS = ("amostra_ajuste.txt", "amostra_avaliacao.txt")


class ModelWithInputDtype:
    def __init__(self, model, dtype):
        self.model = model
        self.dtype = dtype

    def __call__(self, inputs):
        return self.model(inputs.to(dtype=self.dtype)).float()


def available_games(root):
    games = []
    for name in MANIFESTS:
        path = root / name
        if path.exists() and path.stat().st_size:
            games.extend(read_games(path))
    if len(games) != len(set(games)):
        raise ValueError("Um jogo aparece nos dois lotes.")
    return games


def acquisition_status(root):
    path = root / "amostra_estado.json"
    return json.loads(path.read_text())["estado"] if path.exists() else "executando"


def score_game(game, root, processor, model, index, batch_sizes, device):
    for half in (1, 2):
        wav = root / "audio" / game / f"{half}.wav"
        for window in WINDOWS:
            output = root / f"escores-{window}s" / game / f"{half}.csv"
            score_audio(wav, output, processor, model, index, batch_sizes[window], device, 0, None, window_s=window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, default=Path(".cache/checkpoint-2-ampliado"))
    parser.add_argument("--modelo-local", type=Path, default=Path(".cache/checkpoint-2/model"))
    parser.add_argument("--lote", type=int, default=8)
    parser.add_argument("--lote-2s", type=int, help="Tamanho do lote específico para janelas de 2 s")
    parser.add_argument("--precisao", choices=("float32", "float16"), default="float32")
    parser.add_argument("--dispositivo", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--acompanhar-midias", action="store_true")
    args = parser.parse_args()
    batch_sizes = {4: args.lote, 2: args.lote if args.lote_2s is None else args.lote_2s}
    if min(batch_sizes.values()) < 1:
        parser.error("Os lotes devem ser positivos.")
    if args.dispositivo == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    dtype = getattr(torch, args.precisao)
    args.raiz.mkdir(parents=True, exist_ok=True)
    profile_path = args.raiz / "perfil_escores.json"
    profile = {
        "modelo": MODEL, "revisao": REVISION, "torch": torch.__version__,
        "precisao": args.precisao, "dispositivo": args.dispositivo,
        "lotes_por_janela": {str(window): size for window, size in batch_sizes.items()},
    }
    if profile_path.exists() and json.loads(profile_path.read_text()) != profile:
        parser.error("Os escores existentes usam outro perfil. Indique uma raiz separada.")
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    processor, model, index = load_model(args.modelo_local, args.dispositivo)
    model = ModelWithInputDtype(model.to(dtype=dtype), dtype)
    print(
        f"{MODEL}@{REVISION} | torch {torch.__version__} | "
        f"{args.dispositivo} {args.precisao} | lotes {batch_sizes} por janela",
        flush=True,
    )
    processed = set()
    while True:
        status = acquisition_status(args.raiz) if args.acompanhar_midias else "completo"
        for game in available_games(args.raiz):
            if game in processed:
                continue
            score_game(game, args.raiz, processor, model, index, batch_sizes, args.dispositivo)
            processed.add(game)
            print(f"Jogos pontuados: {len(processed)}", flush=True)
        if status in {"completo", "incompleto", "falha"}:
            if not processed:
                raise SystemExit("Nenhuma partida disponível para pontuação.")
            print(f"Fim da pontuação: {len(processed)} jogos | amostra {status}", flush=True)
            return
        sleep(5)


if __name__ == "__main__":
    main()
