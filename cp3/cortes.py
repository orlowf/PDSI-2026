"""Gera um clipe de 30 s por alerta a partir do vídeo local de um tempo de jogo."""

import argparse
import csv
import subprocess
from pathlib import Path

ALERTAS = Path("outputs/checkpoint-3/teste/alertas.csv")


def comandos_corte(video, horarios_s, saida, antes_s=15, depois_s=15):
    comandos = []
    for i, t in enumerate(horarios_s, 1):
        inicio = max(0.0, t - antes_s)
        duracao = t + depois_s - inicio
        destino = Path(saida) / f"corte_{i:03d}_{int(t):04d}s.mp4"
        comandos.append(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{inicio:g}", "-i", str(video),
                         "-t", f"{duracao:g}", "-c:v", "libx264", "-c:a", "aac", str(destino)])
    return comandos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="vídeo local do tempo de jogo")
    parser.add_argument("--jogo", required=True)
    parser.add_argument("--tempo", type=int, required=True)
    parser.add_argument("--metodo", default="classico")
    parser.add_argument("--saida", type=Path, default=Path(".cache/checkpoint-3/cortes"))
    args = parser.parse_args(argv)
    with ALERTAS.open(newline="", encoding="utf-8") as stream:
        horarios = sorted(float(l["pico_s"]) for l in csv.DictReader(stream)
                          if l["metodo"] == args.metodo and l["ponto"] == "f1"
                          and l["jogo"] == args.jogo and int(l["tempo"]) == args.tempo)
    args.saida.mkdir(parents=True, exist_ok=True)
    for comando in comandos_corte(args.video, horarios, args.saida):
        subprocess.run(comando, check=True)
    print(f"{len(horarios)} cortes em {args.saida}")


if __name__ == "__main__":
    main()
