"""Batch evaluation of ouvindo_o_gol over several SoccerNet games.

Extracts each half's audio (cached in data/audio/), runs the pipeline with every
score variant and prints micro-averaged precision/recall/F1 at top-k, for key
events and for goals only, side by side.

Usage:
    uv run avaliar_lote.py data/games_valid.txt
"""
import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

from ouvindo_o_gol import (
    ANNOTATIONS_DIR,
    KEY_EVENTS,
    combine_score,
    evaluate,
    load_events,
    normalized_features,
    score_and_detect,
)

AUDIO_DIR = Path("data/audio")
TOP_K = (1, 3, 5, None)  # None = every detection
VARIANTS = {
    "min(vol,tom)": lambda z_vol, z_pitch: np.minimum(z_vol, z_pitch),  # v1: require both
    "so volume": lambda z_vol, z_pitch: z_vol,
    "vol+bonus tom": combine_score,  # current default
}


def extract_audio(game, half):
    wav = AUDIO_DIR / game / f"{half}.wav"
    if not wav.exists():
        wav.parent.mkdir(parents=True, exist_ok=True)
        mkv = ANNOTATIONS_DIR / game / f"{half}_224p.mkv"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(mkv), "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(wav)]
        subprocess.run(cmd, check=True)
    return wav


def goal_ranks(detections, goals_s, tol_s):
    """Rank (1-based) of the strongest detection within +-T of each goal, or None if missed."""
    ranks = []
    for g in goals_s:
        near = [i for i, (_, peak, *_) in enumerate(detections) if abs(peak - g) <= tol_s]
        ranks.append(min(near) + 1 if near else None)
    return ranks


def prf(tp, n_det, n_gt):
    p = tp / n_det if n_det else 0.0
    r = tp / n_gt if n_gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", help="text file, one SoccerNet game path per line")
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--out", default="outputs/avaliacao_lote.csv")
    args = ap.parse_args()

    games = [g.strip() for g in Path(args.games).read_text().splitlines() if g.strip()]
    # totals[variant][target][k] = [tp, n_det, n_gt], micro-averaged over every half
    totals = {v: {t: {k: [0, 0, 0] for k in TOP_K} for t in ("key", "goal")} for v in VARIANTS}
    rows = []

    for game in games:
        for half in (1, 2):
            # Features (incl. the slow pYIN) are computed once; only the score changes per variant.
            z_vol, z_pitch, _ = normalized_features(extract_audio(game, half))
            _, events = load_events(game, half)
            truth = {
                "key": [t for t, lab in events if lab in KEY_EVENTS],
                "goal": [t for t, lab in events if lab == "Goal"],
            }
            row = {"game": game.split("/")[-1], "half": half, "key_events": len(truth["key"]), "goals": len(truth["goal"])}
            for variant, combine in VARIANTS.items():
                _, _, dets = score_and_detect(z_vol, z_pitch, combine)
                for target, truth_s in truth.items():
                    for k in TOP_K:
                        sub = dets if k is None else dets[:k]
                        _, _, _, matched = evaluate(sub, truth_s, args.tol)
                        acc = totals[variant][target][k]
                        acc[0] += len(matched)
                        acc[1] += len(sub)
                        acc[2] += len(truth_s)
                ranks = goal_ranks(dets, truth["goal"], args.tol)
                row[f"ranks {variant}"] = " ".join("-" if r is None else str(r) for r in ranks)
            rows.append(row)
            print(f"done: {row['game']} | half {half}")

    names = list(VARIANTS)
    print(f"\ngoal ranks per half (- = missed)")
    print(f"{'game':48s} half goals  " + "  ".join(f"{n:>22s}" for n in names))
    for r in rows:
        ranks = "  ".join(f"{r[f'ranks {n}']:>22s}" for n in names)
        print(f"{r['game'][:48]:48s} {r['half']:>4} {r['goals']:>5}  {ranks}")

    print(f"\naggregated over {len(rows)} halves (micro-average, T=+-{args.tol:.0f}s, top-k per half)")
    print("cells: precision / recall / F1")
    print(f"{'target':6s} {'k':>4s}  " + "  ".join(f"{n:>18s}" for n in names))
    metrics = []
    for target in ("key", "goal"):
        for k in TOP_K:
            cells = []
            for n in names:
                tp, n_det, n_gt = totals[n][target][k]
                p, r, f1 = prf(tp, n_det, n_gt)
                cells.append(f"{p:.2f} / {r:.2f} / {f1:.2f}")
                metrics.append({
                    "variant": n, "target": target, "k": "all" if k is None else k,
                    "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                    "tp": tp, "detections": n_det, "ground_truth": n_gt,
                })
            print(f"{target:6s} {('all' if k is None else k):>4}  " + "  ".join(f"{c:>18s}" for c in cells))

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    metrics_out = out.with_name(f"{out.stem}_metricas.csv")
    for path, table in ((out, rows), (metrics_out, metrics)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=table[0].keys())
            w.writeheader()
            w.writerows(table)
    print(f"\nper-half goal ranks: {out}\naggregated metrics:  {metrics_out}")


if __name__ == "__main__":
    main()
