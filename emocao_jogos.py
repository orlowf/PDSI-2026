"""Which games have the most excited commentary? Volume-only, so it skips the slow pYIN.

The detector's percentile threshold fires on ~2% of every half by design, so it
cannot compare games. Robust z doesn't work either: silent pauses inflate the MAD,
so peaks never pass a fixed z. Everything here is in dB above the half's median.

Usage:
    uv run emocao_jogos.py data/games_valid.txt data/games_valid_holdout.txt
"""
import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

from avaliar_lote import extract_audio
from ouvindo_o_gol import HOP_S, KEY_EVENTS, VOICE_BAND_HZ, bandpass, detect, load_events

SHOUT_DB = 6.0  # "shouting": 6 dB (2x amplitude) above the half's median level
MIN_SHOUT_S = 2.0
GOAL_WINDOW_S = (-5, 20)  # the goal shout starts a bit before the annotation and lasts a while


def volume_db(wav):
    y, sr = sf.read(wav, dtype="float32")
    S = np.abs(librosa.stft(bandpass(y, sr, *VOICE_BAND_HZ), n_fft=2048, hop_length=int(HOP_S * sr)))
    return librosa.amplitude_to_db(librosa.feature.rms(S=S, frame_length=2048)[0], ref=np.max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games_files", nargs="+")
    ap.add_argument("--out", default="outputs/emocao_jogos.csv")
    args = ap.parse_args()

    games = [g.strip() for f in args.games_files for g in Path(f).read_text().splitlines() if g.strip()]
    rows = []
    for game in games:
        rise_db, goal_rise, shout_frames, shouts, frames, goals, key = [], [], 0, 0, 0, 0, 0
        for half in (1, 2):
            vol = volume_db(extract_audio(game, half))
            rel = uniform_filter1d(vol, int(3.0 / HOP_S)) - np.median(vol)  # dB above typical, 3 s smoothing
            rise_db.append(np.percentile(rel, 99.5))
            shout_frames += int(np.sum(rel > SHOUT_DB))
            shouts += len(detect(rel, SHOUT_DB, MIN_SHOUT_S))
            frames += len(rel)
            _, events = load_events(game, half)
            for t, lab in events:
                if lab == "Goal":
                    lo, hi = (int((t + d) / HOP_S) for d in GOAL_WINDOW_S)
                    goal_rise.append(rel[max(lo, 0):hi].max())
            goals += sum(lab == "Goal" for _, lab in events)
            key += sum(lab in KEY_EVENTS for _, lab in events)
        minutes = frames * HOP_S / 60
        rows.append({
            "game": game.split("/")[-1],
            "league": game.split("/")[0],
            "goals": goals,
            "key_events": key,
            "peak_rise_db": round(float(np.mean(rise_db)), 1),
            "goal_rise_db": round(float(np.mean(goal_rise)), 1) if goal_rise else None,
            "pct_shouting": round(100 * shout_frames / frames, 2),
            "shouts_per_90": round(shouts / minutes * 90, 1),
        })
        print(f"done: {rows[-1]['game']}")

    rows.sort(key=lambda r: -r["shouts_per_90"])
    print(f"\n{'game':38s} {'goals':>5} {'key':>4} {'rise dB':>8} {'goal dB':>8} {'% shout':>8} {'shouts/90':>10}")
    for r in rows:
        print(f"{r['game'][:38]:38s} {r['goals']:>5} {r['key_events']:>4} {r['peak_rise_db']:>8} "
              f"{str(r['goal_rise_db']):>8} {r['pct_shouting']:>8} {r['shouts_per_90']:>10}")
    g = np.array([r["goals"] for r in rows])
    for col in ("peak_rise_db", "goal_rise_db", "pct_shouting", "shouts_per_90"):
        rho = np.corrcoef(np.argsort(np.argsort(g)), np.argsort(np.argsort([r[col] for r in rows])))[0, 1]
        print(f"Spearman(goals, {col}) = {rho:+.2f}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
