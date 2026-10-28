"""Ouvindo o gol: find a match's key moments from the commentator's voice.

Pipeline (slide 5): audio -> voice band filter -> STFT -> detection -> evaluation.

Usage:
    uv run ouvindo_o_gol.py data/audio/burnley_arsenal_1.wav \
        --game "england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal" --half 1
"""
import argparse
import json
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

HOP_S = 0.25  # one feature frame every 250 ms
VOICE_BAND_HZ = (300, 3400)  # telephone band: keeps the voice, cuts crowd rumble
KEY_EVENTS = {"Goal", "Penalty", "Shots on target", "Shots off target", "Red card", "Yellow->red card"}
ANNOTATIONS_DIR = Path("data/soccernet/videos-224p")


def bandpass(y, sr, low, high):
    sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def robust_z(x):
    """Normalize by the narrator's own typical level (median/MAD), per slide 9."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad


def extract_features(y, sr):
    hop = int(HOP_S * sr)
    n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(S=S, frame_length=n_fft)[0], ref=np.max)
    # pYIN gives a voicing flag: plain YIN guesses a pitch even for crowd noise.
    # max_transition_rate must shrink as the hop grows, or the HMM transition matrix breaks.
    f0, voiced, _ = librosa.pyin(
        y, fmin=80, fmax=600, sr=sr, frame_length=n_fft, hop_length=hop, max_transition_rate=10
    )
    f0 = np.where(voiced, f0, np.nan)
    n = min(len(rms_db), len(f0))
    return rms_db[:n], f0[:n], S


PITCH_BONUS_WEIGHT = 0.5
PITCH_BONUS_CAP = 2.0  # max bonus = 0.5 * 2 = +1 sigma


def combine_score(z_vol, z_pitch):
    """Excitement score per frame from normalized volume and pitch.

    Volume drives the score; pitch only adds a capped bonus when it rises.
    Requiring both (np.minimum) missed most goals on 6 valid games: volume peaked
    near every goal, but pitch tracking failed for several narrators.
    """
    return z_vol + PITCH_BONUS_WEIGHT * np.clip(z_pitch, 0, PITCH_BONUS_CAP)


def detect(score, threshold, min_dur_s):
    """Return (start_s, peak_s, end_s, peak_score, strength) for runs above threshold lasting >= min_dur_s.

    Sorted by strength, the area of the score above the threshold (score-seconds):
    it grows with both intensity and duration, so a long sustained shout outranks a short spike.
    """
    above = score > threshold
    edges = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    out = []
    for s, e in zip(starts, ends):
        if (e - s) * HOP_S >= min_dur_s:
            p = s + int(np.argmax(score[s:e]))
            strength = float(np.sum(score[s:e] - threshold) * HOP_S)
            out.append((s * HOP_S, p * HOP_S, e * HOP_S, float(score[p]), strength))
    return sorted(out, key=lambda d: -d[4])


def dedupe_events(events, gap_s=2.0):
    """Drop same-label events closer than gap_s: some games annotate each goal twice (e.g. 33.18' and 33.19')."""
    kept = []
    for t, lab in sorted(events):
        if not any(lab == klab and t - kt < gap_s for kt, klab in kept):
            kept.append((t, lab))
    return kept


def load_events(game, half):
    for split in ("train", "valid", "test"):
        data = json.loads((ANNOTATIONS_DIR / f"{split}.json").read_text())["data"]
        for item in data:
            if item["id"] == f"{game}/{half}_224p.mkv":
                return split, dedupe_events([(e["position_ms"] / 1000, e["label"]) for e in item["events"]])
    raise SystemExit(f"game not found in annotations: {game}")


def evaluate(detections, truth_s, tol_s):
    """Slide 8: a detection is a hit if it falls within +-T of an unmatched ground-truth event."""
    used, tp = set(), 0
    for _, peak, *_ in detections:
        hits = [i for i, t in enumerate(truth_s) if i not in used and abs(peak - t) <= tol_s]
        if hits:
            used.add(min(hits, key=lambda i: abs(peak - truth_s[i])))
            tp += 1
    p = tp / len(detections) if detections else 0.0
    r = len(used) / len(truth_s) if truth_s else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, used


def normalized_features(wav, smooth=3.0):
    """Filter -> STFT features -> per-narrator normalization -> smoothing. Returns (z_vol, z_pitch, minutes)."""
    y, sr = sf.read(wav, dtype="float32")
    y_voice = bandpass(y, sr, *VOICE_BAND_HZ)
    vol_db, f0, _ = extract_features(y_voice, sr)

    win = max(1, int(smooth / HOP_S))
    z_vol = uniform_filter1d(robust_z(vol_db), win)
    z_pitch = uniform_filter1d(np.nan_to_num(robust_z(f0), nan=0.0), win)  # unvoiced -> neutral
    return z_vol, z_pitch, len(y) / sr / 60


def score_and_detect(z_vol, z_pitch, combine=combine_score, percentile=98.0, min_dur=2.0):
    score = combine(z_vol, z_pitch)
    # Broadcast audio is heavily compressed, so a fixed z threshold rarely fires;
    # a percentile adapts to each match and narrator.
    threshold = float(np.percentile(score, percentile))
    return score, threshold, detect(score, threshold, min_dur)


def analyze(wav, percentile=98.0, min_dur=2.0, smooth=3.0):
    """Run filter -> features -> score -> detection on one half. Returns a dict of signals and detections."""
    z_vol, z_pitch, minutes = normalized_features(wav, smooth)
    score, threshold, detections = score_and_detect(z_vol, z_pitch, combine_score, percentile, min_dur)
    return {
        "duration_min": minutes,
        "z_vol": z_vol,
        "z_pitch": z_pitch,
        "score": score,
        "threshold": threshold,
        "detections": detections,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--game", required=True)
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--percentile", type=float, default=98.0, help="threshold = this percentile of the score")
    ap.add_argument("--min-dur", type=float, default=2.0, help="seconds the shout must last")
    ap.add_argument("--smooth", type=float, default=3.0, help="moving-average window, seconds")
    ap.add_argument("--tol", type=float, default=15.0, help="+-T tolerance window, seconds")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    res = analyze(args.wav, args.percentile, args.min_dur, args.smooth)
    z_vol, z_pitch, score, detections = res["z_vol"], res["z_pitch"], res["score"], res["detections"]
    args.threshold = res["threshold"]

    split, events = load_events(args.game, args.half)
    key = [(t, lab) for t, lab in events if lab in KEY_EVENTS]
    truth_s = [t for t, _ in key]
    p, r, f1, matched = evaluate(detections, truth_s, args.tol)

    print(f"{args.game} | half {args.half} | split={split} | {res['duration_min']:.1f} min")
    print(f"key events in ground truth: {len(key)} | detections: {len(detections)} (T=+-{args.tol:.0f}s)")
    for k in (1, 3, 5, len(detections)):
        pk, rk, fk, _ = evaluate(detections[:k], truth_s, args.tol)
        print(f"  top-{k:<3d} precision={pk:.2f} recall={rk:.2f} F1={fk:.2f}")
    print("\ndetections ranked by strength (score above threshold x duration):")
    for start, peak, end, sc, strength in detections[:10]:
        near = [f"{lab}@{t / 60:.1f}'" for t, lab in events if abs(t - peak) <= args.tol]
        print(
            f"  {peak / 60:5.1f}'  strength={strength:5.2f}  score={sc:4.1f}  dur={end - start:4.1f}s"
            f"  nearby: {', '.join(near) or '-'}"
        )
    print("\nkey events and whether they were found:")
    for i, (t, lab) in enumerate(key):
        print(f"  {'OK ' if i in matched else 'MISS'} {t / 60:5.1f}'  {lab}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    t = np.arange(len(score)) * HOP_S / 60
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, z_vol, lw=0.6)
    axes[0].set_ylabel("volume (z)")
    axes[1].plot(t, z_pitch, lw=0.6, color="tab:cyan")
    axes[1].set_ylabel("pitch F0 (z)")
    axes[2].plot(t, score, lw=0.6, color="tab:olive")
    axes[2].axhline(args.threshold, ls="--", color="gray", label="threshold")
    for _, peak, *_ in detections:
        axes[2].axvline(peak / 60, color="tab:green", alpha=0.4)
    for ts, lab in key:
        for ax in axes:
            ax.axvline(ts / 60, color="red" if lab == "Goal" else "orange", ls=":", lw=1)
    axes[2].set_ylabel("score")
    axes[2].set_xlabel("minute (red=goal, orange=other key event, green=detection)")
    axes[0].set_title(f"{Path(args.game).name} - half {args.half}")
    fig.tight_layout()
    png = out / f"{Path(args.wav).stem}.png"
    fig.savefig(png, dpi=110)
    print(f"\nplot: {png}")


if __name__ == "__main__":
    main()
