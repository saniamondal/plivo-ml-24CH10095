"""
EOT predict.py — causal, cross-lingual inference.
Usage:
    python predict.py --data_dir ../eot_data/english --out predictions_english.csv
    python predict.py --data_dir ../eot_data/hindi  --out predictions_hindi.csv
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse, csv, pickle
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "eot_model.pkl"
HOP_MS, FRAME_MS = 10, 25
N_FEAT = 42


def load_audio(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=False)
    return (x.mean(axis=1) if x.ndim > 1 else x), sr


def _seg(x, sr, ps, win=1.5):
    end = int(ps * sr)
    return x[max(0, end - int(win * sr)):end]


def _frames(x, sr, fms=FRAME_MS, hms=HOP_MS):
    fl, hp = int(sr * fms / 1000), int(sr * hms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    return x[np.arange(fl) + hp * np.arange(n)[:, None]]


def _rms_db(x, sr):
    fr = _frames(x, sr)
    if len(fr) == 0:
        return np.array([], dtype=np.float32)
    return (20 * np.log10(np.sqrt(np.mean(fr**2, axis=1) + 1e-12) + 1e-12)).astype(np.float32)


def _pitch_frame(frame, sr):
    frame = frame - frame.mean()
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, "full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo, hi = int(sr / 400), min(int(sr / 60), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(sr / lag) if ac[lag] >= 0.3 else 0.0


def _f0(x, sr):
    fr = _frames(x, sr, fms=40)
    if len(fr) == 0:
        return np.array([], dtype=np.float32)
    return np.array([_pitch_frame(f, sr) for f in fr], dtype=np.float32)


def _slope(a):
    if len(a) < 2:
        return 0.0
    t = np.arange(len(a), dtype=np.float64)
    t -= t.mean()
    d = t @ t
    return float(t @ (a - a.mean()) / d) if d > 1e-12 else 0.0


def _voiced_stretches(vm):
    stretches, c = [], 0
    for v in vm:
        if v:
            c += 1
        elif c > 0:
            stretches.append(c)
            c = 0
    if c > 0:
        stretches.append(c)
    return stretches


def extract(x, sr, ps, pidx, prev_pause_durs, full_turn_x=None):
    seg = _seg(x, sr, ps)
    if len(seg) < sr // 10:
        return np.zeros(N_FEAT, dtype=np.float32)

    e = _rms_db(seg, sr)
    pitch = _f0(seg, sr)
    vm = pitch > 0
    vp = pitch[vm]

    n15 = max(1, int(150 / HOP_MS))
    n30 = max(1, int(300 / HOP_MS))
    n50 = max(1, int(500 / HOP_MS))

    def tail(a, n): return a[-n:] if len(a) >= n else a
    def tmn(a, n, d=0.0):
        t = tail(a, n)
        return float(t.mean()) if len(t) > 0 else d

    f = []
    # Energy (6)
    f.append(tmn(e, n15, -100)); f.append(tmn(e, n50, -100))
    f.append(_slope(tail(e, n30))); f.append(_slope(tail(e, n15)))
    f.append(float(e.std()) if len(e) > 1 else 0.0)
    f.append((tmn(e, n15) - float(e.mean())) if len(e) > n15 else 0.0)

    # F0 (8)
    f.append(_slope(tail(vp, 10))); f.append(_slope(tail(vp, 5)))
    f.append(float(vp[-3:].mean() / (vp.mean() + 1e-6)) if len(vp) >= 3 else 1.0)
    f.append(float(np.mean(tail(pitch, n50) > 0)) if len(pitch) > 0 else 0.0)
    f.append(float(np.mean(pitch > 0)) if len(pitch) > 0 else 0.0)
    f.append(float(vp.std() / (vp.mean() + 1e-6)) if len(vp) > 1 else 0.0)
    f.append(float(vp.max() - vp.min()) if len(vp) > 1 else 0.0)
    f.append(tmn(vp, 5))

    # Final syllable (2)
    stretches = _voiced_stretches(vm)
    if stretches:
        f.append(float(stretches[-1] * HOP_MS / 1000))
        f.append(float(stretches[-1] / (np.mean(stretches[:-1]) + 1e-6)) if len(stretches) >= 2 else 1.0)
    else:
        f.extend([0.0, 0.0])

    # Spectral (5)
    hp_s = int(sr * HOP_MS / 1000)
    nfft = int(sr * FRAME_MS / 1000); nfft += nfft % 2
    S = np.abs(librosa.stft(seg, n_fft=nfft, hop_length=hp_s))
    sc = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    f.append(tmn(sc, 5)); f.append(_slope(tail(sc, n15)))
    sro = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]; f.append(tmn(sro, 5))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)) if S.shape[1] > 1 else np.array([0.0])
    f.append(tmn(flux, 10))
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]; f.append(tmn(bw, 5))

    # MFCCs (10)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S**2 + 1e-12), sr=sr, n_mfcc=6)
    mt = mfccs[:, -n30:] if mfccs.shape[1] > n30 else mfccs
    for i in range(1, 6):
        f.append(float(mt[i].mean()) if i < mt.shape[0] else 0.0)
    if mfccs.shape[1] >= 3:
        dm = librosa.feature.delta(mfccs, width=3)
        dt = dm[:, -n30:] if dm.shape[1] > n30 else dm
        for i in range(1, 6):
            f.append(float(dt[i].mean()) if i < dt.shape[0] else 0.0)
    else:
        f.extend([0.0] * 5)

    # Speaking rate (2)
    dur_s = len(seg) / sr
    f.append(float(len(stretches) / (dur_s + 1e-6)))
    f.append(float(np.mean(stretches) * HOP_MS / 1000) if stretches else 0.0)

    # Discourse (5)
    f.append(float(pidx)); f.append(float(ps))
    cum_sil = sum(prev_pause_durs)
    cum_sp = max(0.0, ps - cum_sil)
    f.append(float(cum_sil)); f.append(float(cum_sp))
    f.append(float(cum_sp / (ps + 1e-6)) if ps > 0 else 1.0)

    # Full-turn energy features (4)
    full_seg = x[:int(ps * sr)] if full_turn_x is not None else seg
    e_full = _rms_db(full_seg, sr)
    if len(e_full) >= 2:
        f.append(float(e_full[-n50:].mean() - e_full.mean()) if len(e_full) >= n50 else 0.0)
        f.append(_slope(tail(e_full, n50)))
        f.append(float(e_full[-1] - e_full[0]))
        f.append(float(np.percentile(e_full[-n50:], 25) - np.percentile(e_full, 25)) if len(e_full) >= n50 else 0.0)
    else:
        f.extend([0.0] * 4)

    out = np.array(f[:N_FEAT], dtype=np.float32)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    return out


def build_dataset(data_dir):
    data_dir = Path(data_dir)
    rows = list(csv.DictReader(open(data_dir / "labels.csv")))
    turn_pauses = {}
    for r in rows:
        turn_pauses.setdefault(r["turn_id"], []).append(r)

    audio_cache = {}
    X, keys = [], []
    for r in rows:
        p = data_dir / r["audio_file"]
        k = str(p)
        if k not in audio_cache:
            audio_cache[k] = load_audio(p)
        audio, sr = audio_cache[k]
        tid, pidx = r["turn_id"], int(r["pause_index"])
        ps = float(r["pause_start"])
        prev_durs = [
            float(prev["pause_end"]) - float(prev["pause_start"])
            for prev in turn_pauses[tid]
            if int(prev["pause_index"]) < pidx
        ]
        X.append(extract(audio, sr, ps, pidx, prev_durs, full_turn_x=audio))
        keys.append((tid, pidx))
    return np.array(X), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out", default="predictions.csv", type=Path)
    ap.add_argument("--model", default=None, type=Path)
    args = ap.parse_args()

    model_path = args.model or MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_eot.py first.")

    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)
    clf = bundle["model"]
    scaler = bundle.get("scaler")

    X, keys = build_dataset(args.data_dir)
    np.nan_to_num(X, copy=False)
    if scaler is not None:
        X = scaler.transform(X)
        np.nan_to_num(X, copy=False)
    probs = clf.predict_proba(X)[:, 1]

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, probs):
            w.writerow([tid, pi, f"{p:.6f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
