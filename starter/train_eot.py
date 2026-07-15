"""
EOT Training Pipeline — causal, cross-lingual, competition-metric-optimised.
Usage:
    python train_eot.py --data_dir ../eot_data/english [--hindi_dir ../eot_data/hindi]
                        [--out eot_model.pkl] [--cache_dir .cache]
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse, csv, hashlib, pickle, sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

# ── constants ────────────────────────────────────────────────────────────────
HOP_MS, FRAME_MS = 10, 25
N_FEAT = 42
TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


# ── audio utilities ───────────────────────────────────────────────────────────
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


# ── feature extraction ────────────────────────────────────────────────────────
def extract(x, sr, ps, pidx, prev_pause_durs, full_turn_x=None):
    """Strictly causal: uses only audio[:pause_start]."""
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

    # — Energy (6) —
    f.append(tmn(e, n15, -100))
    f.append(tmn(e, n50, -100))
    f.append(_slope(tail(e, n30)))
    f.append(_slope(tail(e, n15)))
    f.append(float(e.std()) if len(e) > 1 else 0.0)
    f.append((tmn(e, n15) - float(e.mean())) if len(e) > n15 else 0.0)

    # — F0 (8) —
    f.append(_slope(tail(vp, 10)))
    f.append(_slope(tail(vp, 5)))
    f.append(float(vp[-3:].mean() / (vp.mean() + 1e-6)) if len(vp) >= 3 else 1.0)
    f.append(float(np.mean(tail(pitch, n50) > 0)) if len(pitch) > 0 else 0.0)
    f.append(float(np.mean(pitch > 0)) if len(pitch) > 0 else 0.0)
    f.append(float(vp.std() / (vp.mean() + 1e-6)) if len(vp) > 1 else 0.0)
    f.append(float(vp.max() - vp.min()) if len(vp) > 1 else 0.0)
    f.append(tmn(vp, 5))

    # — Final syllable (2) —
    stretches = _voiced_stretches(vm)
    if stretches:
        f.append(float(stretches[-1] * HOP_MS / 1000))
        f.append(float(stretches[-1] / (np.mean(stretches[:-1]) + 1e-6)) if len(stretches) >= 2 else 1.0)
    else:
        f.extend([0.0, 0.0])

    # — Spectral via librosa (5) —
    hp_s = int(sr * HOP_MS / 1000)
    nfft = int(sr * FRAME_MS / 1000)
    nfft += nfft % 2
    S = np.abs(librosa.stft(seg, n_fft=nfft, hop_length=hp_s))

    sc = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    f.append(tmn(sc, 5))
    f.append(_slope(tail(sc, n15)))

    sro = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    f.append(tmn(sro, 5))

    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0)) if S.shape[1] > 1 else np.array([0.0])
    f.append(tmn(flux, 10))

    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    f.append(tmn(bw, 5))

    # — MFCCs (10) —
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

    # — Speaking rate (2) —
    dur_s = len(seg) / sr
    f.append(float(len(stretches) / (dur_s + 1e-6)))
    f.append(float(np.mean(stretches) * HOP_MS / 1000) if stretches else 0.0)

    # — Discourse (5) —
    f.append(float(pidx))
    f.append(float(ps))
    cum_sil = sum(prev_pause_durs)
    cum_sp = max(0.0, ps - cum_sil)
    f.append(float(cum_sil))
    f.append(float(cum_sp))
    f.append(float(cum_sp / (ps + 1e-6)) if ps > 0 else 1.0)

    # — Full-turn energy ratio (4, causal) —
    # Uses only speech from t=0..ps (already in seg for last 1.5s,
    # but also compute full-turn normalised metrics)
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


# ── dataset building with caching ─────────────────────────────────────────────
def _cache_key(data_dir):
    p = str(Path(data_dir).resolve())
    return hashlib.md5(p.encode()).hexdigest()[:12]


def build_dataset(data_dir, cache_dir=None):
    data_dir = Path(data_dir)
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cf = cache_dir / f"feat_{_cache_key(data_dir)}.pkl"
        if cf.exists():
            with open(cf, "rb") as fh:
                return pickle.load(fh)

    rows = list(csv.DictReader(open(data_dir / "labels.csv")))
    turn_pauses = {}
    for r in rows:
        turn_pauses.setdefault(r["turn_id"], []).append(r)

    audio_cache = {}
    X, y, groups, keys = [], [], [], []
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
        y.append(1 if r.get("label", "") == "eot" else 0)
        groups.append(tid)
        keys.append((tid, pidx))

    result = (np.array(X), np.array(y), groups, keys)
    if cache_dir:
        with open(cf, "wb") as fh:
            pickle.dump(result, fh)
    return result


# ── competition metric ────────────────────────────────────────────────────────
def competition_score(pauses, budget=0.05):
    """pauses: list of dicts {turn_id, dur, label, p}"""
    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            turns_cut, turn_ids, lats = set(), set(), []
            for pz in pauses:
                turn_ids.add(pz["turn_id"])
                fires = pz["p"] >= t
                if pz["label"] == "hold":
                    if fires and d < pz["dur"]:
                        turns_cut.add(pz["turn_id"])
                else:
                    lats.append(d if fires else TIMEOUT_S)
            cut = len(turns_cut) / max(1, len(turn_ids))
            lat = float(np.mean(lats)) if lats else TIMEOUT_S
            if cut <= budget and (best is None or lat < best["lat"]):
                best = {"lat": lat, "cut": cut, "t": t, "d": d}
    if best is None:
        best = {"lat": TIMEOUT_S, "cut": 0.0, "t": 1.0, "d": TIMEOUT_S}
    return best


def oof_score(X, y, groups, keys, labels_csv, data_dir, clf_factory, scaler_factory, budget=0.05):
    """Out-of-fold competition score using GroupKFold(5)."""
    groups_arr = np.array(groups)
    probs = np.zeros(len(y), dtype=np.float64)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups_arr):
        scaler = scaler_factory()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        np.nan_to_num(Xtr, copy=False)
        np.nan_to_num(Xte, copy=False)
        clf = clf_factory()
        clf.fit(Xtr, y[tr])
        probs[te] = clf.predict_proba(Xte)[:, 1]

    # build pauses list
    rows = list(csv.DictReader(open(Path(data_dir) / "labels.csv")))
    pause_map = {(r["turn_id"], int(r["pause_index"])): r for r in rows}
    pauses = []
    for i, (tid, pidx) in enumerate(keys):
        r = pause_map[(tid, pidx)]
        pauses.append({
            "turn_id": tid,
            "dur": float(r["pause_end"]) - float(r["pause_start"]),
            "label": r["label"],
            "p": probs[i],
        })
    return competition_score(pauses, budget), probs


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--hindi_dir", type=Path, default=None)
    ap.add_argument("--out", default="eot_model.pkl", type=Path)
    ap.add_argument("--cache_dir", default=".cache", type=Path)
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args()

    print(f"[1/6] Loading English data from {args.data_dir}")
    Xe, ye, ge, ke = build_dataset(args.data_dir, args.cache_dir)
    print(f"      {len(ye)} pauses, {len(set(ge))} turns, eot={ye.mean():.2%}")

    Xall, yall, gall, kall = Xe, ye, ge, ke
    if args.hindi_dir and args.hindi_dir.exists():
        print(f"[2/6] Loading Hindi data from {args.hindi_dir}")
        Xh, yh, gh, kh = build_dataset(args.hindi_dir, args.cache_dir)
        print(f"      {len(yh)} pauses, {len(set(gh))} turns, eot={yh.mean():.2%}")
        Xall = np.vstack([Xe, Xh])
        yall = np.concatenate([ye, yh])
        gall = ge + gh
        kall = ke + kh
    else:
        print("[2/6] No Hindi dir — training on English only")

    def mk_scaler(): return RobustScaler()

    candidates = {
        "LR": lambda: LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", solver="lbfgs"),
        "GBT": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                   subsample=0.8, min_samples_leaf=4, random_state=42),
        "RF": lambda: RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=3,
                                              class_weight="balanced", random_state=42, n_jobs=-1),
        "ET": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
    }

    print("[3/6] OOF evaluation of candidate models (English only for comparability):")
    results = {}
    for name, factory in candidates.items():
        sc, probs = oof_score(Xe, ye, ge, ke, args.data_dir / "labels.csv",
                              args.data_dir, factory, mk_scaler, args.budget)
        results[name] = (sc, probs)
        print(f"      {name:5s}: latency={sc['lat']*1000:.0f}ms  cut={sc['cut']*100:.1f}%  "
              f"t={sc['t']}  d={sc['d']*1000:.0f}ms")

    best_name = min(results, key=lambda n: results[n][0]["lat"])
    print(f"[4/6] Best model: {best_name} ({results[best_name][0]['lat']*1000:.0f} ms)")

    # cross-lingual eval if Hindi available
    if args.hindi_dir and args.hindi_dir.exists():
        print("[4b] Cross-lingual (train EN → eval HI):")
        scaler = mk_scaler()
        Xetr = scaler.fit_transform(Xe)
        np.nan_to_num(Xetr, copy=False)
        clf_cross = candidates[best_name]()
        clf_cross.fit(Xetr, ye)
        Xhte = scaler.transform(Xh)
        np.nan_to_num(Xhte, copy=False)
        ph = clf_cross.predict_proba(Xhte)[:, 1]
        rows_h = list(csv.DictReader(open(args.hindi_dir / "labels.csv")))
        pm_h = {(r["turn_id"], int(r["pause_index"])): r for r in rows_h}
        pz_h = [{"turn_id": tid, "dur": float(pm_h[(tid, pi)]["pause_end"]) - float(pm_h[(tid, pi)]["pause_start"]),
                  "label": pm_h[(tid, pi)]["label"], "p": ph[i]}
                for i, (tid, pi) in enumerate(kh)]
        sc_cross = competition_score(pz_h, args.budget)
        print(f"      EN→HI: latency={sc_cross['lat']*1000:.0f}ms  cut={sc_cross['cut']*100:.1f}%")

    print("[5/6] Training final model on all available data...")
    scaler = mk_scaler()
    Xfull = scaler.fit_transform(Xall)
    np.nan_to_num(Xfull, copy=False)

    base_clf = candidates[best_name]()
    # Isotonic calibration for better probability estimates
    final_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    final_clf.fit(Xfull, yall)

    bundle = {"model": final_clf, "scaler": scaler, "best_model_name": best_name,
              "n_feat": N_FEAT, "oof_results": {k: v[0] for k, v in results.items()}}
    with open(args.out, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"      Saved → {args.out}")

    print("[6/6] Writing RUNLOG entry...")
    log = Path("RUNLOG.md")
    with open(log, "a") as fh:
        fh.write(f"\n## Run — train_eot.py (all data)\n")
        fh.write(f"- Best model: {best_name}\n")
        for name, (sc, _) in results.items():
            fh.write(f"- {name}: latency={sc['lat']*1000:.0f}ms cut={sc['cut']*100:.1f}% t={sc['t']} d={sc['d']*1000:.0f}ms\n")
    print("      Done.")


if __name__ == "__main__":
    main()
