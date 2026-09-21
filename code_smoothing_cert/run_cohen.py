from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc"), str(_ROOT / "baseline_fgsm_pgd"),
           str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, footprint_mask, detect, missed
from smap_msl_dataset_api import (Quantizer, working_channels, flat_anomaly_channels, EXCLUDED_CHANNELS)
import cohen_smoothing as CS

DEFAULT_SIGMAS = [0.05, 0.1, 0.25, 0.5]


def make_noised_detector(chan, model, tf, tele_q, cmds, cfg, F_idx, label):
    tele_q = np.asarray(tele_q, np.float64)
    def fn(_ignored, sigma, rng):
        pert = tele_q.copy()
        pert[F_idx] = pert[F_idx] + rng.normal(0.0, float(sigma), size=F_idx.size)
        E = detect(chan, model, tf, pert, cmds, cfg)
        return not missed(E, label)
    return fn


def run_channel(chan, data_dir, runs_dir, cfg, q, sigmas, n0, n, alpha, seed):
    tele, cmds, tf, labels, T = load_streams(chan, data_dir)
    if not labels:
        return {"chan": chan, "n_labels": 0, "labels": []}
    tele = np.asarray(tele, np.float64)
    n_feat = tf.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    tele_q = np.asarray(q.dequantize(q.quantize(tele)), np.float64)
    out = {"chan": chan, "spacecraft": "SMAP" if n_feat == 25 else "MSL",
           "sigmas": list(sigmas), "n0": n0, "n": n, "alpha": alpha, "n_labels": len(labels), "labels": []}
    for label in labels:
        F = footprint_mask(label, cfg.l_s, cfg.error_buffer, T)
        F_idx = np.where(F > 0)[0].astype(np.int64)
        d_coords = int(F_idx.size)
        clean_detected = not missed(detect(chan, model, tf, tele_q, cmds, cfg), label)
        if not clean_detected:
            out["labels"].append({"label": list(label), "d_coords": d_coords,
                                  "clean_detected": False, "best_radius": 0.0, "in_scope": False,
                                  "note": "not cleanly detected -- out of scope"})
            continue
        fn = make_noised_detector(chan, model, tf, tele_q, cmds, cfg, F_idx, label)
        res = CS.certify_over_sigmas(None, fn, sigmas, n0=n0, n=n, alpha=alpha, seed=seed)
        out["labels"].append({"label": list(label), "d_coords": d_coords, "clean_detected": True,
                              "in_scope": True, "best_radius": res["best_radius"],
                              "best_sigma": res["best_sigma"], "per_sigma": res["per_sigma"]})
    return out

def population(data_dir):
    us = set(working_channels(data_dir=Path(data_dir)))
    fl = set(flat_anomaly_channels(data_dir=Path(data_dir)))
    return sorted((us | fl) - set(EXCLUDED_CHANNELS))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_cohen"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--sigmas", nargs="*", type=float, default=DEFAULT_SIGMAS)
    ap.add_argument("--n0", type=int, default=100, help="CRK selection samples")
    ap.add_argument("--n", type=int, default=1000, help="CRK estimation samples (Clopper-Pearson)")
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig(); q = Quantizer()
    chans = args.channels if args.channels else population(args.data_dir)
    print(f"device={DEVICE}  channels={len(chans)}  sigmas={args.sigmas}  n0={args.n0} n={args.n}\n"
          f"(CRK Gaussian smoothing on dequantized telemetry; L2 radius in signal units)\n")
    radii = []
    for i, chan in enumerate(chans, 1):
        cpath = out_dir / f"{chan}.json"
        if cpath.exists() and not args.force:
            print(f"[{i}/{len(chans)}] {chan}: cached"); continue
        t0 = time.time()
        try:
            rec = run_channel(chan, Path(args.data_dir), Path(args.runs_dir), cfg, q,
                              args.sigmas, args.n0, args.n, args.alpha, args.seed)
        except Exception as e:
            print(f"[{i}/{len(chans)}] {chan}: FAILED {type(e).__name__}: {e}"); continue
        tmp = cpath.with_suffix(".json.tmp"); json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(cpath)
        dt = time.time() - t0
        ns = sum(1 for l in rec.get("labels", []) if l.get("in_scope"))
        for l in rec.get("labels", []):
            if l.get("in_scope"):
                radii.append(l["best_radius"])
        print(f"[{i}/{len(chans)}] {chan}: {rec.get('n_labels',0)} labels, {ns} in-scope   {dt:.0f}s")
    if radii:
        a = np.array(radii)
        print(f"\n=== Cohen summary over {len(a)} in-scope points ===")
        print(f"L2 certified radius (signal units): median={np.median(a):.3f}  max={a.max():.3f}  "
              f"frac>0 (non-abstain): {np.mean(a > 0):.1%}")
    print("done")

if __name__ == "__main__":
    main()