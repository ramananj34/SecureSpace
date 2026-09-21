from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc"), str(_ROOT / "baseline_fgsm_pgd"),
           str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"),
           str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, footprint_mask
from smap_msl_dataset_api import (Quantizer, working_channels, flat_anomaly_channels, EXCLUDED_CHANNELS)
from eta_w import eta_curve, max_eta
from run_e14 import r_cert as _r_cert_frozen
import certificate as CERT

FROZEN_GRID = [1, 4, 16, 32, 64, 128, 256, 512]
FROZEN_N = 300
FROZEN_SEED = 0


def run_channel(chan, data_dir, runs_dir, cfg, q, grid, n_samples, seed):
    tele, cmds, tf, labels, T = load_streams(chan, data_dir)
    if not labels:
        return {"chan": chan, "n_labels": 0, "labels": [], "note": "no labeled test anomalies"}
    tele = np.asarray(tele, np.float64)
    n_feat = tf.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    b = int(getattr(q, "b", 8))
    out = {"chan": chan, "spacecraft": "SMAP" if n_feat == 25 else "MSL",
           "grid": list(grid), "n_samples": int(n_samples), "n_labels": len(labels), "labels": []}
    for label in labels:
        F = footprint_mask(label, cfg.l_s, cfg.error_buffer, T)
        n_bits = int(F.sum()) * b
        wgrid = [w for w in grid if 1 <= w <= n_bits]
        if not wgrid:
            out["labels"].append({"label": list(label), "footprint_bits": n_bits,
                                  "note": "footprint too small for grid", "r_cert": None})
            continue
        curve = eta_curve(chan, model, tf, tele, cmds, labels, label, cfg, q,
                          weights=wgrid, n_samples=n_samples, seed=seed)
        missed_clean = bool(curve[0]["missed_clean"]) if curve else None
        me = max_eta(curve)
        rc, simple = _r_cert_frozen(curve)
        rc_ci = CERT.r_cert_ci_aware(curve)
        in_scope = (missed_clean is False)
        out["labels"].append({
            "label": list(label), "footprint_bits": n_bits, "missed_clean": missed_clean,
            "in_scope": in_scope,
            "max_eta": me["max_eta_hat"], "max_eta_ci_hi": me["max_ci_hi"], "argmax_w": me["argmax_w"],
            "r_cert": (rc if in_scope else None),
            "r_cert_simple": (simple if in_scope else None),
            "r_cert_ci_aware": (rc_ci if in_scope else None),
            "monotone": CERT.curve_is_monotone(curve),
            "curve": [{"w": c["w"], "eta": c["eta_hat"], "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"]}
                      for c in curve],
        })
    return out

def population(data_dir):
    us = set(working_channels(data_dir=Path(data_dir)))
    fl = set(flat_anomaly_channels(data_dir=Path(data_dir)))
    return sorted((us | fl) - set(EXCLUDED_CHANNELS))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_e14_full"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--channels", nargs="*", default=None, help="subset; default = full usable+flat population")
    ap.add_argument("--grid", nargs="*", type=int, default=FROZEN_GRID)
    ap.add_argument("--n-samples", type=int, default=FROZEN_N,
                    help="MC samples per (point,w). Default 300 = W7 config -> EXACT frozen cross-check. "
                         "Raising it changes eta within CI, not qualitative r_cert.")
    ap.add_argument("--seed", type=int, default=FROZEN_SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig(); q = Quantizer()
    chans = args.channels if args.channels else population(args.data_dir)
    frozen_cfg = (list(args.grid) == FROZEN_GRID and args.n_samples == FROZEN_N and args.seed == FROZEN_SEED)
    print(f"device={DEVICE}  channels={len(chans)}  grid={args.grid}  N={args.n_samples}  "
          f"frozen-config(exact cross-check)={frozen_cfg}\n")
    all_rc, oos = [], 0
    for i, chan in enumerate(chans, 1):
        cpath = out_dir / f"{chan}.json"
        if cpath.exists() and not args.force:
            print(f"[{i}/{len(chans)}] {chan}: cached"); continue
        t0 = time.time()
        try:
            rec = run_channel(chan, Path(args.data_dir), Path(args.runs_dir), cfg, q,
                              args.grid, args.n_samples, args.seed)
        except Exception as e:
            print(f"[{i}/{len(chans)}] {chan}: FAILED {type(e).__name__}: {e}"); continue
        tmp = cpath.with_suffix(".json.tmp"); json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(cpath)
        dt = time.time() - t0
        n_scope = sum(1 for l in rec.get("labels", []) if l.get("in_scope"))
        for l in rec.get("labels", []):
            if l.get("in_scope") and l.get("r_cert") is not None:
                all_rc.append(l["r_cert"])
            elif l.get("missed_clean"):
                oos += 1
        print(f"[{i}/{len(chans)}] {chan}: {rec.get('n_labels',0)} labels, {n_scope} in-scope   {dt:.0f}s")
    if all_rc:
        a = np.array(all_rc)
        print(f"\n=== E14-full summary over {len(a)} in-scope certified points ({oos} out-of-scope) ===")
        print(f"strict r_cert: median={np.median(a):.0f}  min={a.min():.0f}  max={a.max():.0f}  "
              f"frac r_cert>=50: {np.mean(a >= 50):.1%}")
        print("  -> certified-accuracy curve + Cohen comparison in Day-6 verify.")
    print("done")

if __name__ == "__main__":
    main()