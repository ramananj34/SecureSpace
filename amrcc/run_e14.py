from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, footprint_mask
from smap_msl_dataset_api import Quantizer
from eta_w import eta_curve, max_eta

DEFAULT_GRID = [1, 4, 16, 32, 64, 128, 256, 512]

def r_cert(curve):
    rc = 0
    for c in sorted(curve, key=lambda x: x["w"]):
        if c["eta_hat"] < 0.5:
            rc = c["w"]
        else:
            break
    simple = max([c["w"] for c in curve if c["eta_hat"] < 0.5], default=0)
    return rc, simple

def load_attackable(runs_e2_dir, only=None):
    out = {}
    for jp in sorted(Path(runs_e2_dir).glob("*.json")):
        rec = json.load(open(jp)); chan = rec["chan"]
        if only and chan not in only:
            continue
        labs = [[int(me["label"][0]), int(me["label"][1])] for me in rec.get("min_evasion", []) if me.get("clean_hit_quantized") and me.get("min_evasion_eps") is not None]
        if labs:
            out[chan] = labs
    return out

def run_channel(chan, labels_attack, data_dir, runs_dir, cfg, q, grid, n_samples, seed):
    tele, cmds, tf, labels, T = load_streams(chan, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = tf.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    b = int(getattr(q, "b", 8))
    out = {"chan": chan, "spacecraft": "SMAP" if n_feat == 25 else "MSL", "grid": list(grid), "n_samples": int(n_samples), "labels": []}
    for label in labels_attack:
        F = footprint_mask(label, cfg.l_s, cfg.error_buffer, T)
        n_bits = int(F.sum()) * b
        wgrid = [w for w in grid if 1 <= w <= n_bits]
        curve = eta_curve(chan, model, tf, tele, cmds, labels, label, cfg, q, weights=wgrid, n_samples=n_samples, seed=seed)
        me = max_eta(curve); rc, simple = r_cert(curve)
        out["labels"].append({"label": label, "footprint_bits": n_bits, "missed_clean": (curve[0]["missed_clean"] if curve else None), "max_eta": me["max_eta_hat"], "max_eta_ci_hi": me["max_ci_hi"], "argmax_w": me["argmax_w"], "r_cert": rc, "r_cert_simple": simple, "curve": [{"w": c["w"], "eta": c["eta_hat"], "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"]} for c in curve]})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-e2", default=str(_ROOT / "nullspace_attack" / "runs_e2"))
    ap.add_argument("--out", default=str(_THIS / "runs_e14"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--grid", nargs="*", type=int, default=DEFAULT_GRID)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig(); q = Quantizer()
    attack = load_attackable(args.runs_e2, set(args.channels) if args.channels else None)
    chans = sorted(attack)
    print(f"device={DEVICE}  channels={len(chans)}  grid={args.grid}  N={args.n_samples}\n")
    fragile, allcells = [], []
    for i, chan in enumerate(chans, 1):
        cpath = out_dir / f"{chan}.json"
        if cpath.exists() and not args.force:
            print(f"[{i}/{len(chans)}] {chan}: cached"); continue
        t0 = time.time()
        try:
            rec = run_channel(chan, attack[chan], Path(args.data_dir), Path(args.runs_dir), cfg, q, args.grid, args.n_samples, args.seed)
        except Exception as e:
            print(f"[{i}/{len(chans)}] {chan}: FAILED {type(e).__name__}: {e}"); continue
        tmp = cpath.with_suffix(".json.tmp"); json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(cpath)
        dt = time.time() - t0
        for lab in rec["labels"]:
            allcells.append(lab["max_eta"])
            tag = "FRAGILE" if lab["max_eta"] >= 0.5 else "ok"
            if lab["max_eta"] >= 0.5:
                fragile.append((chan, lab["label"], lab["max_eta"], lab["r_cert"]))
            print(f"[{i}/{len(chans)}] {chan} {lab['label']}: max_w eta_w={lab['max_eta']:.2f} "
                  f"@w={lab['argmax_w']}  r_cert={lab['r_cert']}  [{tag}]   {dt:.0f}s")
    if allcells:
        a = np.array(allcells)
        print(f"\n=== E14 summary over {len(a)} (channel,label) ===")
        print(f"max_w eta_w: median={np.median(a):.3f}  max={a.max():.2f}  "
              f"channels with eta_w<0.5 (r_cert>=top-grid): {int((a < 0.5).sum())}/{len(a)}")
    if fragile:
        print("\nFRAGILE (max_w eta_w >= 1/2 -> small r_cert; defense reduces to eta_w, E9 fails here):")
        for c, l, e, rc in fragile:
            print(f"  {c} {l}: max_w eta_w={e:.2f}  r_cert={rc}")
    print("\ndone -- E9 defended success @ realized slot weight = eta_{w_slot} <= max_w eta_w (Thm 2)")

if __name__ == "__main__":
    main()