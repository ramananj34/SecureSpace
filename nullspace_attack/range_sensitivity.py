from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent; _ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT/"baseline_fgsm_pgd"), str(_ROOT/"smap_msl_data"), str(_ROOT/"telemanom_reproduction"), str(_ROOT/"nullspace_attack_utils"), str(_ROOT/"nullspace_attack")]:
    if _p not in sys.path: sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, detect, missed, footprint_mask
from smap_msl_dataset_api import Quantizer
from nullspace_attack.nullspace_attack import c2_ceiling_attack, CeilingConfig

RANGES = [1, 2, 4, 8, 16]
EPS_GRID = [2**e for e in range(-3, 6)]

def es_peak_over_base(chan, model, tf, tele_in, cmds, cfg, label):
    E = detect(chan, model, tf, tele_in, cmds, cfg)
    return (0.0 if missed(E, label) else 1.0)

def run_channel(chan, data_dir, runs_dir, cfg):
    tele, cmds, tf, labels, T = load_streams(chan, data_dir); tele = np.asarray(tele, np.float64)
    label = labels[0]
    model = prepare_model(load_trained_model(runs_dir/chan/"model.pt", n_features=tf.shape[1], device=DEVICE))
    sig_abs = float(np.std(tele[(np.arange(T)<label[0]-100)|(np.arange(T)>label[1]+100)]))
    rows = []
    for R in RANGES:
        q = Quantizer(x_min=-float(R), x_max=float(R), b=8)
        dlsb = q.delta_lsb
        tele_q = q.dequantize(q.quantize(tele))
        oor = float(np.mean(np.abs(tele) > R))
        clean_hit = not missed(detect(chan, model, tf, tele_q, cmds, cfg), label)
        ccfg = CeilingConfig(band_T_steps=60, band_restarts=3, square_iters=200, query_budget=80)
        me_abs = None
        for eps in EPS_GRID:
            if not clean_hit: break
            r = c2_ceiling_attack(chan, model, tf, tele, cmds, labels, label, cfg, ccfg, float(eps), q, return_arrays=False)
            if r["ceiling_missed_lattice"]:
                me_abs = float(eps); break
        rows.append({"range": R, "delta_lsb": dlsb, "oor_frac": oor, "clean_hit": clean_hit, "min_evasion_abs": me_abs, "min_evasion_in_sigma": (me_abs/sig_abs if me_abs and sig_abs>0 else None)})
        me_s = "robust(>32)" if (clean_hit and me_abs is None) else ("n/a-notdet" if not clean_hit else f"{me_abs/sig_abs:.2f}σ")
        print(f"  {chan} R=±{R:<2} oor={oor:5.1%} cleanHit={str(clean_hit):5} minEvas={me_s}")
    return {"chan": chan, "clean_sigma_abs": sig_abs, "rows": rows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=["D-1","M-5","P-2","F-5","M-7"])
    ap.add_argument("--data-dir", default=str(_ROOT/"smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT/"runs"))
    ap.add_argument("--out", default=str(_THIS/"runs_range"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig()
    print(f"range sensitivity: {args.channels}  R×{RANGES} (b=8 fixed; min-evasion in σ, not LSB)")
    recs = []
    for chan in args.channels:
        try: recs.append(run_channel(chan, Path(args.data_dir), Path(args.runs_dir), cfg))
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  {chan} FAILED {type(e).__name__}: {e}")
    json.dump(recs, open(out/"range.json","w"), indent=2)
    print(f"\nexpect: oor_frac -> 0 as R grows; clipping channels' min-evasion rises (reverts to robust)")
    print(f"        once R exceeds the anomaly's excursion. summary -> {out/'range.json'}")

if __name__ == "__main__":
    main()