from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_THIS), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model, evaluate_anomalies
from fgsm_pgd_attacks import (DEVICE, prepare_model, load_streams, detect, missed)
from ceil_attack import ceiling_attack, CeilingConfig

HEADLINE = ["A-2", "A-3", "A-4", "A-7", "D-11", "D-16", "E-2", "E-3", "E-8", "F-2", "F-8", "G-6", "G-7", "M-3", "M-4", "M-5", "P-3", "P-7", "T-12"]
GRADIENT = ["F-5", "M-7", "S-1", "E-7", "D-1", "P-2"]
GRADIENT_ES_MAX = {"F-5": 1.8, "M-7": 1.6, "S-1": 2.0, "E-7": 4.6, "D-1": 4.9, "P-2": 11.4}
ALL_CHANNELS = HEADLINE + GRADIENT
EPS_GRID = tuple(2.0 ** e for e in range(-8, 1))
N_BISECT = 4
EXT_MAX_EPS = 32.0

def spacecraft_of(n_feat):
    return {25: "SMAP", 55: "MSL"}.get(int(n_feat), f"unknown({n_feat})")

def run_channel(chan_id, data_dir, runs_dir, cfg, ccfg, stratum):
    tele, cmds, train_features, labels, T = load_streams(chan_id, data_dir)
    n_feat = train_features.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan_id / "model.pt", n_features=n_feat, device=DEVICE))
    E0 = detect(chan_id, model, train_features, tele, cmds, cfg)
    clean = evaluate_anomalies(E0, labels)
    sc = spacecraft_of(n_feat)
    def _attack(label, eps, phase):
        r = ceiling_attack(chan_id, model, train_features, tele, cmds, labels, label, cfg, ccfg, float(eps))
        r["label"] = [int(label[0]), int(label[1])]
        r["stratum"], r["spacecraft"], r["clean_hit"] = stratum, sc, True
        r["e_s_max"] = GRADIENT_ES_MAX.get(chan_id, None)
        r["phase"] = phase
        return r
    trials, min_evasion = [], []
    for label in labels:
        lbl = [int(label[0]), int(label[1])]
        if missed(E0, label):
            trials.append({"chan": chan_id, "label": lbl, "stratum": stratum, "spacecraft": sc, "clean_hit": False})
            continue
        grid = [_attack(label, eps, "grid") for eps in EPS_GRID]
        trials.extend(grid)
        ext = []
        if not any(r["ceiling_missed"] for r in grid):
            e = EPS_GRID[-1]
            while e < EXT_MAX_EPS:
                e *= 2.0
                r = _attack(label, e, "grid_ext")
                ext.append(r); trials.append(r)
                if r["ceiling_missed"]:
                    break
        allg = grid + ext
        missed_eps = sorted(r["eps"] for r in allg if r["ceiling_missed"])
        if missed_eps:
            hi = missed_eps[0]
            below = [r["eps"] for r in allg if (not r["ceiling_missed"]) and r["eps"] < hi]
            lo = max(below) if below else hi / 4.0
            for _ in range(N_BISECT):
                mid = float((lo * hi) ** 0.5)
                rb = _attack(label, mid, "bisect")
                trials.append(rb)
                if rb["ceiling_missed"]:
                    hi = mid
                else:
                    lo = mid
            me = hi
        else:
            me = None
        min_evasion.append({"label": lbl, "min_evasion_eps": me, "min_evasion_lsb": (me / (2.0 ** -7)) if me is not None else None})
    dlsb = 2.0 ** -7
    nat_range_lsb = float(np.percentile(tele, 95) - np.percentile(tele, 5)) / dlsb
    nat_std_lsb = float(np.std(tele)) / dlsb
    return {"chan": chan_id, "stratum": stratum, "spacecraft": sc, "n_features": int(n_feat), "n_timesteps": int(T), "labels": [[int(a), int(b)] for (a, b) in labels], "clean_E_seq": [[int(a), int(b)] for (a, b) in E0], "clean_f0_5": float(clean["f0_5"]), "eps_grid": [float(e) for e in EPS_GRID], "nat_range_lsb": nat_range_lsb, "nat_std_lsb": nat_std_lsb, "min_evasion": min_evasion, "ceiling_config": vars(ccfg), "trials": trials}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "runs_e1_ceiling"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--query-budget", type=int, default=80)
    ap.add_argument("--square-iters", type=int, default=400)
    ap.add_argument("--ewma-steps", type=int, default=150)
    ap.add_argument("--ewma-restarts", type=int, default=5)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir, runs_dir = Path(args.data_dir), Path(args.runs_dir)
    cfg = VendoredConfig()
    ccfg = CeilingConfig(band_T_steps=args.ewma_steps, band_restarts=args.ewma_restarts, square_iters=args.square_iters, query_budget=args.query_budget)
    todo = args.channels if args.channels else ALL_CHANNELS
    print(f"device={DEVICE}  channels={len(todo)}  query_budget={ccfg.query_budget}  "
      f"square_iters={ccfg.square_iters}  ewma_steps={ccfg.band_T_steps}x{ccfg.band_restarts}")
    print(f"out={out_dir}\n")
    log = out_dir / "sweep_log.jsonl"
    for i, chan_id in enumerate(todo, 1):
        stratum = "headline" if chan_id in HEADLINE else "gradient"
        cpath = out_dir / f"{chan_id}.json"
        if cpath.exists() and not args.force:
            print(f"[{i}/{len(todo)}] {chan_id}: cached, skipping"); continue
        t0 = time.time()
        try:
            rec = run_channel(chan_id, data_dir, runs_dir, cfg, ccfg, stratum)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {chan_id}: FAILED {type(e).__name__}: {e}")
            with open(log, "a") as fh:
                fh.write(json.dumps({"chan": chan_id, "error": repr(e)}) + "\n")
            continue
        tmp = cpath.with_suffix(".json.tmp")
        json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(cpath)
        dt = time.time() - t0
        att = [t for t in rec["trials"] if t.get("clean_hit")]
        e2 = sum(1 for t in att if t.get("ewma_pgd_missed"))
        e3 = sum(1 for t in att if t.get("ceiling_missed"))
        print(f"[{i}/{len(todo)}] {chan_id} ({rec['spacecraft']}, {stratum}): "
              f"{len(att)} trials | rung2 miss={e2} rung3 miss={e3} | {dt:.0f}s")
        with open(log, "a") as fh:
            fh.write(json.dumps({"chan": chan_id, "seconds": dt, "rung2": e2, "rung3": e3}) + "\n")
    print("\ndone")

if __name__ == "__main__":
    main()