from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "nullspace_attack")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import (DEVICE, prepare_model, load_streams, detect, missed, footprint_mask)
from smap_msl_dataset_api import Quantizer
from c2_attack import c2_ceiling_attack, CeilingConfig
from frame_ops import get_long_code, frame_analysis_over_stream

DLSB = 2.0 ** -7
DMIN_UB = 1423
W_INFO_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500]
K_GRID = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000]

def parity_curve(code, slot=0, header=400, slot_bits=800, trials=5, seed=0):
    rng = np.random.default_rng(seed)
    k, n = code.k, code.n
    half = (n - k) / 2.0
    out = []
    for w in W_INFO_GRID:
        tots, pars = [], []
        for _ in range(trials):
            msg = np.zeros(k, dtype=np.uint8)
            lo = header + slot * slot_bits
            idx = rng.choice(slot_bits, size=min(w, slot_bits), replace=False)
            msg[lo + idx] = 1
            cw = np.asarray(code.encode(msg)).astype(np.uint8)
            tot = int(cw.sum()); tots.append(tot); pars.append(tot - int(msg.sum()))
        out.append({"w_info": w, "wt_total_med": int(np.median(tots)), "wt_parity_med": int(np.median(pars)), "parity_over_half": float(np.median(pars) / half)})
    return out, half

def low_weight_search(code, n_adjacent=300, n_random=300, seed=0):
    rng = np.random.default_rng(seed)
    k = code.k
    best = None
    def wt_pair(i, j):
        msg = np.zeros(k, dtype=np.uint8); msg[i] = 1; msg[j] = 1
        return int(np.asarray(code.encode(msg)).astype(np.uint8).sum())
    for i in rng.choice(k - 1, size=min(n_adjacent, k - 1), replace=False):
        w = wt_pair(int(i), int(i) + 1)
        if best is None or w < best[0]:
            best = (w, int(i), int(i) + 1)
    rand_best = None
    for _ in range(n_random):
        i, j = rng.integers(0, k, size=2)
        if i == j:
            continue
        w = wt_pair(int(i), int(j))
        if rand_best is None or w < rand_best[0]:
            rand_best = (w, int(i), int(j))
    ub = min(best[0], rand_best[0])
    return {"min_pair_weight_found": best[0], "min_pair": [best[1], best[2]], "random_pair_min": int(rand_best[0]), "random_pair": [rand_best[1], rand_best[2]], "dmin_upper_bound_search": int(ub), "dmin_upper_bound_D30": DMIN_UB}

def evasion_vs_samples(chan, data_dir, runs_dir, e2_dir, cfg, q, code):
    j = json.load(open(Path(e2_dir) / f"{chan}.json"))
    me = next((m for m in j["min_evasion"] if m.get("clean_hit_quantized") and m.get("min_evasion_eps")), None)
    if me is None:
        print(f"  {chan}: no evading label in E2 sweep -- skipping (C)")
        return None
    eps = float(me["min_evasion_eps"])
    label = tuple(me["label"])
    cfg_used = me.get("config_used", "base")
    tele, cmds, train_features, labels, T = load_streams(chan, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = train_features.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    x_q = q.quantize(tele); tele_q = q.dequantize(x_q)
    if cfg_used == "strong":
        ccfg = CeilingConfig(band_T_steps=150, band_restarts=5, square_iters=400, query_budget=80)
    else:
        ccfg = CeilingConfig(band_T_steps=60, band_restarts=3, square_iters=200, query_budget=80)
    r = c2_ceiling_attack(chan, model, train_features, tele, cmds, labels, label, cfg, ccfg, eps, q, return_arrays=True)
    if not r["ceiling_missed_lattice"]: 
        r = c2_ceiling_attack(chan, model, train_features, tele, cmds, labels, label, cfg, ccfg, eps + DLSB, q, return_arrays=True)
        eps = eps + DLSB
    delta_snap = r["_delta_snap"]
    tele_recv = r["_tele_q"] + delta_snap
    full_info_wt = int(r["delta_info_weight_stream"])
    order = np.argsort(-np.abs(delta_snap))
    n_acted = int(np.count_nonzero(delta_snap))
    kgrid = [k for k in K_GRID if k < n_acted] + [n_acted]
    rows = []
    for k in kgrid:
        keep = order[:k]
        pruned = tele_q.copy()
        pruned[keep] = tele_recv[keep]
        E = detect(chan, model, train_features, pruned, cmds, cfg)
        miss = bool(missed(E, label))
        lev_p = q.quantize(pruned)
        di = (q.levels_to_bits(lev_p) ^ q.levels_to_bits(x_q)).astype(np.uint8)
        fa = frame_analysis_over_stream(di, code=code, window=100, slot=0)
        rows.append({"k_samples": int(k), "info_bit_weight": int(di.sum()), "missed": miss, "codeword_wt_med": int(fa["delta_total_weight_med"]), "frames_touched": int(fa["n_frames_touched"]), "syndrome0": fa["frac_frames_syndrome0"], "naive_flagged": fa["frac_frames_naive_flagged"]})
    k_star = next((row["k_samples"] for row in rows if row["missed"]), None)
    cw_at_kstar = next((row["codeword_wt_med"] for row in rows if row["missed"]), None)
    print(f"  {chan}: eps={eps/DLSB:.1f} LSB, full attack {n_acted} samples / "
          f"{full_info_wt} info-bits; evades from k*={k_star} samples "
          f"(codeword wt {cw_at_kstar})")
    return {"chan": chan, "eps_lsb": eps / DLSB, "label": list(label), "n_acted_samples": n_acted, "full_info_bit_weight": full_info_wt, "k_star_samples": k_star, "codeword_wt_at_kstar": cw_at_kstar, "sweep": rows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--e2-dir", default=str(_THIS / "runs_e2"))
    ap.add_argument("--out", default=str(_THIS / "runs_e3"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig(); q = Quantizer(); code = get_long_code()
    print("=== (A) parity-domination curve (code domain) ===")
    curve, half = parity_curve(code)
    for c in curve:
        print(f"  w_info={c['w_info']:4d}  wt_total={c['wt_total_med']:6d}  "
              f"wt_parity={c['wt_parity_med']:6d}  parity/((n-k)/2)={c['parity_over_half']:.2f}")
    print(f"  (n-k)/2 = {int(half)}")
    print("\n=== (B) low-weight library / d_min floor ===")
    t0 = time.time()
    libr = low_weight_search(code)
    print(f"  adjacent-pair min weight: {libr['min_pair_weight_found']} at info bits {libr['min_pair']}")
    print(f"  random-pair  min weight: {libr['random_pair_min']} at info bits {libr['random_pair']} "
          f"(QC orbit; not a ~(n-k)/2 baseline)")
    print(f"  d_min upper bound (this search): <= {libr['dmin_upper_bound_search']}  | "
          f"prior bound (D.30): <= {libr['dmin_upper_bound_D30']}  | true d_min via ISD = E6  "
          f"({time.time()-t0:.0f}s)")
    todo = args.channels or sorted(p.stem for p in Path(args.e2_dir).glob("*.json"))
    print(f"\n=== (C) LSTM evasion vs footprint samples perturbed ({len(todo)} channels) ===")
    evas = []
    for chan in todo:
        try:
            ev = evasion_vs_samples(chan, Path(args.data_dir), Path(args.runs_dir), Path(args.e2_dir), cfg, q, code)
        except Exception as e:
            print(f"  {chan}: FAILED {type(e).__name__}: {e}")
            continue
        if ev:
            evas.append(ev)
    json.dump({"parity_curve": curve, "half_parity": half, "low_weight": libr, "evasion_vs_samples": evas}, open(out_dir / "e3_summary.json", "w"), indent=2)
    print(f"\nsummary -> {out_dir / 'e3_summary.json'}  "
          f"(visuals deferred to the Day-6 notebook)")
    print("done")

if __name__ == "__main__":
    main()