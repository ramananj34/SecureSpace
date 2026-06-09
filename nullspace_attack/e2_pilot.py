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
from pipeline import load_trained_model, evaluate_anomalies
from fgsm_pgd_attacks import (DEVICE, prepare_model, load_streams, detect, missed, footprint_mask)
from smap_msl_dataset_api import Quantizer
from nullspace_attack.nullspace_attack import c2_ceiling_attack, CeilingConfig
from frame_ops import get_long_code, frame_analysis_over_stream

HEADLINE = {"A-2", "A-3", "A-4", "A-7", "D-11", "D-16", "E-2", "E-3", "E-8", "F-2", "F-8", "G-6", "G-7", "M-3", "M-4", "M-5", "P-3", "P-7", "T-12"}
GRADIENT = {"F-5", "M-7", "S-1", "E-7", "D-1", "P-2"}
DLSB = 2.0 ** -7
PILOT_EPS = tuple(2.0 ** e for e in (-5, -4, -3, -2, -1, 0))

def saturation_frac(tele_q, delta_star, fp_mask, quantizer):
    target = np.asarray(tele_q) + np.asarray(delta_star)
    acted = (fp_mask > 0) & (np.abs(delta_star) > 0)
    if acted.sum() == 0:
        return 0.0
    sat = acted & ((target > quantizer.x_max) | (target < quantizer.x_min))
    return float(sat.sum()) / float(acted.sum())

def load_e1_min_evasion(chan_id, e1_runs_dir):
    p = Path(e1_runs_dir) / f"{chan_id}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return {tuple(m["label"]): m for m in d.get("min_evasion", [])}

def run_pilot(chan_id, data_dir, runs_dir, e1_runs_dir, cfg, ccfg, quantizer, eps_list, code):
    stratum = "headline" if chan_id in HEADLINE else ("gradient" if chan_id in GRADIENT else "other")
    tele, cmds, train_features, labels, T = load_streams(chan_id, data_dir)
    n_feat = train_features.shape[1]
    sc = {25: "SMAP", 55: "MSL"}.get(int(n_feat), f"unknown({n_feat})")
    model = prepare_model(load_trained_model(runs_dir / chan_id / "model.pt", n_features=n_feat, device=DEVICE))
    x_q_stream = quantizer.quantize(np.asarray(tele, np.float64))
    tele_q = quantizer.dequantize(x_q_stream)
    E0_raw = detect(chan_id, model, train_features, tele, cmds, cfg)
    E0_q = detect(chan_id, model, train_features, tele_q, cmds, cfg)
    clean_raw = evaluate_anomalies(E0_raw, labels)
    clean_q = evaluate_anomalies(E0_q, labels)
    e1_me = load_e1_min_evasion(chan_id, e1_runs_dir)
    print(f"\n=== {chan_id} ({sc}, {stratum}) ===")
    print(f"  clean f0.5: raw={clean_raw['f0_5']:.3f}  quantized={clean_q['f0_5']:.3f}  "
          f"(quantization {'PRESERVES' if clean_q['f0_5'] >= clean_raw['f0_5'] - 1e-9 else 'DEGRADES'} detection)")
    rec = {"chan": chan_id, "stratum": stratum, "spacecraft": sc, "n_features": int(n_feat), "n_timesteps": int(T), "labels": [[int(a), int(b)] for a, b in labels], "clean_f0_5_raw": float(clean_raw["f0_5"]), "clean_f0_5_quantized": float(clean_q["f0_5"]), "eps_grid_lsb": [e / DLSB for e in eps_list], "ceiling_config": vars(ccfg), "labels_detail": []}
    for label in labels:
        lbl = [int(label[0]), int(label[1])]
        clean_hit_q = not missed(E0_q, label)
        ld = {"label": lbl, "clean_hit_quantized": clean_hit_q, "trials": []}
        if not clean_hit_q:
            print(f"  label {lbl}: clean detection BROKEN by quantization — skip (not attackable)")
            rec["labels_detail"].append(ld)
            continue
        fp = footprint_mask(label, cfg.l_s, cfg.error_buffer, T)
        keep = {}
        for eps in eps_list:
            r = c2_ceiling_attack(chan_id, model, train_features, tele, cmds, labels, label, cfg, ccfg, float(eps), quantizer, return_arrays=True)
            sat = saturation_frac(r["_tele_q"], r["_delta_star"], fp, quantizer)
            trial = {"eps": float(eps), "eps_lsb": float(eps) / DLSB, "missed": bool(r["ceiling_missed_lattice"]), "ewma_pgd_missed": bool(r["ewma_pgd_missed"]), "delta_info_weight_stream": int(r["delta_info_weight_stream"]), "realized_linf_lsb": float(r["realized_linf_lsb"]), "requant_loss_lsb": float(r["requant_loss_lsb"]), "collateral": int(r["ceiling_collateral"]), "saturation_frac": float(sat)}
            ld["trials"].append(trial)
            keep[float(eps)] = (r["_delta_info_bits"], r["_delta_star"])
            flag = "MISS" if trial["missed"] else "hit "
            print(f"  label {lbl}  eps={eps:.5f} ({trial['eps_lsb']:5.1f} LSB): "
                  f"{flag} | wt(δ_info)={trial['delta_info_weight_stream']:5d} "
                  f"L∞={trial['realized_linf_lsb']:6.1f} LSB "
                  f"sat={sat:4.0%} req={trial['requant_loss_lsb']:.2f} "
                  f"fp={trial['collateral']}")
        missed_eps = sorted(t["eps"] for t in ld["trials"] if t["missed"])
        c2_me_eps = missed_eps[0] if missed_eps else None
        c2_me_lsb = (c2_me_eps / DLSB) if c2_me_eps is not None else None
        ld["c2_min_evasion_eps"] = c2_me_eps
        ld["c2_min_evasion_lsb"] = c2_me_lsb
        e1_lsb = None
        if e1_me and tuple(label) in e1_me:
            e1_lsb = e1_me[tuple(label)].get("min_evasion_lsb")
        ld["e1_min_evasion_lsb"] = e1_lsb
        c2_str = f"{c2_me_lsb:.2f} LSB" if c2_me_lsb is not None else f"none ≤{eps_list[-1]/DLSB:.0f} LSB"
        e1_str = (f"{e1_lsb:.2f} LSB" if e1_lsb is not None else ("none in E1 sweep" if e1_me is not None else "E1 n/a"))
        print(f"  → C2 min-evasion {c2_str}  |  E1 min-evasion {e1_str}")
        fa_eps = c2_me_eps if c2_me_eps is not None else eps_list[-1]
        di_bits, _ = keep[float(fa_eps)]
        fa = frame_analysis_over_stream(di_bits, code=code, window=100, slot=0)
        ld["frame_analysis"] = {"at_eps_lsb": float(fa_eps) / DLSB, "evading": c2_me_eps is not None, **{k: v for k, v in fa.items() if k != "per_frame"}}
        print(f"  → frames@{ld['frame_analysis']['at_eps_lsb']:.0f}LSB"
              f"{'(evading)' if c2_me_eps is not None else '(non-evading; mechanics only)'}: "
              f"touched={fa['n_frames_touched']} "
              f"syndrome0={fa['frac_frames_syndrome0']:.0%} "
              f"naive_flagged={fa['frac_frames_naive_flagged']:.0%} "
              f"med wt(δ)={fa['delta_total_weight_med']}")
        rec["labels_detail"].append(ld)
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=["A-3", "A-2", "D-1"])
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--e1-runs-dir", default=str(_ROOT / "baseline_fgsm_pgd" / "runs"))
    ap.add_argument("--out", default=str(_THIS / "runs_pilot"))
    ap.add_argument("--ewma-steps", type=int, default=60)
    ap.add_argument("--ewma-restarts", type=int, default=3)
    ap.add_argument("--square-iters", type=int, default=200)
    ap.add_argument("--query-budget", type=int, default=80)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig()
    ccfg = CeilingConfig(band_T_steps=args.ewma_steps, band_restarts=args.ewma_restarts, square_iters=args.square_iters, query_budget=args.query_budget)
    quantizer = Quantizer()
    code = get_long_code()
    print(f"device={DEVICE}  pilots={args.channels}  "
          f"config={ccfg.band_T_steps}x{ccfg.band_restarts}, square={ccfg.square_iters}")
    print(f"pilot eps grid (LSB): {[round(e/DLSB,1) for e in PILOT_EPS]}")
    for chan_id in args.channels:
        t0 = time.time()
        try:
            rec = run_pilot(chan_id, Path(args.data_dir), Path(args.runs_dir), Path(args.e1_runs_dir), cfg, ccfg, quantizer, PILOT_EPS, code)
        except Exception as e:
            print(f"  {chan_id}: FAILED {type(e).__name__}: {e}")
            continue
        rec["seconds"] = time.time() - t0
        p = out_dir / f"{chan_id}.json"
        tmp = p.with_suffix(".json.tmp")
        json.dump(rec, open(tmp, "w"), indent=2)
        tmp.replace(p)
        print(f"  [{chan_id}] done in {rec['seconds']:.0f}s -> {p}")
    print("\npilot done")


if __name__ == "__main__":
    main()