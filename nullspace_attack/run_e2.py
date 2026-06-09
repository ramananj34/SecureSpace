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
from VENDOR_telemanom import VendoredConfig, Errors, batch_predict
from pipeline import load_trained_model, evaluate_anomalies
from fgsm_pgd_attacks import (DEVICE, prepare_model, load_streams, detect, missed, footprint_mask, make_channel)
from smap_msl_dataset_api import Quantizer
from nullspace_attack.nullspace_attack import c2_ceiling_attack, CeilingConfig
from frame_ops import get_long_code, frame_analysis_over_stream

HEADLINE = ["A-2", "A-3", "A-4", "A-7", "D-11", "D-16", "E-2", "E-3", "E-8", "F-2", "F-8", "G-6", "G-7", "M-3", "M-4", "M-5", "P-3", "P-7", "T-12"]
GRADIENT = ["F-5", "M-7", "S-1", "E-7", "D-1", "P-2"]
GRADIENT_ES_MAX = {"F-5": 1.8, "M-7": 1.6, "S-1": 2.0, "E-7": 4.6, "D-1": 4.9, "P-2": 11.4}
ALL_CHANNELS = HEADLINE + GRADIENT
DLSB = 2.0 ** -7
EPS_GRID = tuple(2.0 ** e for e in range(-8, 1))
N_BISECT = 4
EXT_MAX_EPS = 32.0
EPS_BAR = 2.0 ** -3
OOR_CLIP_THRESH = 0.05

def spacecraft_of(n_feat):
    return {25: "SMAP", 55: "MSL"}.get(int(n_feat), f"unknown({n_feat})")

def es_clean(chan, model, train_features, tele_test, cmds, cfg):
    ch = make_channel(chan, train_features, tele_test, cmds, cfg)
    batch_predict(model, ch, cfg, method="first", device=next(model.parameters()).device)
    errs = Errors(ch, cfg)
    errs.process_batches(ch)
    return errs.e_s.copy()

def saturation_frac(tele_q, delta_star, fp_mask, q):
    target = np.asarray(tele_q) + np.asarray(delta_star)
    acted = (fp_mask > 0) & (np.abs(delta_star) > 0)
    if acted.sum() == 0:
        return 0.0
    sat = acted & ((target > q.x_max) | (target < q.x_min))
    return float(sat.sum()) / float(acted.sum())

def run_channel(chan_id, data_dir, runs_dir, cfg, base_ccfg, strong_ccfg, q, code, stratum):
    tele, cmds, train_features, labels, T = load_streams(chan_id, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = train_features.shape[1]
    sc = spacecraft_of(n_feat)
    model = prepare_model(load_trained_model(runs_dir / chan_id / "model.pt", n_features=n_feat, device=DEVICE))
    l_s, buf = cfg.l_s, cfg.error_buffer
    x_q_stream = q.quantize(tele)
    tele_q = q.dequantize(x_q_stream)
    E0_raw = detect(chan_id, model, train_features, tele, cmds, cfg)
    E0_q = detect(chan_id, model, train_features, tele_q, cmds, cfg)
    clean_raw = evaluate_anomalies(E0_raw, labels)
    clean_q = evaluate_anomalies(E0_q, labels)
    es_q = es_clean(chan_id, model, train_features, tele_q, cmds, cfg)
    es_q_med = float(np.median(es_q)) if es_q.size else 0.0
    def _attack(label, eps, ccfg, cfg_tag, phase):
        r = c2_ceiling_attack(chan_id, model, train_features, tele, cmds, labels, label, cfg, ccfg, float(eps), q, return_arrays=True)
        fp = footprint_mask(label, l_s, buf, T)
        sat = saturation_frac(r["_tele_q"], r["_delta_star"], fp, q)
        trial = {"chan": chan_id, "label": [int(label[0]), int(label[1])], "eps": float(eps), "eps_lsb": float(eps) / DLSB, "phase": phase, "config": cfg_tag, "stratum": stratum, "spacecraft": sc, "clean_hit": True, "missed": bool(r["ceiling_missed_lattice"]), "ewma_pgd_missed": bool(r["ewma_pgd_missed"]), "ewma_pgd_restarts_used": int(r["ewma_pgd_restarts_used"]), "delta_info_weight_stream": int(r["delta_info_weight_stream"]), "realized_linf_lsb": float(r["realized_linf_lsb"]), "requant_loss_lsb": float(r["requant_loss_lsb"]), "saturation_frac": float(sat), "collateral": int(r["ceiling_collateral"]), "baseline_b": float(r["baseline_b"])}
        return trial, r["_delta_info_bits"]
    trials, min_evasion = [], []
    for label in labels:
        lbl = [int(label[0]), int(label[1])]
        a, b = int(label[0]), int(label[1])
        fp_lo, fp_hi = max(0, a - l_s - buf), min(T - 1, b + buf)
        oor_anom = float(np.mean(np.abs(tele[a:b + 1]) > 1.0))
        oor_foot = float(np.mean(np.abs(tele[fp_lo:fp_hi + 1]) > 1.0))
        i0, i1 = max(0, a - l_s), min(len(es_q) - 1, b - l_s)
        peak_q = float(es_q[i0:i1 + 1].max()) if i1 >= i0 else 0.0
        pob_q = peak_q / (es_q_med + 1e-12)
        clean_hit_q = not missed(E0_q, label)
        me_entry = {"label": lbl, "clean_hit_quantized": clean_hit_q, "oor_fraction_anomaly": oor_anom, "oor_fraction_footprint": oor_foot, "quantized_peak_over_baseline": pob_q, "min_evasion_eps": None, "min_evasion_lsb": None, "config_used": None}
        if not clean_hit_q:
            trials.append({"chan": chan_id, "label": lbl, "stratum": stratum, "spacecraft": sc, "clean_hit": False, "clean_hit_quantized": False})
            min_evasion.append(me_entry)
            continue
        arrays = {}
        base_grid = []
        for eps in EPS_GRID:
            tr, di = _attack(label, eps, base_ccfg, "base", "grid_base")
            base_grid.append(tr); trials.append(tr); arrays[float(eps)] = di
        if any(t["missed"] for t in base_grid):
            cfg_used, bisect_ccfg = "base", base_ccfg
            allg = base_grid
        else:
            cfg_used, bisect_ccfg = "strong", strong_ccfg
            strong_grid = []
            for eps in EPS_GRID:
                tr, di = _attack(label, eps, strong_ccfg, "strong", "grid_strong")
                strong_grid.append(tr); trials.append(tr); arrays[float(eps)] = di  # strong supersedes
            ext = []
            e = EPS_GRID[-1]
            while e < EXT_MAX_EPS:
                e *= 2.0
                tr, di = _attack(label, e, strong_ccfg, "strong", "grid_ext_strong")
                ext.append(tr); trials.append(tr); arrays[float(e)] = di
                if tr["missed"]:
                    break
            allg = strong_grid + ext
        missed_eps = sorted(t["eps"] for t in allg if t["missed"])
        if missed_eps:
            hi = missed_eps[0]
            below = [t["eps"] for t in allg if (not t["missed"]) and t["eps"] < hi]
            lo = max(below) if below else hi / 4.0
            for _ in range(N_BISECT):
                mid = float((lo * hi) ** 0.5)
                tr, di = _attack(label, mid, bisect_ccfg, cfg_used, f"bisect_{cfg_used}")
                trials.append(tr); arrays[float(mid)] = di
                if tr["missed"]:
                    hi = mid
                else:
                    lo = mid
            me = hi
        else:
            me = None
        me_entry["min_evasion_eps"] = me
        me_entry["min_evasion_lsb"] = (me / DLSB) if me is not None else None
        me_entry["config_used"] = cfg_used
        fa_eps = me if me is not None else max(arrays.keys())
        di_bits = arrays.get(float(fa_eps), arrays[max(arrays.keys())])
        fa = frame_analysis_over_stream(di_bits, code=code, window=100, slot=0)
        me_entry["frame_analysis"] = {"at_eps_lsb": float(fa_eps) / DLSB, "evading": me is not None, **{k: v for k, v in fa.items() if k != "per_frame"}}
        min_evasion.append(me_entry)
        arrays.clear()
    oor_max = max((m["oor_fraction_anomaly"] for m in min_evasion if "oor_fraction_anomaly" in m), default=0.0)
    pob_max = max((m["quantized_peak_over_baseline"] for m in min_evasion if "quantized_peak_over_baseline" in m), default=0.0)
    clipping_affected = bool(oor_max >= OOR_CLIP_THRESH)
    quant_stratum = "clipping_affected" if clipping_affected else ("q_headline" if pob_max < 1.0 else "q_gradient")
    nat_range_lsb = float(np.percentile(tele, 95) - np.percentile(tele, 5)) / DLSB
    nat_std_lsb = float(np.std(tele)) / DLSB
    return {"chan": chan_id, "stratum": stratum, "spacecraft": sc, "n_features": int(n_feat), "n_timesteps": int(T), "labels": [[int(a), int(b)] for a, b in labels], "clean_E_seq_raw": [[int(a), int(b)] for a, b in E0_raw], "clean_E_seq_quantized": [[int(a), int(b)] for a, b in E0_q], "clean_f0_5_raw": float(clean_raw["f0_5"]), "clean_f0_5_quantized": float(clean_q["f0_5"]), "eps_grid": [float(e) for e in EPS_GRID], "nat_range_lsb": nat_range_lsb, "nat_std_lsb": nat_std_lsb, "oor_fraction_anomaly_max": oor_max, "quantized_peak_over_baseline_max": pob_max, "clipping_affected": clipping_affected, "quant_stratum": quant_stratum, "min_evasion": min_evasion, "base_config": vars(base_ccfg), "strong_config": vars(strong_ccfg), "trials": trials}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_e2"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--base-ewma-steps", type=int, default=60)
    ap.add_argument("--base-ewma-restarts", type=int, default=3)
    ap.add_argument("--base-square-iters", type=int, default=200)
    ap.add_argument("--strong-ewma-steps", type=int, default=150)
    ap.add_argument("--strong-ewma-restarts", type=int, default=5)
    ap.add_argument("--strong-square-iters", type=int, default=400)
    ap.add_argument("--query-budget", type=int, default=80)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir, runs_dir = Path(args.data_dir), Path(args.runs_dir)
    cfg = VendoredConfig()
    base_ccfg = CeilingConfig(band_T_steps=args.base_ewma_steps, band_restarts=args.base_ewma_restarts, square_iters=args.base_square_iters, query_budget=args.query_budget)
    strong_ccfg = CeilingConfig(band_T_steps=args.strong_ewma_steps, band_restarts=args.strong_ewma_restarts, square_iters=args.strong_square_iters, query_budget=args.query_budget)
    q = Quantizer()
    code = get_long_code()
    todo = args.channels if args.channels else ALL_CHANNELS
    print(f"device={DEVICE}  channels={len(todo)}")
    print(f"base config (evaders): {base_ccfg.band_T_steps}x{base_ccfg.band_restarts}, "
          f"square={base_ccfg.square_iters}, qbudget={base_ccfg.query_budget}")
    print(f"strong config (robust tail, escalate where base finds no evasion): "
          f"{strong_ccfg.band_T_steps}x{strong_ccfg.band_restarts}, square={strong_ccfg.square_iters}")
    print(f"out={out_dir}  bar=2^-3={EPS_BAR/DLSB:.0f} LSB\n")
    log = out_dir / "sweep_log.jsonl"
    bar_miss, bar_total, clip_count = 0, 0, 0
    for i, chan_id in enumerate(todo, 1):
        stratum = "headline" if chan_id in HEADLINE else ("gradient" if chan_id in GRADIENT else "other")
        cpath = out_dir / f"{chan_id}.json"
        if cpath.exists() and not args.force:
            print(f"[{i}/{len(todo)}] {chan_id}: cached, skipping")
            continue
        t0 = time.time()
        try:
            rec = run_channel(chan_id, data_dir, runs_dir, cfg, base_ccfg, strong_ccfg, q, code, stratum)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {chan_id}: FAILED {type(e).__name__}: {e}")
            with open(log, "a") as fh:
                fh.write(json.dumps({"chan": chan_id, "error": repr(e)}) + "\n")
            continue
        tmp = cpath.with_suffix(".json.tmp")
        json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(cpath)
        dt = time.time() - t0
        labels_hit = [m for m in rec["min_evasion"] if m.get("clean_hit_quantized")]
        bar_hits = sum(1 for m in labels_hit if m.get("min_evasion_lsb") is not None and m["min_evasion_lsb"] <= EPS_BAR / DLSB + 1e-9)
        bar_miss += bar_hits; bar_total += len(labels_hit)
        clip_count += int(rec["clipping_affected"])
        me_lsbs = [(m["min_evasion_lsb"], m["config_used"]) for m in rec["min_evasion"] if m.get("min_evasion_lsb")]
        me_str = ", ".join(f"{x:.1f}({c[0] if c else '?'})" for x, c in me_lsbs) if me_lsbs else "none"
        print(f"[{i}/{len(todo)}] {chan_id} ({rec['spacecraft']}, {rec['quant_stratum']}"
              f"{', CLIP' if rec['clipping_affected'] else ''}): "
              f"min-evas LSB(cfg)=[{me_str}]  oor={rec['oor_fraction_anomaly_max']:.0%}  "
              f"q_pk/base={rec['quantized_peak_over_baseline_max']:.1f}  {dt:.0f}s")
        with open(log, "a") as fh:
            fh.write(json.dumps({"chan": chan_id, "seconds": dt, "min_evasion_lsb": [x for x, _ in me_lsbs], "clipping_affected": rec["clipping_affected"], "quant_stratum": rec["quant_stratum"]}) + "\n")
    if bar_total:
        print(f"\ncoverage @ 2^-3 ({EPS_BAR/DLSB:.0f} LSB) = min-evasion <= 16 LSB: "
              f"{bar_miss}/{bar_total} = {bar_miss/bar_total:.0%}  "
              f"| clipping-affected channels: {clip_count}/{len(todo)}")
    print("done")

if __name__ == "__main__":
    main()