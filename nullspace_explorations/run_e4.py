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
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, detect, missed, footprint_mask
from smap_msl_dataset_api import Quantizer
from nullspace_attack import c2_ceiling_attack, CeilingConfig
from dvb_s2_rates import make_code, ALL_RATES
from frame_ops_rates import frame_analysis_over_stream

DLSB = 2.0 ** -7

def attackable_channels(e2_dir):
    out = []
    for p in sorted(Path(e2_dir).glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if any(m.get("clean_hit_quantized") and m.get("min_evasion_eps")
               for m in j.get("min_evasion", [])):
            out.append(p.stem)
    return out

def load_e2_min_evasion(chan, e2_dir):
    p = Path(e2_dir) / f"{chan}.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    me = next((m for m in j.get("min_evasion", [])
               if m.get("clean_hit_quantized") and m.get("min_evasion_eps")), None)
    if me is None:
        return None
    return {"label": tuple(me["label"]), "eps": float(me["min_evasion_eps"]), "config_used": me.get("config_used", "base")}

def run_channel(chan, data_dir, runs_dir, e2_dir, cfg, q, codes):
    info = load_e2_min_evasion(chan, e2_dir)
    if info is None:
        print(f"  {chan}: no evading label in E2 sweep -- skipping")
        return None
    eps, label, cfg_used = info["eps"], info["label"], info["config_used"]
    tele, cmds, train_features, labels, T = load_streams(chan, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = train_features.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    if cfg_used == "strong":
        ccfg = CeilingConfig(band_T_steps=150, band_restarts=5, square_iters=400, query_budget=80)
    else:
        ccfg = CeilingConfig(band_T_steps=60, band_restarts=3, square_iters=200, query_budget=80)
    r = c2_ceiling_attack(chan, model, train_features, tele, cmds, labels, label, cfg, ccfg, eps, q, return_arrays=True)
    if not r["ceiling_missed_lattice"]:
        eps += DLSB
        r = c2_ceiling_attack(chan, model, train_features, tele, cmds, labels, label, cfg, ccfg, eps, q, return_arrays=True)
    di_bits = r["_delta_info_bits"]
    verdict = bool(r["ceiling_missed_lattice"])
    info_wt = int(r["delta_info_weight_stream"])
    print(f"  {chan}: eps={eps/DLSB:.1f} LSB, label {list(label)}, real-NPDT missed={verdict}, "
          f"wt(δ_info,stream)={info_wt}")
    print(f"  {'rate':>5} {'k':>6} {'(n-k)/2':>8} {'frames':>6} {'syndrome0':>9} "
          f"{'naive_flag':>10} {'wt(δ)_med':>9} {'verdict':>7}")
    per_rate = []
    for rate in ALL_RATES:
        code = codes[rate]
        fa = frame_analysis_over_stream(di_bits, code=code, window=100, slot=0)
        floor = code.m // 2
        row = {"rate": rate, "k": code.k, "parity_floor": floor, "n_frames_touched": fa["n_frames_touched"], "frac_frames_syndrome0": fa["frac_frames_syndrome0"], "frac_frames_naive_flagged": fa["frac_frames_naive_flagged"], "delta_total_weight_med": fa["delta_total_weight_med"], "delta_total_weight_min": fa["delta_total_weight_min"], "delta_total_weight_max": fa["delta_total_weight_max"], "delta_info_weight_med": fa["delta_info_weight_med"], "real_npdt_missed": verdict}
        per_rate.append(row)
        print(f"  {rate:>5} {code.k:>6} {floor:>8} {fa['n_frames_touched']:>6} "
              f"{fa['frac_frames_syndrome0']:>9.0%} {fa['frac_frames_naive_flagged']:>10.0%} "
              f"{fa['delta_total_weight_med']:>9} {str(verdict):>7}")
    syndrome_all0 = all(p["frac_frames_syndrome0"] == 1.0 for p in per_rate)
    naive_all_flagged = all(p["frac_frames_naive_flagged"] == 1.0 for p in per_rate)
    verdict_invariant = len({p["real_npdt_missed"] for p in per_rate}) == 1
    print(f"  -> syndrome≡0 all rates: {syndrome_all0} | naive flagged all rates: {naive_all_flagged} "
          f"| verdict invariant: {verdict_invariant}")
    return {"chan": chan, "spacecraft": {25: "SMAP", 55: "MSL"}.get(int(n_feat), f"?{n_feat}"), "eps_lsb": eps / DLSB, "label": list(label), "real_npdt_missed": verdict, "delta_info_weight_stream": info_wt, "syndrome_all0": syndrome_all0, "naive_all_flagged": naive_all_flagged, "verdict_invariant": verdict_invariant, "per_rate": per_rate}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--e2-dir", default=str(_ROOT / "nullspace_attack" / "runs_e2"))
    ap.add_argument("--out", default=str(_THIS / "runs_e4"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    todo = args.channels if args.channels else attackable_channels(Path(args.e2_dir))
    print(f"attackable channels from E2 ({len(todo)}): {todo}")
    cfg = VendoredConfig(); q = Quantizer()
    print(f"device={DEVICE}  E4 code-blindness over rates {ALL_RATES}")
    print("building 7 rate codes (each H built once, reused across channels)...")
    codes = {rate: make_code(rate) for rate in ALL_RATES}
    print("done.\n")
    recs = []
    for chan in todo:
        t0 = time.time()
        try:
            r = run_channel(chan, Path(args.data_dir), Path(args.runs_dir), Path(args.e2_dir), cfg, q, codes)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {chan}: FAILED {type(e).__name__}: {e}")
            continue
        if r:
            r["seconds"] = time.time() - t0
            recs.append(r)
            p = out_dir / f"{chan}.json"
            tmp = p.with_suffix(".json.tmp"); json.dump(r, open(tmp, "w"), indent=2); tmp.replace(p)
        print()
    if recs:
        n = len(recs)
        all_synd = all(r["syndrome_all0"] for r in recs)
        all_naive = all(r["naive_all_flagged"] for r in recs)
        all_inv = all(r["verdict_invariant"] for r in recs)
        json.dump(recs, open(out_dir / "e4_summary.json", "w"), indent=2)
        print(f"=== E4 summary ({n} channels) ===")
        print(f"  syndrome≡0 every rate (β3, null-space passes):   {all_synd}")
        print(f"  naive flagged every rate (β2, LDPC stops naive): {all_naive}")
        print(f"  real-NPDT verdict rate-invariant (code-blind):   {all_inv}")
        print(f"  -> bypass COST moves with rate (parity floor); telemetry SUCCESS does not.")
        print(f"summary -> {out_dir / 'e4_summary.json'}")
    print("done")

if __name__ == "__main__":
    main()