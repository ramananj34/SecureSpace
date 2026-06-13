from __future__ import annotations
import sys
import json
import time
import argparse
import platform
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "nullspace_attack")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import (DEVICE, prepare_model, load_streams, detect, missed, collateral_fp)
from smap_msl_dataset_api import Quantizer
from nullspace_attack import c2_ceiling_attack, CeilingConfig
from combined_defense import CombinedDefense, build_long, HEADER_BITS, SLOT_BITS, N_SLOTS
from e8_helpers import windows_of, aggregate_windows, collateral_msb_fraction

HEADLINE = ["A-2", "A-3", "A-4", "A-7", "D-11", "D-16", "E-2", "E-3", "E-8", "F-2", "F-8", "G-6", "G-7", "M-3", "M-4", "M-5", "P-3", "P-7", "T-12"]
GRADIENT = ["F-5", "M-7", "S-1", "E-7", "D-1", "P-2"]
ALL_CHANNELS = HEADLINE + GRADIENT
DLSB = 2.0 ** -7
SEED = 20250612

def _ccfgs(e2rec):
    return CeilingConfig(**e2rec["base_config"]), CeilingConfig(**e2rec["strong_config"])

def _demo_label(chan, model, train_features, tele, cmds, labels, label, cfg, r, cd, per_window):
    tele_recv = np.asarray(r["_tele_q"]) + np.asarray(r["_delta_snap"])
    E = detect(chan, model, train_features, tele_recv, cmds, cfg)
    tgt_missed = missed(E, label)
    demo = {"target_NPDT_missed_under_combined": bool(tgt_missed), "matches_E2_verdict": bool(tgt_missed == bool(r["ceiling_missed_lattice"])), "target_collateral_fp": int(collateral_fp(E, labels)), "E_seq": [[int(a), int(b)] for a, b in E]}
    if per_window:
        jmax = int(np.argmax([w.get("collateral_info_weight", 0) for w in per_window]))
        w0 = per_window[jmax]["window_start"]
        wbits = np.asarray(r["_delta_info_bits"])[w0:w0 + 100].reshape(800)
        proj = cd.project(wbits, 0)
        if proj is not None:
            v, _ = proj
            msb_frac, coll_tot = collateral_msb_fraction(v, HEADER_BITS, SLOT_BITS, N_SLOTS, 0)
            demo.update(heaviest_window_start=int(w0), heaviest_window_collateral_weight=int(coll_tot), heaviest_window_collateral_msb_fraction=float(msb_frac), heaviest_window_slots_touched=int(per_window[jmax]["n_slots_touched"]))
    return demo

def run_channel_e8(chan, e2rec, cd, cfg, q, data_dir, runs_dir, demo=False, keep_windows=4):
    tele, cmds, train_features, labels, T = load_streams(chan, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = train_features.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    base_ccfg, strong_ccfg = _ccfgs(e2rec)
    out_labels = []
    for me in e2rec["min_evasion"]:
        label = [int(me["label"][0]), int(me["label"][1])]
        if not me.get("clean_hit_quantized"):
            out_labels.append({"label": label, "attackable": False})
            continue
        evading = me.get("min_evasion_eps") is not None
        if evading:
            fa_eps = float(me["min_evasion_eps"])
            cfg_used = me.get("config_used") or "base"
        else:
            fa_eps = float(max(e2rec["eps_grid"]))
            cfg_used = "strong"
        ccfg = strong_ccfg if cfg_used == "strong" else base_ccfg
        r = c2_ceiling_attack(chan, model, train_features, tele, cmds, labels, label, cfg, ccfg, fa_eps, q, return_arrays=True)
        di = r["_delta_info_bits"]
        base_missed = bool(r["ceiling_missed_lattice"])
        per_window = []
        for w0, wbits in windows_of(di):
            a = cd.arms(wbits, slot=0)
            a["window_start"] = int(w0)
            per_window.append(a)
        rec = {"label": label, "attackable": True, "evading_E2": bool(evading), "base_missed_rerun": base_missed, "verdict_consistent": bool(base_missed == evading), "fa_eps": fa_eps, "fa_eps_lsb": fa_eps / DLSB, "config_used": cfg_used, "aggregate": aggregate_windows(per_window), "sample_windows": per_window[:keep_windows]}
        if demo:
            rec["demo"] = _demo_label(chan, model, train_features, tele, cmds, labels, label, cfg, r, cd, per_window)
        out_labels.append(rec)
    return {"chan": chan, "spacecraft": e2rec.get("spacecraft"), "stratum": e2rec.get("stratum"), "quant_stratum": e2rec.get("quant_stratum"), "n_timesteps": int(T), "labels": out_labels}

def _global_summary(out_dir, channels):
    n_attackable = n_evading = 0
    pass_both, naive_flag, slots, coll_w = [], [], [], []
    verdict_ok = True
    per_chan = []
    for chan in channels:
        p = out_dir / f"{chan}.json"
        if not p.exists():
            continue
        rec = json.load(open(p))
        for lab in rec["labels"]:
            if not lab.get("attackable"):
                continue
            n_attackable += 1
            ev = lab["evading_E2"]
            n_evading += int(ev)
            verdict_ok &= lab["verdict_consistent"]
            ag = lab["aggregate"]
            if ev and ag.get("n_windows_touched", 0):
                pass_both.append(ag["frac_combined_passes_both"])
                naive_flag.append(ag["frac_naive_flagged_by_combined"])
                slots.append(ag.get("n_slots_touched_med", 0))
                coll_w.append(ag.get("collateral_info_weight_med", 0))
        per_chan.append(chan)
    mean = lambda x: float(np.mean(x)) if x else None
    return {"n_channels_processed": len(per_chan), "n_labels_attackable": n_attackable, "n_labels_evading_E2": n_evading, "all_verdicts_consistent_with_E2": bool(verdict_ok), "combined_frac_passes_both_mean": mean(pass_both), "combined_evasion_equals_E2": bool(all(abs(x - 1.0) < 1e-9 for x in pass_both)) if pass_both else None, "naive_frac_flagged_by_Hprime_mean": mean(naive_flag), "collateral_slots_touched_med_mean": mean(slots), "collateral_info_weight_med_mean": mean(coll_w)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_e8"))
    ap.add_argument("--e2-dir", default=str(_ROOT / "nullspace_attack" / "runs_e2"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--demo-channel", default="A-2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    e2_dir, data_dir, runs_dir = Path(args.e2_dir), Path(args.data_dir), Path(args.runs_dir)
    cfg = VendoredConfig()
    q = Quantizer()
    print(f"platform: {platform.platform()}  |  device={DEVICE}")
    print("building B_comb (M' = H' G_LDPC ; basis of N([H;H'])) ...")
    cd = build_long(seed=args.seed, verbose=True)
    print(f"  dim N([H;H'])={cd.dim}  targetability(slot0)={cd.targetability(0)}\n")
    todo = args.channels if args.channels else ALL_CHANNELS
    log = out_dir / "sweep_log.jsonl"
    for i, chan in enumerate(todo, 1):
        cpath = out_dir / f"{chan}.json"
        if cpath.exists() and not args.force:
            print(f"[{i}/{len(todo)}] {chan}: cached, skipping")
            continue
        e2p = e2_dir / f"{chan}.json"
        if not e2p.exists():
            print(f"[{i}/{len(todo)}] {chan}: no E2 record, skipping")
            continue
        e2rec = json.load(open(e2p))
        t0 = time.time()
        try:
            rec = run_channel_e8(chan, e2rec, cd, cfg, q, data_dir, runs_dir, demo=(chan == args.demo_channel))
        except Exception as e:
            print(f"[{i}/{len(todo)}] {chan}: FAILED {type(e).__name__}: {e}")
            with open(log, "a") as fh:
                fh.write(json.dumps({"chan": chan, "error": repr(e)}) + "\n")
            continue
        tmp = cpath.with_suffix(".json.tmp")
        json.dump(rec, open(tmp, "w"), indent=2, default=float); tmp.replace(cpath)
        dt = time.time() - t0
        ev = [l for l in rec["labels"] if l.get("attackable") and l["evading_E2"]]
        if ev:
            ag = ev[0]["aggregate"]
            print(f"[{i}/{len(todo)}] {chan}: {len(ev)} evading label(s); "
                  f"combined passes-both={ag.get('frac_combined_passes_both'):.2f}, "
                  f"naive flagged-by-H'={ag.get('frac_naive_flagged_by_combined'):.2f}, "
                  f"slots touched≈{ag.get('n_slots_touched_med')}  {dt:.0f}s")
        else:
            print(f"[{i}/{len(todo)}] {chan}: not attackable / no evasion  {dt:.0f}s")
        with open(log, "a") as fh:
            fh.write(json.dumps({"chan": chan, "seconds": dt}) + "\n")
    summary = _global_summary(out_dir, todo)
    json.dump(summary, open(out_dir / "_summary.json", "w"), indent=2, default=float)
    print("\n=== E8 global summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nsaved -> {out_dir}")

if __name__ == "__main__":
    main()