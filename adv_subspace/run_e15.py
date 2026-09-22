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
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, footprint_mask
from smap_msl_dataset_api import (Quantizer, working_channels, flat_anomaly_channels, EXCLUDED_CHANNELS)
import jacobian_dadv as JD
import theorem4_energy as T4
import energy_analysis as EA

L_S = 250
FRAME_COORDS = 100
B = 8
SLOT_BITS = FRAME_COORDS * B
TAU = 0.1

def anomaly_window_indices(label, T, l_s=L_S):
    a, b = label
    t = int(max(a, l_s))
    t = min(t, T - 1)
    start = t - l_s
    if start < 0:
        return None
    return start, t

def run_point(chan, model, tf, tele, cmds, cfg, q, label, T, n_mc, seed):
    win = anomaly_window_indices(label, T, L_S)
    if win is None:
        return {"label": list(label), "note": "window out of range", "d_adv": None}
    start, t = win
    tele = np.asarray(tele, np.float64)
    x_window = np.concatenate([tele[start:start + L_S, None], cmds[start:start + L_S]], axis=1).astype(np.float32)
    input_dim = x_window.shape[1]
    J = JD.jacobian_fscore(model, x_window)
    tele_cols = JD.telemetry_columns(L_S, input_dim, telem_feature=0)
    sub = JD.adversarial_subspace(J, cols=tele_cols, tau=TAU)
    Pi_V_250 = sub["Pi_V"]; d_adv_int = sub["d_adv_int"]
    frame_region = np.arange(L_S - FRAME_COORDS, L_S).astype(np.int64)
    three = JD.dadv_all_three(Pi_V_250, frame_region)
    d_adv_eff = three["effective_trace"]

    Pi_V_100 = Pi_V_250[np.ix_(frame_region, frame_region)]
    D_frame = JD.bit_to_coord_matrix(q, FRAME_COORDS)
    c_quant = JD.c_quant_from_quantizer(q)
    frame_tele = tele[t - FRAME_COORDS:t] if t - FRAME_COORDS >= 0 else tele[start + (L_S - FRAME_COORDS):t]
    levels = q.quantize(frame_tele)
    x_q_bits = np.asarray(q.levels_to_bits(levels), np.uint8).reshape(-1)

    out = {"label": list(label), "d_adv_int": int(d_adv_int),
           "d_adv_effective_trace": float(d_adv_eff),
           "d_adv_geometric_dim": int(three["geometric_dim"]),
           "d_adv_frobenius_sq": float(three["frobenius_sq"]),
           "singular_values_top5": [float(x) for x in sub["singular_values"][:5]],
           "c_quant": float(c_quant), "k_slot": SLOT_BITS, "d_frame": FRAME_COORDS}

    if d_adv_eff <= 1e-9:
        out["note"] = "zero adversarial energy in frame region"; out["theorem4"] = None
        return out

    thm4 = {}
    for w in [1, 8, 32, 100]:
        if w >= SLOT_BITS:
            continue
        res = T4.verify_theorem4(Pi_V_100, D_frame, x_q_bits, w, SLOT_BITS, c_quant, d_adv_eff,
                                 n_mc=n_mc, seed=seed, tol=0.20)
        thm4[str(w)] = {"lhs_mc": res["lhs_mc"], "rhs_total": res["rhs"]["total"],
                        "main": res["rhs"]["main_term"], "cross_scaled": res["rhs"]["cross_term_scaled"],
                        "cross_raw": res["cross_term_raw"], "rel_error": res["rel_error"],
                        "within_20pct": res["within_tol"]}
    out["theorem4"] = thm4
    out["cross_term_distribution"] = T4.cross_equals_trace_on_average(
        Pi_V_100, D_frame, SLOT_BITS, c_quant, d_adv_eff, n_xq=1000, seed=seed)
    out["reduction_factor_at_dadv_eff"] = EA.reduction_factor(SLOT_BITS, max(d_adv_eff, 1e-6),
                                                             delta_msb=float(q.delta_msb), c_quant=c_quant)
    return out


def run_channel(chan, data_dir, runs_dir, cfg, q, n_mc, seed):
    tele, cmds, tf, labels, T = load_streams(chan, data_dir)
    if not labels:
        return {"chan": chan, "n_labels": 0, "labels": []}
    tele = np.asarray(tele, np.float64)
    n_feat = tf.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    out = {"chan": chan, "spacecraft": "SMAP" if n_feat == 25 else "MSL",
           "n_labels": len(labels), "labels": []}
    for label in labels:
        try:
            out["labels"].append(run_point(chan, model, tf, tele, cmds, cfg, q, label, T, n_mc, seed))
        except Exception as e:
            out["labels"].append({"label": list(label), "error": f"{type(e).__name__}: {e}"})
    return out


def population(data_dir):
    us = set(working_channels(data_dir=Path(data_dir)))
    fl = set(flat_anomaly_channels(data_dir=Path(data_dir)))
    return sorted((us | fl) - set(EXCLUDED_CHANNELS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_e15"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--n-mc", type=int, default=20000, help="MC draws for the Theorem-4 energy LHS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig(); q = Quantizer()
    chans = args.channels if args.channels else population(args.data_dir)
    print(f"device={DEVICE}  channels={len(chans)}  n_mc={args.n_mc}  (frame scale k={SLOT_BITS}, d={FRAME_COORDS})\n")
    dadv_all, within = [], []
    for i, chan in enumerate(chans, 1):
        cpath = out_dir / f"{chan}.json"
        if cpath.exists() and not args.force:
            print(f"[{i}/{len(chans)}] {chan}: cached"); continue
        t0 = time.time()
        try:
            rec = run_channel(chan, Path(args.data_dir), Path(args.runs_dir), cfg, q, args.n_mc, args.seed)
        except Exception as e:
            print(f"[{i}/{len(chans)}] {chan}: FAILED {type(e).__name__}: {e}"); continue
        tmp = cpath.with_suffix(".json.tmp"); json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(cpath)
        for l in rec.get("labels", []):
            if l.get("d_adv_effective_trace") is not None:
                dadv_all.append(l["d_adv_effective_trace"])
            if l.get("theorem4"):
                within += [v["within_20pct"] for v in l["theorem4"].values()]
        print(f"[{i}/{len(chans)}] {chan}: {rec.get('n_labels',0)} labels   {time.time()-t0:.0f}s")
    if dadv_all:
        a = np.array(dadv_all)
        print(f"\n=== E15 summary over {len(a)} points ===")
        print(f"per-LSTM d_adv (effective trace, frame region): median={np.median(a):.3f} "
              f"min={a.min():.3f} max={a.max():.3f}")
        print(f"implied d_adv|frame (x40 channels): ~{40*np.median(a):.0f}  "
              f"(scalar f_score -> low end of proposal's [25,100]; higher-rank f_score would raise it)")
    if within:
        print(f"Theorem 4 within-20%: {sum(within)}/{len(within)} (point,w) cells")
    print("done")

if __name__ == "__main__":
    main()