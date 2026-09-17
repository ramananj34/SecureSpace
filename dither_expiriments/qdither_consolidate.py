from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent

def _load(runs_dir):
    out = {}
    for jp in sorted(Path(runs_dir).glob("*.json")):
        out[jp.stem] = json.load(open(jp))
    return out

def _sigma_grid(rec):
    return list(rec.get("sigmas_lsb", []))

def _size_sigma_row(size_rec, i_s):
    ps = size_rec.get("per_sigma")
    if not ps or i_s >= len(ps):
        return None
    s = ps[i_s]
    marg = s.get("marginal", {})
    sa = s.get("analytic_sa", {})
    ceil = s.get("ceiling", {})
    base = s.get("m4_baseline", float("nan"))
    m4 = marg.get("m4_msb_loc", float("nan"))
    return {
        "sigma_lsb": s.get("sigma_lsb"),
        "attacker_m1_ratio": sa.get("m1_ratio"),
        "marginal_m1_ratio": marg.get("m1_ratio"),
        "marginal_m1_lift": marg.get("m1_lift_over_random"),
        "m4_msb_loc": m4, "m4_baseline": base,
        "m4_over_chance": (m4 / base) if (base and np.isfinite(base) and base > 0) else float("nan"),
        "ceiling_abs_spearman": ceil.get("telem_order_abs_spearman"),
        "m1_random_baseline": s.get("m1_baseline"),
    }

def _channel_curves_at_sigma(chan_rec, i_s):
    cc = chan_rec.get("channel_curves", {})
    rc = cc.get("reconstruction_cost", [])
    im = cc.get("info_measures", {}).get("by_sigma", [])
    det = cc.get("detection_cost", {})
    row = {}
    if i_s < len(rc):
        r = rc[i_s]
        row.update({"added_rms_lsb": r.get("added_rms_lsb"), "noise_floor_db": r.get("noise_floor_db"),
                    "lsb_flip_rate": r.get("lsb_flip_rate"), "msb_flip_rate": r.get("msb_flip_rate"),
                    "level_change_rate": r.get("level_change_rate")})
    if i_s < len(im):
        m = im[i_s]
        row.update({"adjacency_mi_total": m.get("adjacency_mi_total"),
                    "signal_mi_total": m.get("signal_mi_total"),
                    "level_kl_mean": m.get("level_kl_mean"),
                    "bitplane_kl_total": m.get("bitplane_kl_total")})
    if det:
        bs = det.get("by_sigma", [])
        if i_s < len(bs):
            d = bs[i_s]
            row.update({"detection_f0_5": d.get("f0_5_mean"),
                        "detection_any_label_missed": d.get("any_label_ever_missed"),
                        "detection_f0_5_min": d.get("f0_5_min")})
        row["detection_clean_f0_5"] = det.get("clean", {}).get("f0_5")
    return row

def _adv_t5_proxy(size_rows):
    out = []
    n_sig = max((len(v) for v in size_rows.values()), default=0)
    for i in range(n_sig):
        m4x, ceil = [], []
        sigma_lsb = None
        for size, rows in size_rows.items():
            if i < len(rows) and rows[i] is not None:
                r = rows[i]; sigma_lsb = r["sigma_lsb"]
                if np.isfinite(r.get("m4_over_chance", np.nan)):
                    m4x.append(max(0.0, r["m4_over_chance"] - 1.0))
                if r.get("ceiling_abs_spearman") is not None and np.isfinite(r["ceiling_abs_spearman"]):
                    ceil.append(r["ceiling_abs_spearman"])
        coord_excess = float(np.max(m4x)) if m4x else float("nan")
        ceil_max = float(np.max(ceil)) if ceil else float("nan")
        comps = [c for c in [coord_excess, 0.5 * ceil_max] if np.isfinite(c)]
        proxy = float(np.nanmax(comps)) if comps else float("nan")
        out.append({"sigma_lsb": sigma_lsb, "coord_excess_over_chance": coord_excess,
                    "ceiling_abs_spearman_max": ceil_max, "adv_t5_proxy": proxy})
    return out

def consolidate_channel(chan, rec, cost_db_target=0.1):
    sig = _sigma_grid(rec)
    sizes = rec.get("sizes", {})
    size_rows = {sz: [_size_sigma_row(sr, i) for i in range(len(sig))] for sz, sr in sizes.items()}
    per_sigma = []
    for i, s_lsb in enumerate(sig):
        rows = {sz: r[i] for sz, r in size_rows.items() if r[i] is not None}
        cc = _channel_curves_at_sigma(rec, i)
        # pooled-over-size recovery at this sigma
        att = [r["attacker_m1_ratio"] for r in rows.values() if r["attacker_m1_ratio"] is not None]
        marg_lift = [r["marginal_m1_lift"] for r in rows.values() if r["marginal_m1_lift"] is not None]
        m4x = [r["m4_over_chance"] for r in rows.values() if np.isfinite(r.get("m4_over_chance", np.nan))]
        ceil = [r["ceiling_abs_spearman"] for r in rows.values()
                if r["ceiling_abs_spearman"] is not None and np.isfinite(r["ceiling_abs_spearman"])]
        per_sigma.append({
            "sigma_lsb": s_lsb,
            "attacker_m1_ratio_mean": float(np.mean(att)) if att else float("nan"),
            "attacker_m1_ratio_max": float(np.max(att)) if att else float("nan"),
            "marginal_m1_lift_mean": float(np.mean(marg_lift)) if marg_lift else float("nan"),
            "m4_over_chance_mean": float(np.mean(m4x)) if m4x else float("nan"),
            "m4_over_chance_max": float(np.max(m4x)) if m4x else float("nan"),
            "ceiling_abs_spearman_max": float(np.max(ceil)) if ceil else float("nan"),
            **cc,
        })
    adv = _adv_t5_proxy(size_rows)
    op = _operating_point(per_sigma, cost_db_target)
    return {"chan": chan, "spacecraft": rec.get("spacecraft"), "train_std": rec.get("train_std"),
            "sigmas_lsb": sig, "per_sigma": per_sigma, "adv_t5_curve": adv,
            "operating_point": op}

def _operating_point(per_sigma, cost_db_target):
    have_det = any("detection_f0_5" in r for r in per_sigma)
    clean_f0 = None
    for r in per_sigma:
        if r.get("detection_clean_f0_5") is not None:
            clean_f0 = r["detection_clean_f0_5"]; break
    detects_clean = None
    if have_det:
        r0 = per_sigma[0]
        detects_clean = (r0.get("detection_any_label_missed") is False)
    best = None
    for r in per_sigma:
        s = r["sigma_lsb"]
        cost_ok = (r.get("noise_floor_db") is None) or (r["noise_floor_db"] <= cost_db_target)
        if not have_det:
            det_ok = True
        elif not detects_clean:
            det_ok = False
        else:
            det_ok = (r.get("detection_any_label_missed") is False)
        if det_ok and cost_ok:
            best = s
    return {"cost_db_target": cost_db_target, "detection_available": have_det,
            "detects_cleanly_at_sigma0": detects_clean,
            "in_scope_for_detection": bool(detects_clean) if have_det else None,
            "detection_clean_f0_5": clean_f0,
            "largest_sigma_meeting_cost_and_detection": best,
            "anchor_sigma_lsb": 0.25, "anchor_sigma_value": 0.25 * (2.0 ** -7),
            "note": ("recovery is at baseline for all sigma incl 0 (E10); binding constraints are "
                     "legitimate cost and detection-preserved (relative to clean sigma=0), not recovery. "
                     "Channels failing clean detection are out-of-scope, like flat channels.")}

def pool(consolidated):
    sig = None
    for c in consolidated.values():
        sig = c["sigmas_lsb"]; break
    pooled = []
    for i in range(len(sig or [])):
        att, m4x, ceil, adj, sigm, det_missed, ndb, detf = [], [], [], [], [], [], [], []
        for c in consolidated.values():
            r = c["per_sigma"][i]
            flat = (c.get("train_std") is not None and c["train_std"] < 0.01)
            if np.isfinite(r.get("attacker_m1_ratio_mean", np.nan)):
                att.append(r["attacker_m1_ratio_mean"])
            if (not flat) and np.isfinite(r.get("m4_over_chance_mean", np.nan)):
                m4x.append(r["m4_over_chance_mean"])
            if np.isfinite(r.get("ceiling_abs_spearman_max", np.nan)):
                ceil.append(r["ceiling_abs_spearman_max"])
            if r.get("adjacency_mi_total") is not None:
                adj.append(r["adjacency_mi_total"])
            if r.get("signal_mi_total") is not None:
                sigm.append(r["signal_mi_total"])
            in_scope = (not flat) and bool(c.get("operating_point", {}).get("detects_cleanly_at_sigma0"))
            if in_scope and (r.get("detection_any_label_missed") is not None):
                det_missed.append(bool(r["detection_any_label_missed"]))
            if in_scope and (r.get("detection_f0_5") is not None):
                detf.append(r["detection_f0_5"])
            is_synth = str(c.get("chan", "")).startswith("synth") or str(c.get("spacecraft", "")) == "SYNTH"
            if (not flat) and (not is_synth) and (r.get("noise_floor_db") is not None) \
                    and np.isfinite(r["noise_floor_db"]):
                ndb.append(r["noise_floor_db"])
        pooled.append({
            "sigma_lsb": sig[i],
            "attacker_m1_ratio_mean": float(np.mean(att)) if att else float("nan"),
            "m4_over_chance_mean_nondegenerate": float(np.mean(m4x)) if m4x else float("nan"),
            "ceiling_abs_spearman_max": float(np.max(ceil)) if ceil else float("nan"),
            "adjacency_mi_total_mean": float(np.mean(adj)) if adj else float("nan"),
            "signal_mi_total_mean": float(np.mean(sigm)) if sigm else float("nan"),
            "any_inscope_label_missed": bool(any(det_missed)) if det_missed else None,
            "n_inscope_detection_channels": int(len(detf)),
            "noise_floor_db_mean_nondegenerate": float(np.mean(ndb)) if ndb else float("nan"),
            "detection_f0_5_mean_nondegenerate": float(np.mean(detf)) if detf else float("nan"),
        })
    return pooled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(_THIS / "runs_e10"))
    ap.add_argument("--out", default=str(_THIS / "runs_e10" / "_qdither_consolidated.json"))
    ap.add_argument("--cost-db-target", type=float, default=0.1)
    args = ap.parse_args()
    data = _load(args.runs)
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    consolidated = {c: consolidate_channel(c, r, args.cost_db_target) for c, r in data.items()}
    pooled = pool(consolidated)
    out = {"channels": consolidated, "pooled": pooled}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out).with_suffix(".json.tmp")
    json.dump(out, open(tmp, "w"), indent=2); tmp.replace(args.out)

    print("=== Q-DITHER consolidation ===")
    hdr = ("sigma  attackerM1r  M4/chance*  ceil|rho|  adjMI   sigMI   noiseFloor_dB  detF0.5  missed")
    print(hdr)
    for i, pr in enumerate(pooled):
        db = pr.get("noise_floor_db_mean_nondegenerate", float("nan"))
        f0 = pr.get("detection_f0_5_mean_nondegenerate", float("nan"))
        ms = pr.get("any_inscope_label_missed", None)
        db_s = f"{db:>12.4f}" if np.isfinite(db) else f"{'nan':>12}"
        f0_s = f"{f0:.3f}" if np.isfinite(f0) else "nan"
        print(f"{pr['sigma_lsb']:>4}  {pr['attacker_m1_ratio_mean']:>10.3f}  "
              f"{pr['m4_over_chance_mean_nondegenerate']:>9.2f}  "
              f"{pr['ceiling_abs_spearman_max']:>8.2f}  "
              f"{pr['adjacency_mi_total_mean']:>6.3f}  {pr['signal_mi_total_mean']:>6.3f}  "
              f"{db_s}  {f0_s:>7}  {ms}")
    print("\n* M4/chance, dB, detF0.5, missed: over NON-degenerate channels (flat_train excluded).")
    print("  Degenerate flat channels (A-1, and any std<0.01) have undefined NPDT (np.ptp(y_test)=0);")
    print("  their operating point is None by construction, not a dither failure.")
    print("Adv_T5 proxy (coord-recovery-above-chance / ceiling), per channel, is in the JSON adv_t5_curve.")
    print("\nOperating points (largest sigma_lsb meeting cost<=%.2gdB AND detection-preserved):" % args.cost_db_target)
    for c, cc in consolidated.items():
        op = cc["operating_point"]
        print(f"  {c:6} (std={cc['train_std']}): sigma*={op['largest_sigma_meeting_cost_and_detection']} "
              f"LSB  [anchor {op['anchor_sigma_lsb']} LSB ~ {op['anchor_sigma_value']:.4f}]")
    inscope = [c for c, cc in consolidated.items()
               if cc["operating_point"].get("in_scope_for_detection")]
    outscope = [c for c, cc in consolidated.items()
                if cc["operating_point"].get("detection_available")
                and not cc["operating_point"].get("in_scope_for_detection")]
    print(f"\nDetection scope: in-scope (detect cleanly at sigma=0) = {inscope}; "
          f"out-of-scope (fail clean / flat) = {outscope}.")
    print("  Detection-preserved claim + operating points are over IN-SCOPE channels only (D.61).")
    print("\nHANDOFF (Week 9): adjacency/signal-MI (residual dependence, DECREASING -> Pinsker step) and "
          "level/bitplane-KL (proposal-literal, INCREASING -> flagged loose) are in the per-sigma records; "
          "Adv_T5 proxy -> Adv_T4 (Thm 7 ~0) as sigma grows.")
    print("done")

if __name__ == "__main__":
    main()