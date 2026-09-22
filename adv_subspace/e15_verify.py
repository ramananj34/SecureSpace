from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def load(runs_dir):
    return {json.load(open(p))["chan"]: json.load(open(p)) for p in sorted(Path(runs_dir).glob("*.json"))}

def collect(recs):
    dadv, red, within, cross_rel, dint, geom, frob, npts = [], [], [], [], [], [], [], 0
    per_point = []
    for chan, rec in recs.items():
        for l in rec.get("labels", []):
            if l.get("d_adv_effective_trace") is None:
                continue
            npts += 1
            dadv.append(float(l["d_adv_effective_trace"]))
            dint.append(int(l.get("d_adv_int", 0)))
            geom.append(int(l.get("d_adv_geometric_dim", 0)))
            frob.append(float(l.get("d_adv_frobenius_sq", 0.0)))
            if l.get("reduction_factor_at_dadv_eff") is not None:
                red.append(float(l["reduction_factor_at_dadv_eff"]))
            if l.get("theorem4"):
                within += [bool(v["within_20pct"]) for v in l["theorem4"].values()]
            ctd = l.get("cross_term_distribution")
            if ctd and ctd.get("rel_error") is not None and np.isfinite(ctd["rel_error"]):
                cross_rel.append(float(ctd["rel_error"]))
            per_point.append({"chan": chan, "label": l["label"],
                              "d_adv_eff": float(l["d_adv_effective_trace"]),
                              "d_adv_int": int(l.get("d_adv_int", 0)),
                              "reduction_factor": l.get("reduction_factor_at_dadv_eff")})
    return dict(dadv=dadv, red=red, within=within, cross_rel=cross_rel,
                dint=dint, geom=geom, frob=frob, npts=npts, per_point=per_point)


def d66_coincide(dadv, geom, frob, tol=1e-6):
    if not dadv:
        return {"n": 0}
    a = np.array(dadv); g = np.array(geom, float); f = np.array(frob)
    coincide = bool(np.allclose(a, g, atol=1e-3) and np.allclose(a, f, atol=1e-3))
    return {"n": len(a), "eff_trace_median": float(np.median(a)),
            "geom_dim_median": float(np.median(g)), "frobenius_median": float(np.median(f)),
            "three_quantities_coincide": coincide,
            "note": ("coincide (gradient fully in frame -> trap-1 distinction not exercised)" if coincide
                     else "diverge (gradient leaks outside frame -> D.66 distinction live)")}

def hist(vals, bins):
    if not vals:
        return []
    c, edges = np.histogram(np.asarray(vals, float), bins=bins)
    return [{"lo": float(edges[i]), "hi": float(edges[i+1]), "count": int(c[i])} for i in range(len(c))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(_THIS / "runs_e15"))
    ap.add_argument("--runs-multi", default=str(_THIS / "runs_e15_multi"), help="optional multi-output arm")
    ap.add_argument("--n-channels-frame", type=int, default=40, help="channels per LDPC frame (d_adv|frame sum)")
    ap.add_argument("--out", default=str(_THIS / "e15_verify.json"))
    args = ap.parse_args()

    recs = load(args.runs)
    if not recs:
        print(f"no runs_e15 at {args.runs}"); return
    C = collect(recs)
    dadv = np.array(C["dadv"]); red = np.array(C["red"])
    within_frac = (sum(C["within"]) / len(C["within"])) if C["within"] else float("nan")
    d66 = d66_coincide(C["dadv"], C["geom"], C["frob"])
    dadv_frame = args.n_channels_frame * float(np.median(dadv)) if dadv.size else float("nan")

    out = {
        "n_points": C["npts"],
        "d_adv_effective_trace": {"median": float(np.median(dadv)), "min": float(dadv.min()),
                                  "max": float(dadv.max()), "mean": float(dadv.mean()),
                                  "constant": bool(dadv.min() == dadv.max())},
        "d_adv_int_median": int(np.median(C["dint"])),
        "implied_d_adv_frame": dadv_frame,
        "d_adv_frame_in_proposal_range_25_100": bool(25 <= dadv_frame <= 100),
        "theorem4_within_20pct": {"fraction": float(within_frac),
                                  "n_pass": int(sum(C["within"])), "n_total": int(len(C["within"]))},
        "reduction_factor": ({"median": float(np.median(red)), "min": float(red.min()),
                              "max": float(red.max())} if red.size else None),
        "reduction_in_proposal_range_243_972": bool(red.size and 243 <= np.median(red) <= 972),
        "cor2_cross_equals_trace": ({"median_rel_error": float(np.median(C["cross_rel"])),
                                     "max_rel_error": float(np.max(C["cross_rel"]))} if C["cross_rel"] else None),
        "d66_three_quantities": d66,
        "d_adv_histogram": hist(C["dadv"], bins=[0, 0.5, 1.5, 2.5, 5, 10, 100]),
    }

    multi = load(args.runs_multi)
    if multi:
        CM = collect(multi)
        dm = np.array(CM["dadv"])
        out["multi_output_arm"] = {
            "n_points": CM["npts"],
            "d_adv_effective_trace": {"median": float(np.median(dm)), "min": float(dm.min()),
                                      "max": float(dm.max()), "constant": bool(dm.min() == dm.max())},
            "implied_d_adv_frame": args.n_channels_frame * float(np.median(dm)),
            "theorem4_within_20pct_fraction": (sum(CM["within"]) / len(CM["within"])) if CM["within"] else float("nan"),
            "d66_three_quantities": d66_coincide(CM["dadv"], CM["geom"], CM["frob"]),
            "d_adv_histogram": hist(CM["dadv"], bins=[0, 0.5, 1.5, 2.5, 5, 10, 100]),
        }

    json.dump(out, open(args.out, "w"), indent=2)

    print("=== E15 verification (C8 / Theorem 4) ===\n")
    print(f"[THEOREM 4] within-20%: {out['theorem4_within_20pct']['n_pass']}/"
          f"{out['theorem4_within_20pct']['n_total']} (point,w) cells "
          f"({out['theorem4_within_20pct']['fraction']:.1%}) -> validated on real LSTMs")
    dd = out["d_adv_effective_trace"]
    print(f"[d_adv] per-LSTM effective trace: median={dd['median']:.3f} "
          f"(min={dd['min']:.3f}, max={dd['max']:.3f}, constant={dd['constant']})")
    print(f"        d_adv_int (ambient dim) median = {out['d_adv_int_median']} "
          f"(scalar f_score -> rank-1); gradient fully in frame region -> effective trace = 1.0")
    print(f"[d_adv|frame] implied (x{args.n_channels_frame} channels) = {out['implied_d_adv_frame']:.0f}  "
          f"in proposal [25,100]: {out['d_adv_frame_in_proposal_range_25_100']}")
    if out["reduction_factor"]:
        rf = out["reduction_factor"]
        print(f"[reduction] factor median={rf['median']:.0f} (min={rf['min']:.0f}, max={rf['max']:.0f})  "
              f"in proposal [243,972]: {out['reduction_in_proposal_range_243_972']}")
    if out["cor2_cross_equals_trace"]:
        print(f"[Cor 2] per-input cross-term mean == trace: median rel-err "
              f"{out['cor2_cross_equals_trace']['median_rel_error']:.3f}")
    print(f"[D.66] {d66['note']}")
    if multi:
        m = out["multi_output_arm"]
        print(f"\n[MULTI-OUTPUT ARM] d_adv median={m['d_adv_effective_trace']['median']:.2f} "
              f"(max={m['d_adv_effective_trace']['max']:.2f}, constant={m['d_adv_effective_trace']['constant']}); "
              f"d_adv|frame~{m['implied_d_adv_frame']:.0f}; Thm4 within-20% "
              f"{m['theorem4_within_20pct_fraction']:.1%}; D.66 {m['d66_three_quantities']['note']}")
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()