from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import certificate as CERT
import compare_norms as CN

DEFAULT_GRID = [1, 4, 16, 32, 64, 128, 256, 512]


def _load_dir(d):
    return {json.load(open(p))["chan"]: json.load(open(p)) for p in sorted(Path(d).glob("*.json"))}

def gate_frozen_repro(full_recs, frozen_dir):
    frozen = _load_dir(frozen_dir)
    diffs, checked = [], 0
    for chan, rec in full_recs.items():
        if chan not in frozen:
            continue
        fmap = {tuple(l["label"]): l for l in frozen[chan].get("labels", [])}
        for l in rec.get("labels", []):
            key = tuple(l["label"])
            if key in fmap and l.get("r_cert") is not None and fmap[key].get("r_cert") is not None:
                checked += 1
                if int(l["r_cert"]) != int(fmap[key]["r_cert"]):
                    diffs.append({"chan": chan, "label": list(key),
                                  "full": l["r_cert"], "frozen": fmap[key]["r_cert"]})
    return {"n_checked": checked, "n_mismatch": len(diffs), "mismatches": diffs,
            "exact_reproduction": (len(diffs) == 0 and checked > 0)}

def collect_rcerts(full_recs):
    strict, simple, ci, points, nonmono, oos = [], [], [], [], 0, 0
    for chan, rec in full_recs.items():
        for l in rec.get("labels", []):
            if l.get("missed_clean") is True:
                oos += 1
            if l.get("in_scope") and l.get("r_cert") is not None:
                strict.append(int(l["r_cert"]))
                simp = l.get("r_cert_simple")
                simple.append(int(simp) if simp is not None else int(l["r_cert"]))
                ci_val = l.get("r_cert_ci_aware")
                if ci_val is not None:
                    ci.append(int(ci_val))
                if l.get("monotone") is False:
                    nonmono += 1
                points.append({"chan": chan, "label": l["label"], "r_cert": int(l["r_cert"]),
                               "r_cert_ci_aware": ci_val,
                               "max_eta": l.get("max_eta"), "footprint_bits": l.get("footprint_bits")})
    return strict, simple, ci, points, nonmono, oos

def cohen_table_and_agreement(full_recs, cohen_recs, points):
    cohen_map = {}
    for chan, rec in cohen_recs.items():
        for l in rec.get("labels", []):
            cohen_map[(chan, tuple(l["label"]))] = l
    rows, agree = [], []
    for p in points:
        key = (p["chan"], tuple(p["label"]))
        cl = cohen_map.get(key)
        if cl is None:
            continue
        cohen_R = float(cl.get("best_radius", 0.0))
        d_coords = int(cl.get("d_coords", p["footprint_bits"] // 8 if p.get("footprint_bits") else 1))
        rows.append({"point_id": f"{p['chan']}{p['label']}", "r_cert_bits": p["r_cert"],
                     "k_region_bits": int(p.get("footprint_bits") or (d_coords * 8)),
                     "cohen_R": cohen_R, "d_coords": d_coords})
        agree.append({"point_id": f"{p['chan']}{p['label']}",
                      "ours_fragile_rcert_lt50": bool(p["r_cert"] < 50),
                      "cohen_abstain": bool(cohen_R == 0.0),
                      "cohen_R": cohen_R, "r_cert": p["r_cert"]})
    table = CN.comparison_table(rows) if rows else {"per_point": [], "aggregate": {}}
    if agree:
        both_fragile = sum(1 for a in agree if a["ours_fragile_rcert_lt50"] and a["cohen_abstain"])
        both_robust = sum(1 for a in agree if not a["ours_fragile_rcert_lt50"] and not a["cohen_abstain"])
        n = len(agree)
        agreement_rate = (both_fragile + both_robust) / n
    else:
        both_fragile = both_robust = 0; agreement_rate = float("nan")
    return {"table": table, "per_point_flags": agree,
            "both_fragile": both_fragile, "both_robust": both_robust,
            "agreement_rate": agreement_rate}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=str(_THIS / "runs_e14_full"))
    ap.add_argument("--cohen", default=str(_THIS / "runs_cohen"))
    ap.add_argument("--frozen", default=str(_ROOT / "amrcc" / "runs_e14"))
    ap.add_argument("--grid", nargs="*", type=int, default=DEFAULT_GRID)
    ap.add_argument("--coverage", type=float, default=0.95)
    ap.add_argument("--out", default=str(_THIS / "e14_verify.json"))
    args = ap.parse_args()

    full = _load_dir(args.full)
    cohen = _load_dir(args.cohen)
    if not full:
        print(f"no runs_e14_full at {args.full}"); return

    g1 = gate_frozen_repro(full, args.frozen)
    strict, simple, ci, points, nonmono, oos = collect_rcerts(full)
    ca_strict = CERT.certified_accuracy_curve(strict, args.grid) if strict else []
    ca_ci = CERT.certified_accuracy_curve(ci, args.grid) if ci else []
    r_at_cov = CERT.radius_at_coverage(strict, args.grid, args.coverage) if strict else 0
    r_at_cov_ci = CERT.radius_at_coverage(ci, args.grid, args.coverage) if ci else 0
    summ = CERT.summarize_rcerts(strict, simple) if strict else {}
    cohen_res = cohen_table_and_agreement(full, cohen, points) if cohen else None

    out = {
        "n_in_scope_points": len(strict), "n_out_of_scope": oos, "n_nonmonotone": nonmono,
        "gate_frozen_repro": g1,
        "certified_accuracy_strict": ca_strict, "certified_accuracy_ci_aware": ca_ci,
        "radius_at_coverage": r_at_cov, "radius_at_coverage_ci_aware": r_at_cov_ci,
        "coverage": args.coverage, "rcert_summary": summ,
        "thm8p5_rcert50_at_95": bool(r_at_cov >= 50),
        "cohen_comparison": (cohen_res["table"] if cohen_res else None),
        "cross_method_agreement": ({k: v for k, v in cohen_res.items() if k != "table"} if cohen_res else None),
        "subset_note": ("EVALUATED SUBSET (%d in-scope points), NOT the full SMAP/MSL test set; the full-"
                        "population sweep is a deferred resumable background job." % len(strict)),
    }
    json.dump(out, open(args.out, "w"), indent=2)

    print("=== E14 verification ===\n")
    print(f"[GATE 1: frozen reproduction] checked {g1['n_checked']} overlapping cells, "
          f"mismatches={g1['n_mismatch']} -> EXACT: {g1['exact_reproduction']}")
    if g1["mismatches"]:
        for m in g1["mismatches"]:
            print(f"    MISMATCH {m['chan']} {m['label']}: full={m['full']} frozen={m['frozen']}")
    print()
    print(f"[in-scope points] {len(strict)}  (out-of-scope: {oos}, non-monotone curves: {nonmono})")
    if summ:
        print(f"  strict r_cert: median={summ['r_cert_median']:.0f} min={summ['r_cert_min']:.0f} "
              f"max={summ['r_cert_max']:.0f}  frac>=50: {summ['frac_rcert_ge_50']:.1%}  "
              f"non-monotone(strict<simple): {summ.get('n_strict_lt_simple',0)}")
    print(f"\n[certified-accuracy curve, STRICT r_cert]")
    for row in ca_strict:
        print(f"  radius >= {row['radius']:>4}: {row['certified_accuracy']:.1%}")
    print(f"\n[§8.5 check] radius @ {args.coverage:.0%} coverage = {r_at_cov}  "
          f"(target >= 50 -> {'PASS' if r_at_cov >= 50 else 'BELOW TARGET'})")
    if ci:
        print(f"[CI-aware] radius @ {args.coverage:.0%} = {r_at_cov_ci}  (conservative, Sec 4.6)")
    if cohen_res:
        agg = cohen_res["table"]["aggregate"]
        print(f"\n[Cohen comparison]  (both norms, NO merged scalar -- proposal L11)")
        print(f"  median ours r_cert = {agg.get('median_ours_r_cert_bits'):.0f} bits "
              f"({agg.get('median_ours_hamming_fraction'):.4f} of region)")
        print(f"  median Cohen L2 R  = {agg.get('median_cohen_l2_radius'):.3f} signal units "
              f"({agg.get('median_cohen_l2_fraction'):.4f} L2-fraction; "
              f"~{agg.get('median_cohen_equiv_msb_flips'):.2f} equiv MSB-flips [heuristic])")
        print(f"\n[cross-method fragility agreement]")
        print(f"  both-fragile={cohen_res['both_fragile']}  both-robust={cohen_res['both_robust']}  "
              f"agreement={cohen_res['agreement_rate']:.1%}")
        print("  (points our low-r_cert flags as fragile vs points Cohen abstains on -- two independent")
        print("   certification methods agreeing on which anomalies are hard to certify)")
    print(f"\n[SUBSET] {out['subset_note']}")
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()