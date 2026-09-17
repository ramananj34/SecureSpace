from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent

def _pooled(consolidated_path):
    con = json.load(open(consolidated_path))
    return con.get("pooled", []), con.get("channels", {})

def _adv_t5_from_pooled(pool_rows):
    out = []
    for r in pool_rows:
        a = r.get("attacker_m1_ratio_mean")
        out.append({"sigma_lsb": r["sigma_lsb"],
                    "adv_t5_bitacc_excess": (max(0.0, float(a) - 1.0) if a is not None else float("nan")),
                    "adjacency_mi": r.get("adjacency_mi_total_mean"),
                    "signal_mi": r.get("signal_mi_total_mean")})
    return out

def _adv_t5_ceiling_from_channels(channels):
    by_sigma = {}
    for c, cc in channels.items():
        for row in cc.get("adv_t5_curve", []):
            s = row["sigma_lsb"]
            v = row.get("adv_t5_proxy")
            if v is not None and np.isfinite(v):
                by_sigma.setdefault(s, []).append(v)
    return {s: float(np.max(vs)) for s, vs in by_sigma.items()}

def check(consolidated_path, adv_t4=0.0):
    pool_rows, channels = _pooled(consolidated_path)
    t5 = _adv_t5_from_pooled(pool_rows)
    ceil = _adv_t5_ceiling_from_channels(channels)
    rows = []
    for r in t5:
        s = r["sigma_lsb"]
        adv = r["adv_t5_bitacc_excess"]
        adv_ceiling_diag = ceil.get(s, float("nan"))
        adj, sig = r["adjacency_mi"], r["signal_mi"]
        bound_adj = adv_t4 + (np.sqrt(adj / 2.0) if adj is not None and adj >= 0 else float("nan"))
        bound_sig = adv_t4 + (np.sqrt(sig / 2.0) if sig is not None and sig >= 0 else float("nan"))
        proxies = [b for b in [bound_adj, bound_sig] if np.isfinite(b)]
        bound_min = min(proxies) if proxies else float("nan")
        adv_t5 = adv if np.isfinite(adv) else float("nan")
        rows.append({
            "sigma_lsb": s, "adv_t5_bitacc": adv, "adv_t5_max": adv_t5,
            "ceiling_diag_upper_bound": adv_ceiling_diag,
            "adjacency_mi": adj, "signal_mi": sig,
            "bound_adj": bound_adj, "bound_sig": bound_sig, "bound_min_proxy": bound_min,
            "holds": bool(np.isfinite(bound_min) and np.isfinite(adv_t5) and adv_t5 <= bound_min + 1e-9),
            "margin": (bound_min - adv_t5) if (np.isfinite(bound_min) and np.isfinite(adv_t5)) else float("nan"),
        })
    return rows

def summarize(rows, adv_t4):
    holds = [r for r in rows if r["holds"]]
    adjs = [r["bound_adj"] for r in rows if np.isfinite(r["bound_adj"])]
    monotone = all(adjs[i] >= adjs[i + 1] - 1e-9 for i in range(len(adjs) - 1)) if len(adjs) > 1 else True
    margins = [r["margin"] for r in rows if np.isfinite(r["margin"])]
    return {
        "n": len(rows), "n_hold": len(holds),
        "bound_monotone_decreasing_in_sigma": bool(monotone),
        "bound_at_sigma0": rows[0]["bound_min_proxy"] if rows else float("nan"),
        "adv_t5_at_sigma0": rows[0]["adv_t5_max"] if rows else float("nan"),
        "margin_at_sigma0": rows[0]["margin"] if rows else float("nan"),
        "min_margin": float(np.min(margins)) if margins else float("nan"),
        "bound_at_sigma_max": rows[-1]["bound_min_proxy"] if rows else float("nan"),
        "adv_t4": adv_t4,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidated",
                    default=str(_ROOT / "dither_expiriments" / "runs_e10" / "_qdither_consolidated.json"))
    ap.add_argument("--adv-t4", type=float, default=0.0,
                    help="concrete Adv_T4 value (eps_PRG estimate); default 0 (negligible, Thm 7)")
    ap.add_argument("--out", default=str(_THIS / "qdither_bound_check.json"))
    args = ap.parse_args()
    if not Path(args.consolidated).exists():
        print(f"consolidated file not found: {args.consolidated}\n"
              f"run dither_experiments/qdither_consolidate.py first, or pass --consolidated")
        return
    rows = check(args.consolidated, adv_t4=args.adv_t4)
    if not rows:
        print("no per-sigma rows found in consolidated file"); return
    summ = summarize(rows, args.adv_t4)
    json.dump({"rows": rows, "summary": summ}, open(args.out, "w"), indent=2)

    print("=== Theorem Q numerical check (bound uses measured MI as proxy for I_sigma; proxy <= I_sigma,")
    print("    so sqrt(proxy/2) is a LOWER estimate of the true bound -> holding here holds a fortiori) ===\n")
    print(f"{'sigma':>5} {'Adv_T5':>8} {'sqrt(adjMI/2)':>13} {'sqrt(sigMI/2)':>13} "
          f"{'bound(min)':>11} {'holds':>6} {'margin':>8}")
    for r in rows:
        print(f"{r['sigma_lsb']:>5} {r['adv_t5_max']:>8.4f} {r['bound_adj']:>13.4f} {r['bound_sig']:>13.4f} "
              f"{r['bound_min_proxy']:>11.4f} {'OK' if r['holds'] else 'FAIL':>6} {r['margin']:>8.4f}")
    print(f"\nTheorem Q holds: {summ['n_hold']}/{summ['n']} sigma points  (Adv_T4={args.adv_t4})")
    print(f"bound monotone decreasing in sigma: {summ['bound_monotone_decreasing_in_sigma']}  "
          f"(sqrt(I_sigma/2) -> Adv_T4 as I_sigma -> 0)")
    print(f"bound at sigma=0: {summ['bound_at_sigma0']:.3f}  |  Adv_T5 at sigma=0: {summ['adv_t5_at_sigma0']:.3f}"
          f"  |  MARGIN: {summ['margin_at_sigma0']:.3f}")
    print(f"minimum margin over all sigma: {summ['min_margin']:.3f}")
    print("\n--- two-source reading ---")
    print("The bound is LOOSE at small sigma (I_0>0: telemetry is structured), yet Adv_T5 ~ baseline there.")
    print("That gap is NOT closed by dithering -- it is closed by COMBINATORIAL hardness (E10): the PPT T5")
    print("adversary cannot invert a 32,400-bit permutation to extract the arrangement information, at ANY")
    print("sigma. Dithering closes the remaining INFORMATION-THEORETIC gap (bound -> Adv_T4 as sigma grows),")
    print("bounding even an UNBOUNDED adversary. Two independent mechanisms; the margin is the combinatorial one.")
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()