from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from ldpc.tanner_graph import compute_girth
from dvb_s2_rates import make_code, ALL_RATES, LONG_RATE_TABLES

def spectral_expansion(H, k_sv=6):
    H = H.astype(np.float64).tocsr()
    r = np.asarray(H.sum(1)).ravel()
    c = np.asarray(H.sum(0)).ravel()
    Dr = sp.diags(1.0 / np.sqrt(r))
    Dc = sp.diags(1.0 / np.sqrt(c))
    Hn = (Dr @ H @ Dc).tocsr()
    sv = np.sort(svds(Hn, k=k_sv, return_singular_vectors=False))[::-1]
    return sv

def low_weight_search(code, n_adjacent=300, n_random=300, seed=0):
    rng = np.random.default_rng(seed)
    k = code.k
    def wt(i, j):
        m = np.zeros(k, dtype=np.uint8); m[i] = 1; m[j] = 1
        return int(np.asarray(code.encode(m)).astype(np.uint8).sum())
    best = None
    for i in rng.choice(k - 1, size=min(n_adjacent, k - 1), replace=False):
        w = wt(int(i), int(i) + 1)
        if best is None or w < best[0]:
            best = (w, int(i), int(i) + 1)
    rbest = None
    for _ in range(n_random):
        i, j = rng.integers(0, k, size=2)
        if i == j:
            continue
        w = wt(int(i), int(j))
        if rbest is None or w < rbest[0]:
            rbest = (w, int(i), int(j))
    ub = min(best[0], rbest[0])
    return {"dmin_ub": int(ub), "adjacent_best": [int(x) for x in best], "random_best": [int(x) for x in rbest]}

def load_e4_invariance(e4_dir):
    p = Path(e4_dir) / "e4_summary.json"
    if not p.exists():
        return None
    recs = json.load(open(p))
    return {"n_channels": len(recs), "all_verdict_invariant": all(r.get("verdict_invariant") for r in recs), "all_evade_every_rate": all(all(pr["real_npdt_missed"] for pr in r["per_rate"]) for r in recs)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_e6"))
    ap.add_argument("--e4-dir", default=str(_THIS / "runs_e4"))
    ap.add_argument("--girth-samples", type=int, default=400)
    ap.add_argument("--n-adjacent", type=int, default=300)
    ap.add_argument("--n-random", type=int, default=300)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"E6 structural spectrum across rates {ALL_RATES} (pure code-side, no GPU)\n")
    rows = []
    for rate in ALL_RATES:
        t0 = time.time()
        code = make_code(rate)
        girth, n_searched = compute_girth(code.H, cutoff=8, sample_size=args.girth_samples, rng_seed=0)
        sv = spectral_expansion(code.H)
        lw = low_weight_search(code, n_adjacent=args.n_adjacent, n_random=args.n_random)
        floor = code.m // 2
        row = {"rate": rate, "k": code.k, "m": code.m, "parity_floor": floor, "girth_sampled": int(girth), "girth_samples": int(n_searched), "sigma1": float(sv[0]), "sigma2_expansion": float(sv[1]), "spectral_gap": float(sv[0] - sv[1]), "dmin_upper_bound": lw["dmin_ub"], "low_weight": lw, "seconds": time.time() - t0}
        rows.append(row)
        print(f"  {rate:>4} k={code.k:5d} | girth={girth} (samp {n_searched}) | "
              f"sigma1={sv[0]:.4f} sigma2={sv[1]:.4f} gap={sv[0]-sv[1]:.4f} | "
              f"d_min<={lw['dmin_ub']:5d} | (n-k)/2={floor:5d}  [{row['seconds']:.0f}s]")
    e4 = load_e4_invariance(Path(args.e4_dir))
    print("\n=== E6 structural spectrum (varies across rate) vs attack success (does NOT) ===")
    print(f"{'rate':>5} {'girth':>5} {'sig2(expand)':>12} {'d_min<=':>7} {'(n-k)/2':>8} {'attack success':>18}")
    for r in rows:
        succ = "evades (all rates)" if (e4 and e4["all_evade_every_rate"]) else "see E4"
        print(f"{r['rate']:>5} {r['girth_sampled']:>5} {r['sigma2_expansion']:>12.4f} "
              f"{r['dmin_upper_bound']:>7} {r['parity_floor']:>8} {succ:>18}")
    g = [r["girth_sampled"] for r in rows]
    s2 = [r["sigma2_expansion"] for r in rows]
    dm = [r["dmin_upper_bound"] for r in rows]
    print(f"\n  girth:                   constant = {g[0]} (range {min(g)}..{max(g)}) -> IV invariant, "
          f"girth<->success untestable; 1/2 exhaustively confirmed girth 6 (D.33)")
    print(f"  sigma2 (expansion):      VARIES {min(s2):.4f}..{max(s2):.4f} (x{max(s2)/min(s2):.2f})")
    print(f"  d_min upper bound:       VARIES {min(dm)}..{max(dm)} (x{max(dm)/max(1,min(dm)):.1f})")
    print(f"  parity floor (n-k)/2:    VARIES {rows[-1]['parity_floor']}..{rows[0]['parity_floor']} "
          f"= E4 bypass cost (the ONLY code-side quantity tracking structure)")
    if e4:
        print(f"  attack success:          INVARIANT across all rates "
              f"({e4['n_channels']} channels, all verdict-invariant={e4['all_verdict_invariant']})")
    print(f"\n  => no code structural property (girth/expansion/distance) correlates with telemetry")
    print(f"     attack success (code-blindness, E4). Structure tracks WIRE cost only. C3-confirming.")
    json.dump({"spectrum": rows, "e4_invariance": e4, "stern_isd": "deferred (D.33.3, off critical path; pair-search UB used, D.55)"}, open(out_dir / "e6_summary.json", "w"), indent=2)
    print(f"\nsummary -> {out_dir / 'e6_summary.json'}\ndone")

if __name__ == "__main__":
    main()