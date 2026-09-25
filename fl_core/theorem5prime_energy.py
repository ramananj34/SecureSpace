from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "fl_core")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from federated import coord_median


def coord_median_deviation(honest, byz, g_star):
    allg = np.vstack([np.asarray(honest, np.float64), np.asarray(byz, np.float64)])
    med = coord_median(allg)
    return med - np.asarray(g_star, np.float64), med


def projected_energy(Pi_Vtheta, dev):
    p = np.asarray(Pi_Vtheta, np.float64) @ np.asarray(dev, np.float64)
    return float(p @ p)

def verify_median_robustness(d, M, B, alphas=(1.0, 10.0, 1000.0), n_mc=200, seed=0):
    assert B < M / 2, "coord-median robustness requires B < M/2"
    rng = np.random.default_rng(seed)
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    rows = []
    for alpha in alphas:
        in_range_frac, dev_norms = [], []
        for _ in range(n_mc):
            g_star = rng.normal(0, 0.1, d)
            honest = np.array([g_star + rng.normal(0, 0.05, d) for _ in range(M - B)])
            byz = np.array([g_star + alpha * u for _ in range(B)])
            dev, med = coord_median_deviation(honest, byz, g_star)
            h_lo, h_hi = honest.min(0), honest.max(0)
            in_range_frac.append(float(np.all((med >= h_lo - 1e-9) & (med <= h_hi + 1e-9))))
            dev_norms.append(float(np.linalg.norm(dev)))
        rows.append({"alpha": alpha, "median_in_honest_range_frac": float(np.mean(in_range_frac)),
                     "dev_norm_mean": float(np.mean(dev_norms))})
    dev_norms = [r["dev_norm_mean"] for r in rows]
    bounded = max(dev_norms) < 3.0 * min(dev_norms) + 0.1
    return {"rows": rows, "bounded_regardless_of_alpha": bool(bounded),
            "all_in_range": bool(all(r["median_in_honest_range_frac"] > 0.9 for r in rows))}


def verify_trap_d(d, M, B, d_adv_prime=1, alpha=2.0, n_mc=4000, seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Vb = U[:, :d_adv_prime]; Pi = Vb @ Vb.T
    u = Vb[:, 0]
    def run(aligned):
        pj, fl = [], []
        for _ in range(n_mc):
            g_star = rng.normal(0, 0.1, d)
            honest = [g_star + rng.normal(0, 0.05, d) for _ in range(M - B)]
            if aligned:
                byz = [g_star + alpha * u for _ in range(B)]
            else:
                byz = [g_star + alpha * rng.normal(0, 1, d) for _ in range(B)]
            dev, _ = coord_median_deviation(honest, byz, g_star)
            pj.append(projected_energy(Pi, dev)); fl.append(float(dev @ dev))
        return np.mean(pj) / np.mean(fl)
    r_aligned = run(True); r_random = run(False)
    reduction = d_adv_prime / d
    return {"d_adv_prime_over_d": float(reduction),
            "aligned_proj_over_full": float(r_aligned),
            "random_proj_over_full": float(r_random),
            "aligned_defeats_reduction": bool(r_aligned > 2.0 * reduction),
            "random_gets_reduction": bool(abs(r_random - reduction) < 0.5 * reduction + 0.02)}


def theorem5prime_bound(honest_spread_per_coord, d_grad, B, M):
    s = float(np.max(honest_spread_per_coord)) if np.ndim(honest_spread_per_coord) else float(honest_spread_per_coord)
    return {"bound_full_l2": float(d_grad * s ** 2),
            "note": "full-L2 (honest-spread) bound; d_adv' does NOT reduce it (trap D). B<M/2 required.",
            "requires_B_lt_half_M": bool(B < M / 2)}