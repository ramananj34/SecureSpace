from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "fl_core")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import theorem5prime_energy as T5P

def test_median_robustness_bounded_by_honest_spread():
    res = T5P.verify_median_robustness(d=12, M=9, B=2, alphas=(1.0, 10.0, 1000.0))
    assert res["all_in_range"]
    assert res["bounded_regardless_of_alpha"]

def test_trap_d_aligned_defeats_reduction():
    res = T5P.verify_trap_d(d=12, M=9, B=2, d_adv_prime=1, alpha=2.0)
    assert res["aligned_defeats_reduction"]
    assert res["aligned_proj_over_full"] > res["random_proj_over_full"]

def test_trap_d_random_gets_reduction():
    res = T5P.verify_trap_d(d=20, M=11, B=3, d_adv_prime=1, alpha=2.0)
    assert res["random_gets_reduction"]

def test_theorem5prime_bound_is_full_l2():
    b = T5P.theorem5prime_bound(honest_spread_per_coord=0.05, d_grad=100, B=2, M=9)
    assert abs(b["bound_full_l2"] - 100 * 0.05 ** 2) < 1e-9
    assert b["requires_B_lt_half_M"] is True
    assert "d_adv" in b["note"]

def test_robustness_requires_B_lt_half():
    try:
        T5P.verify_median_robustness(d=10, M=8, B=4)
        assert False, "should require B < M/2"
    except AssertionError:
        pass

def test_projected_energy_matches_manual():
    rng = np.random.default_rng(0)
    d = 8; u = rng.normal(size=d); u /= np.linalg.norm(u); Pi = np.outer(u, u)
    dev = rng.normal(size=d)
    assert abs(T5P.projected_energy(Pi, dev) - (u @ dev) ** 2) < 1e-9

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")