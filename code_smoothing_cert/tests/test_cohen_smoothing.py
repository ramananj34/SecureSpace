from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cohen_smoothing as CS

def test_binary_radius_form_agreement():
    for pA in [0.6, 0.75, 0.9, 0.99]:
        r_bin = CS._phi_inv(pA)
        r_gap = 0.5 * (CS._phi_inv(pA) - CS._phi_inv(1 - pA))
        assert abs(r_bin - r_gap) < 1e-9

def test_cp_lower_is_lower_bound():
    assert CS.cp_lower(900, 1000, 0.001) < 0.9
    assert CS.cp_lower(9000, 10000, 0.001) > CS.cp_lower(900, 1000, 0.001)
    assert CS.cp_lower(0, 1000, 0.001) == 0.0
    assert 0 < CS.cp_lower(1000, 1000, 0.001) < 1.0

def test_linear_classifier_recovers_margin():
    d = 5; w = np.zeros(d); w[0] = 1.0
    def add_noise_and_detect(x, sigma, rng):
        eta = rng.normal(0, sigma, d)
        return bool((x + eta) @ w > 0)
    x = np.zeros(d); x[0] = 0.4
    res = CS.certify_point(x, add_noise_and_detect, sigma=0.5, n0=1000, n=100000, alpha=0.001, rng=np.random.default_rng(0))
    assert res["top_is_detected"] and not res["abstain"]
    assert 0.30 < res["radius"] < 0.40

def test_abstain_on_boundary():
    d = 5; w = np.zeros(d); w[0] = 1.0
    def f(x, sigma, rng): return bool((x + rng.normal(0, sigma, d)) @ w > 0)
    x0 = np.zeros(d)
    res = CS.certify_point(x0, f, sigma=0.5, n0=1000, n=20000, alpha=0.001, rng=np.random.default_rng(1))
    assert res["radius"] == 0.0 and res["abstain"]

def test_radius_grows_with_margin():
    d = 5; w = np.zeros(d); w[0] = 1.0
    def f(x, sigma, rng): return bool((x + rng.normal(0, sigma, d)) @ w > 0)
    r_near = CS.certify_point(np.array([0.3,0,0,0,0.0]), f, 0.5, 1000, 50000, 0.001, np.random.default_rng(2))["radius"]
    r_far = CS.certify_point(np.array([0.9,0,0,0,0.0]), f, 0.5, 1000, 50000, 0.001, np.random.default_rng(3))["radius"]
    assert r_far > r_near

def test_detector_not_robust_abstains():
    def f(x, sigma, rng):
        return bool(rng.random() < 0.1)
    res = CS.certify_point(np.zeros(3), f, 0.5, n0=500, n=2000, alpha=0.001, rng=np.random.default_rng(4))
    assert res["abstain"] and res["radius"] == 0.0
    assert res["top_is_detected"] is False 

def test_certify_over_sigmas_picks_best():
    d = 5; w = np.zeros(d); w[0] = 1.0
    def f(x, sigma, rng): return bool((x + rng.normal(0, sigma, d)) @ w > 0)
    x = np.zeros(d); x[0] = 0.5
    res = CS.certify_over_sigmas(x, f, sigmas=[0.1, 0.5, 1.0], n0=500, n=20000, alpha=0.001, seed=0)
    assert res["best_radius"] == max(r["radius"] for r in res["per_sigma"])
    assert res["best_radius"] > 0

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")