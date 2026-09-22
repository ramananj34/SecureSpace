from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import energy_analysis as EA

K = 32400

def test_reduction_factor_reproduces_appendixA():
    for d_adv, expect in [(100, 243), (50, 486), (25, 972)]:
        f = EA.reduction_factor(K, d_adv, delta_msb=1.0, b=8)
        assert abs(f - expect) < 3, (d_adv, f, expect)

def test_worst_case_ratio_reproduces_appendixA():
    for d_adv, expect in [(100, 0.375), (50, 0.75), (25, 1.5)]:
        r = EA.worst_case_cross_main_ratio(K, d_adv, w=100, b=8)
        assert abs(r - expect) < 0.02, (d_adv, r, expect)

def test_average_ratio_tiny():
    r = EA.average_cross_main_ratio(K, 100)
    assert 0 < r < 0.01
    assert abs(r - 99/32300) < 1e-9

def test_worst_moves_unfavourably_as_dadv_shrinks():
    rf100 = EA.reduction_factor(K, 100); rf25 = EA.reduction_factor(K, 25)
    wr100 = EA.worst_case_cross_main_ratio(K, 100, 100); wr25 = EA.worst_case_cross_main_ratio(K, 25, 100)
    assert rf25 > rf100
    assert wr25 > wr100

def test_regime_table():
    t = EA.energy_regime_table(K, weights=[100], d_adv_values=[100, 50, 25])
    assert abs(t["c_quant"] - 1.3333) < 0.001
    rf = {r["d_adv"]: r["reduction_factor"] for r in t["rows"]}
    assert abs(rf[100] - 243) < 3 and abs(rf[50] - 486) < 3 and abs(rf[25] - 972) < 3

def test_cross_term_distribution_mean_is_trace():
    d, b = 8, 3; k = d * b; dlsb = 0.5
    weights = (2.0 ** np.arange(b)) * dlsb
    D = np.zeros((d, k))
    for j in range(d): D[j, j*b:(j+1)*b] = weights
    rng = np.random.default_rng(0)
    Q_, _ = np.linalg.qr(rng.normal(size=(d, d))); Vb = Q_[:, :2]; Pi_V = Vb @ Vb.T
    dist = EA.cross_term_distribution(Pi_V, D, k, n_xq=6000, seed=1)
    assert dist["mean_matches_trace_rel"] < 0.05
    assert dist["std"] > 0 and dist["max"] > dist["min"]
    assert dist["p95"] > dist["mean"] > dist["p05"]

def test_worst_case_bound_on_cross():
    d, b = 5, 8; k = d * b; dlsb = 2.0**-7
    weights = (2.0 ** np.arange(b)) * dlsb
    D = np.zeros((d, k))
    for j in range(d): D[j, j*b:(j+1)*b] = weights
    wb = EA.worst_case_bound_on_cross(D, k, b=8)
    assert abs(wb["delta_max"] - 255/128) < 1e-6
    assert abs(wb["upper_bound_Ds_sq"] - 5 * (255/128)**2) < 1e-6

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")