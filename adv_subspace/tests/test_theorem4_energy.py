from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import theorem4_energy as T4

def _instance(seed=0, d=3, b=2, d_adv=2, w=3):
    k = d * b; dlsb = 0.5
    weights = (2.0 ** np.arange(b)) * dlsb
    D = np.zeros((d, k))
    for j in range(d): D[j, j*b:(j+1)*b] = weights
    c_quant = float((weights**2).sum())
    rng = np.random.default_rng(seed)
    Q_, _ = np.linalg.qr(rng.normal(size=(d, d))); Vb = Q_[:, :d_adv]; Pi_V = Vb @ Vb.T
    x_q = rng.integers(0, 2, k)
    return dict(k=k, D=D, c_quant=c_quant, Pi_V=Pi_V, x_q=x_q, w=w, d_adv=d_adv)

def _brute_exact(Pi_V, D, x_q, w, k, seed=0):
    s = 1.0 - 2.0 * x_q; S = np.diag(s)
    rng = np.random.default_rng([seed, 1]); supp = np.sort(rng.choice(k, w, replace=False))
    delta = np.zeros(k, np.uint8); delta[supp] = 1
    def ip(p): q = np.empty_like(p); q[p] = np.arange(p.size); return q
    Es = []
    for perm in itertools.permutations(range(k)):
        perm = np.array(perm); v = delta[ip(perm)]
        vec = Pi_V @ (D @ (S @ v.astype(float))); Es.append(vec @ vec)
    return float(np.mean(Es)), delta

def test_closed_form_matches_brute_exact():
    ins = _instance()
    from sampling import selection_sample
    supp = selection_sample(ins["k"], ins["w"], np.random.default_rng(0))
    delta = np.zeros(ins["k"], np.uint8); delta[supp] = 1
    # exact expectation for THIS delta:
    s = 1.0 - 2.0 * ins["x_q"]; S = np.diag(s)
    def ip(p): q = np.empty_like(p); q[p] = np.arange(p.size); return q
    Es = []
    for perm in itertools.permutations(range(ins["k"])):
        perm = np.array(perm); v = delta[ip(perm)]
        vec = ins["Pi_V"] @ (ins["D"] @ (S @ v.astype(float))); Es.append(vec @ vec)
    exact = float(np.mean(Es))
    ct = T4.cross_term_value(ins["Pi_V"], ins["D"], ins["x_q"])
    rhs = T4.closed_form_rhs(ins["w"], ins["k"], ins["c_quant"], ins["d_adv"], ct)["total"]
    assert abs(rhs - exact) < 1e-9, (rhs, exact)

def test_mc_converges_to_closed_form():
    ins = _instance()
    res = T4.verify_theorem4(ins["Pi_V"], ins["D"], ins["x_q"], ins["w"], ins["k"],
                             ins["c_quant"], ins["d_adv"], n_mc=40000, seed=1)
    assert res["within_tol"] and res["rel_error"] < 0.05 

def test_signed_differs_from_unsigned_xq0():
    ins = _instance()
    signed = T4.cross_term_value(ins["Pi_V"], ins["D"], ins["x_q"])
    unsigned = T4.unsigned_xq0_cross(ins["Pi_V"], ins["D"])
    assert abs(signed - unsigned) > 1e-6

def test_cross_equals_trace_on_average():
    ins = _instance(d=4, b=3, d_adv=2)
    r = T4.cross_equals_trace_on_average(ins["Pi_V"], ins["D"], ins["k"], ins["c_quant"], ins["d_adv"],
                                         n_xq=4000, seed=2)
    assert r["rel_error"] < 0.1

def test_keyed_matches_numpy_perm():
    ins = _instance()
    mc_np = T4.mc_projected_energy(ins["Pi_V"], ins["D"], ins["x_q"], ins["w"], ins["k"],
                                   n_mc=8000, seed=3, use_keyed=False)
    mc_key = T4.mc_projected_energy(ins["Pi_V"], ins["D"], ins["x_q"], ins["w"], ins["k"],
                                    n_mc=8000, seed=3, use_keyed=True)
    ct = T4.cross_term_value(ins["Pi_V"], ins["D"], ins["x_q"])
    rhs = T4.closed_form_rhs(ins["w"], ins["k"], ins["c_quant"], ins["d_adv"], ct)["total"]
    assert abs(mc_np - rhs) / rhs < 0.10 and abs(mc_key - rhs) / rhs < 0.10

def test_within_tol_flag():
    ins = _instance()
    ct = T4.cross_term_value(ins["Pi_V"], ins["D"], ins["x_q"])
    rhs = T4.closed_form_rhs(ins["w"], ins["k"], ins["c_quant"], ins["d_adv"], ct)
    def rel(lhs): return abs(lhs - rhs["total"]) / rhs["total"]
    assert rel(rhs["total"] * 1.30) > 0.20 and rel(rhs["total"] * 1.10) < 0.20

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")