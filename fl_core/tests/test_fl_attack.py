from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "fl_core"), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import fl_attack as A
import federated as FED
from grad_channel import GradientQuantizer

def _setup(M=10, B=3, d=500, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 0.02, d)
    honest = [g + rng.normal(0, 0.005, d) for _ in range(M)]
    return g, honest

def _align(u, g):
    return float(u @ g / (np.linalg.norm(u) * np.linalg.norm(g) + 1e-12))

def test_byzantine_mask():
    m = A.byzantine_mask(10, 3, seed=0)
    assert m.sum() == 3 and m.shape == (10,)
    assert A.byzantine_mask(10, 0, seed=0).sum() == 0

def test_regime1_catastrophic():
    g, honest = _setup()
    mask = A.byzantine_mask(10, 3, seed=0)
    U = A.apply_attack(honest, "fedavg", mask, base_seed=0, unbounded_scale=50.0)
    u = FED.fedavg(U)
    assert _align(u, g) < 0.0
    assert np.linalg.norm(u) > 5.0

def test_regime2_robust():
    g, honest = _setup()
    mask = A.byzantine_mask(10, 3, seed=0)
    U = A.apply_attack(honest, "median", mask, base_seed=0, unbounded_scale=50.0)
    u = FED.coord_median(U)
    assert _align(u, g) > 0.5

def test_regime3b_robust():
    g, honest = _setup()
    mask = A.byzantine_mask(10, 3, seed=0)
    gq = GradientQuantizer(b=8)
    U = A.apply_attack(honest, "median_perm", mask, gq=gq, base_seed=0, scatter_w=100)
    u = FED.coord_median(U)
    assert _align(u, g) > 0.5

def test_regime3a_degraded_not_catastrophic():
    g, honest = _setup()
    mask = A.byzantine_mask(10, 3, seed=0)
    gq = GradientQuantizer(b=8)
    U1 = A.apply_attack(honest, "fedavg", mask, base_seed=0, unbounded_scale=50.0)
    U3a = A.apply_attack(honest, "fedavg_perm", mask, gq=gq, base_seed=0, scatter_w=100)
    u1, u3a = FED.fedavg(U1), FED.fedavg(U3a)
    assert np.linalg.norm(u3a) < 0.2 * np.linalg.norm(u1)
    assert _align(u3a, g) > _align(u1, g)
    assert _align(u1, g) < 0.0
    assert np.linalg.norm(u3a) < 5.0

def test_scattered_weight_preserved_and_bounded():
    g, honest = _setup()
    gq = GradientQuantizer(b=8)
    key = A.isl_key(0, 0)
    bits, rng_tuple = gq.encode(honest[0])
    clean_decode = gq.decode(bits, rng_tuple)
    g_poisoned = A.bounded_scattered_gradient(honest[0], gq, key, w=100, t=0, seed=1)
    flip_effect = g_poisoned - clean_decode
    assert np.linalg.norm(flip_effect) < 100.0
    assert np.count_nonzero(np.abs(flip_effect) > 1e-12) <= 100

def test_regime_aggregator_map():
    assert A.regime_aggregator("fedavg") == ("fedavg", False)
    assert A.regime_aggregator("median_perm") == ("coord_median", True)
    assert A.regime_aggregator("krum_perm") == ("krum", True)

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")