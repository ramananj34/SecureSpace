from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import federated as F

def test_fedavg_is_mean():
    U = np.array([[1., 2.], [3., 4.], [5., 6.]])
    assert np.allclose(F.fedavg(U), [3., 4.])
    # weighted
    assert np.allclose(F.fedavg(U, weights=[1, 0, 0]), [1., 2.])

def test_coord_median():
    U = np.array([[1., 2.], [3., 4.], [100., -100.]])
    assert np.allclose(F.coord_median(U), [3., 2.])

def test_krum_selects_honest_scattered():
    rng = np.random.default_rng(0)
    honest = [rng.normal(0, 0.1, 5) for _ in range(7)]
    byz = [rng.normal(50, 1, 5) for _ in range(3)]
    U = np.array(honest + byz)
    sel = F.krum(U, f=3, return_score=True)[1]
    assert sel < 7

def test_krum_selects_honest_colluding():
    rng = np.random.default_rng(1)
    honest = [rng.normal(0, 0.1, 5) for _ in range(7)]
    byz_pt = np.full(5, 30.0)
    byz = [byz_pt + rng.normal(0, 0.01, 5) for _ in range(3)]
    U = np.array(honest + byz)
    assert F.krum(U, f=3, return_score=True)[1] < 7

def test_krum_requires_n_ge_2f_plus_3():
    U = np.random.default_rng(2).normal(size=(4, 3))
    try:
        F.krum(U, f=1)
        assert False, "should have raised"
    except ValueError:
        pass

def test_aggregate_dispatch():
    U = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]])
    assert np.allclose(F.aggregate("fedavg", U), U.mean(0))
    assert np.allclose(F.aggregate("coord_median", U), np.median(U, 0))
    r = F.aggregate("krum", U, f=1)
    assert r.shape == (2,)

def test_temporal_partitions_cover_stream():
    parts = F.temporal_partitions(1000, 40, window=250)
    assert len(parts) == 40
    assert parts[0][0] == 0 and parts[-1][1] == 1000
    for i in range(len(parts) - 1):
        assert parts[i][1] == parts[i + 1][0]

def test_flatten_unflatten_roundtrip():
    try:
        import torch
    except Exception:
        print("  (torch absent: flatten roundtrip skipped)"); return
    sd = {"b.weight": torch.arange(6.).reshape(2, 3), "a.bias": torch.arange(4.)}
    vec, shapes = F.flatten_params(sd)
    assert vec.shape == (10,)
    assert np.allclose(vec[:4], [0, 1, 2, 3]) and np.allclose(vec[4:], [0, 1, 2, 3, 4, 5])
    rec = F.unflatten_params(vec, shapes, sd)
    for k in sd:
        assert torch.allclose(rec[k], sd[k])

def test_byzantine_robustness_median_vs_fedavg():
    rng = np.random.default_rng(3)
    honest = rng.normal(1.0, 0.05, (6, 4))
    byz = np.full((1, 4), 1000.0)
    U = np.vstack([honest, byz])
    fa = F.fedavg(U); cm = F.coord_median(U); kr = F.krum(U, f=1)
    assert np.abs(fa - 1.0).max() > 100
    assert np.abs(cm - 1.0).max() < 0.2
    assert np.abs(kr - 1.0).max() < 0.2

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")