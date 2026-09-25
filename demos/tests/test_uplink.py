from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "fl_core"), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import uplink as U
from grad_channel import GradientQuantizer, isl_key

def _model(d=2000, seed=0):
    return np.random.default_rng(seed).normal(0, 0.1, d)

def test_clean_uplink_roundtrips():
    theta = _model()
    gq = GradientQuantizer(b=8); key = isl_key(0, 0)
    recv = U.uplink_clean(theta, gq, key)
    dlsb = (theta.max() - theta.min()) / 256
    assert np.abs(recv - theta).max() <= dlsb / 2 + 1e-9

def test_targeted_damage_dilution_is_large():
    theta = _model(d=2000)
    sens = np.random.default_rng(1).normal(0, 1, 2000)
    gq = GradientQuantizer(b=8); key = isl_key(0, 0)
    res = U.uplink_experiment(theta, gq, key, w=200, sensitivity=sens, seed=0)
    td = res["targeted_damage"]
    assert td["undefended"] > 20 * td["amrcc"]
    assert td["amrcc"] < td["undefended"]

def test_l2_integrity_honest_modest_dilution():
    theta = _model(d=2000)
    sens = np.ones(2000)
    gq = GradientQuantizer(b=8); key = isl_key(0, 0)
    res = U.uplink_experiment(theta, gq, key, w=200, sensitivity=sens, seed=0)
    l2 = res["l2"]
    assert l2["clean"] < l2["amrcc"] < l2["undefended"]
    assert l2["undefended"] > l2["amrcc"]

def test_amrcc_corruption_is_random_bounded():
    theta = _model(d=1000)
    gq = GradientQuantizer(b=8); key = isl_key(0, 0)
    clean = U.uplink_clean(theta, gq, key)
    amrcc = U.uplink_attacked(theta, gq, key, w=100, perm_on=True, seed=1)
    flip = amrcc - clean
    assert np.count_nonzero(np.abs(flip) > 1e-12) <= 100 + 5
    assert np.linalg.norm(flip) < 100.0

def test_undefended_hits_sensitive_coords():
    d = 500
    theta = _model(d=d)
    sens = np.zeros(d); sens[:50] = 10.0
    gq = GradientQuantizer(b=8); key = isl_key(0, 0)
    clean = U.uplink_clean(theta, gq, key)
    undef = U.uplink_attacked(theta, gq, key, w=50, sensitivity=sens, perm_on=False, seed=0)
    corr = undef - clean
    energy_sensitive = np.sum(corr[:50] ** 2)
    energy_rest = np.sum(corr[50:] ** 2)
    assert energy_sensitive > 10 * (energy_rest + 1e-12)

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")