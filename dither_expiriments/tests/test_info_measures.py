from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from smap_msl_dataset_api import Quantizer
from info_measures import (mi_from_joint, kl_categorical, adjacency_mi, dither_info_curve)

Q = Quantizer()

def _Hb(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def test_mi_known_bsc():
    rng = np.random.default_rng(0); N = 400_000; e = 0.1
    x = rng.integers(0, 2, N)
    flip = rng.random(N) < e
    y = x ^ flip.astype(np.int64)
    c = np.zeros((2, 2))
    for a in (0, 1):
        for bb in (0, 1):
            c[a, bb] = np.sum((x == a) & (y == bb))
    mi = mi_from_joint(c)
    analytic = np.log(2) - _Hb(e)
    assert abs(mi - analytic) < 0.01

def test_kl_known():
    p = [0.5, 0.3, 0.2]; q = [0.4, 0.4, 0.2]
    analytic = sum(pi * np.log(pi / qi) for pi, qi in zip(p, q))
    assert abs(kl_categorical(p, q) - analytic) < 1e-9
    assert kl_categorical(p, p) < 1e-12

def test_adjacency_markov_known():
    rng = np.random.default_rng(1); N = 400_000; s = 0.9
    seq = np.empty(N, np.int64); seq[0] = 0
    u = rng.random(N)
    for t in range(1, N):
        seq[t] = seq[t - 1] if u[t] < s else 1 - seq[t - 1]
    mi = adjacency_mi(seq, miller_madow=True)
    analytic = np.log(2) - _Hb(1 - s)
    assert abs(mi - analytic) < 0.01

def test_independent_floor_near_zero():
    rng = np.random.default_rng(2)
    seq = rng.integers(0, 2, 200_000)
    assert adjacency_mi(seq, miller_madow=True) < 5e-3

def test_dither_collapses_lsb_adjacency():
    T = 4000
    x = np.linspace(-0.5, 0.5, T)
    res = dither_info_curve(x, Q, [0.0, 0.5 * Q.delta_lsb], n_reps=4)
    lsb0 = res["no_dither"]["adjacency_mi_per_plane"][0]
    lsb_d = res["by_sigma"][1]["adjacency_mi_per_plane"][0]
    assert lsb0 > 0.05
    assert lsb_d < 0.25 * lsb0

def test_signal_mi_decreases_kl_increases():
    x = np.random.default_rng(3).uniform(-0.9, 0.9, size=6000)
    res = dither_info_curve(x, Q, [0.1 * Q.delta_lsb, 1.0 * Q.delta_lsb], n_reps=4)
    lo, hi = res["by_sigma"]
    assert hi["signal_mi_total"] < lo["signal_mi_total"]
    assert hi["level_kl_mean"] > lo["level_kl_mean"]

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")