from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "smap_msl_data"), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from smap_msl_dataset_api import Quantizer
from sa_search import SAConfig, anneal_multi
from analytic_attacker import analytic_attack
from recovery_metrics import m1_bit_accuracy, m1_random_baseline, observe, spearman

Q = Quantizer()

def test_coord_order_recovery_when_grouping_revealed():
    rng = np.random.default_rng(0)
    b = 8
    n_coords = 24
    x = np.clip(np.cumsum(rng.normal(0, 0.05, n_coords)), -0.9, 0.9)
    lv = Q.quantize(x)
    xs = np.asarray(Q.dequantize(lv), np.float64)
    order = rng.permutation(n_coords)
    xs_obs = xs[order]

    def fit(perm):
        rec = xs_obs[perm]
        d = np.diff(rec)
        return float(-np.sum(d * d))

    cfg = SAConfig(iters=40000, restarts=6, T0=1.0, T1=1e-3, seed=1, init="random", max_seg=8)
    res = anneal_multi(n_coords, fit, cfg)
    rec_order = xs_obs[res.pi_hat]
    truth_order = xs
    f_truth = fit(np.argsort(order))
    assert res.fitness >= f_truth * 1.02
    assert abs(spearman(rec_order, truth_order)) > 0.6

def test_bit_scramble_near_baseline():
    rng = np.random.default_rng(2)
    b = 8; header = 16; n_coords = 20; slot = n_coords * b; k = header + slot
    x = np.clip(np.cumsum(rng.normal(0, 0.05, n_coords)), -0.9, 0.9)
    bits = np.asarray(Q.levels_to_bits(Q.quantize(x)), np.uint8).reshape(-1)
    m_in = np.concatenate([rng.integers(0, 2, header).astype(np.uint8), bits])
    perm_true = rng.permutation(k)
    m_obs = observe(m_in, perm_true)
    cfg = SAConfig(iters=20000, restarts=2, seed=3, init="random", max_seg=16)
    res = analytic_attack(m_obs, k, fitness="bit_run", sa_config=cfg)
    acc = m1_bit_accuracy(m_in, m_obs, res.pi_hat)
    base = m1_random_baseline(k, int(m_in.sum()))
    assert acc < base + 0.10
    assert acc < 0.70

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")