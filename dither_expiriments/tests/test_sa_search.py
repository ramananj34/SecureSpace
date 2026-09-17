from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sa_search import SAConfig, anneal, anneal_multi

def test_recover_permuted_smooth_signal_small():
    rng = np.random.default_rng(0)
    k = 60
    x = np.cumsum(rng.normal(0, 1, k)); x = (x - x.mean())
    perm_true = rng.permutation(k)
    obs = x[perm_true]

    def fitness_full(perm):
        rec = obs[perm]
        d = np.diff(rec)
        return float(-np.sum(d * d))

    inv_true = np.empty(k, np.int64); inv_true[perm_true] = np.arange(k)
    f_truth = fitness_full(inv_true)
    cfg = SAConfig(iters=40000, restarts=4, T0=2.0, T1=1e-3, seed=1, init="random", max_seg=16)
    res = anneal_multi(k, fitness_full, cfg)
    assert res.fitness >= f_truth * 1.05
    rand_f = np.mean([fitness_full(np.random.default_rng([9, s]).permutation(k)) for s in range(20)])
    assert res.fitness > rand_f

def test_swap_delta_matches_full():
    rng = np.random.default_rng(2)
    k = 40
    w = rng.normal(0, 1, k)
    v = rng.normal(0, 1, k)

    def fitness_full(perm):
        return float(np.sum(w * v[perm]))

    def delta_swap(perm, i, j):
        return float(w[i] * (v[perm[j]] - v[perm[i]]) + w[j] * (v[perm[i]] - v[perm[j]]))

    perm = rng.permutation(k)
    for _ in range(200):
        i, j = rng.integers(0, k), rng.integers(0, k)
        if i == j:
            continue
        f0 = fitness_full(perm)
        d = delta_swap(perm, i, j)
        cand = perm.copy(); cand[i], cand[j] = cand[j], cand[i]
        assert abs((fitness_full(cand) - f0) - d) < 1e-9

def test_incremental_run_matches_full_fitness():
    rng = np.random.default_rng(3)
    k = 50
    w = rng.normal(0, 1, k); v = rng.normal(0, 1, k)
    def fitness_full(perm): return float(np.sum(w * v[perm]))
    def delta_swap(perm, i, j):
        return float(w[i]*(v[perm[j]]-v[perm[i]]) + w[j]*(v[perm[i]]-v[perm[j]]))
    cfg = SAConfig(iters=5000, restarts=1, seed=4, seg_move_frac=0.0, init="random")
    res = anneal(k, fitness_full, cfg, delta_swap=delta_swap,
                 init_perm=np.random.default_rng(5).permutation(k))
    assert abs(res.fitness - fitness_full(res.pi_hat)) < 1e-6

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")