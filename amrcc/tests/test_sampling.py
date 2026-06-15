from __future__ import annotations
import sys
import time
import itertools
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_AMRCC = _THIS.parent
_ROOT = _AMRCC.parent
for _p in [str(_ROOT), str(_AMRCC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sampling import knuth_select_S, selection_sample, clopper_pearson, have_scipy

def test_selection_sample_distinct_in_range():
    rng = np.random.default_rng(0)
    for method in ("choice", "knuth"):
        for _ in range(200):
            s = selection_sample(800, 37, rng, method=method)
            assert s.size == 37
            assert len(set(s.tolist())) == 37
            assert s.min() >= 0 and s.max() < 800
            assert np.all(np.diff(s) > 0)
    assert selection_sample(800, 0, rng).size == 0

def test_knuth_full_subset_uniformity_tiny():
    rng = np.random.default_rng(1)
    N, ksel, M = 5, 2, 60_000
    subsets = {frozenset(c): i for i, c in enumerate(itertools.combinations(range(N), ksel))}
    counts = np.zeros(len(subsets))
    for _ in range(M):
        counts[subsets[frozenset(knuth_select_S(N, ksel, rng).tolist())]] += 1
    exp = M / len(subsets)
    chi2 = ((counts - exp) ** 2 / exp).sum()
    assert chi2 < 27.88 

def test_choice_and_knuth_inclusion_marginals_match():
    rng = np.random.default_rng(2)
    N, ksel, M = 50, 7, 40_000
    p_theory = ksel / N
    for method in ("choice", "knuth"):
        inc = np.zeros(N)
        for _ in range(M):
            inc[selection_sample(N, ksel, rng, method=method)] += 1
        inc /= M
        assert np.abs(inc - p_theory).max() < 0.02

def test_clopper_pearson_edges_and_order():
    lo, hi = clopper_pearson(0, 100)
    assert lo == 0.0 and 0 < hi < 0.06
    lo, hi = clopper_pearson(100, 100)
    assert hi == 1.0 and 0.94 < lo < 1.0
    lo, hi = clopper_pearson(50, 100)
    assert lo < 0.5 < hi
    assert clopper_pearson(0, 0) == (0.0, 1.0)

def test_clopper_pearson_contains_point_and_known():
    if have_scipy():
        lo, hi = clopper_pearson(2, 20, alpha=0.05)
        assert abs(lo - 0.0123) < 2e-3 and abs(hi - 0.3170) < 2e-3
    for k, n in [(3, 50), (17, 17), (0, 9), (25, 100)]:
        lo, hi = clopper_pearson(k, n)
        assert lo <= k / n <= hi

def test_clopper_pearson_empirical_coverage():
    rng = np.random.default_rng(3)
    p, n, trials = 0.1, 200, 3000
    covered = 0
    for _ in range(trials):
        k = int(rng.binomial(n, p))
        lo, hi = clopper_pearson(k, n, alpha=0.05)
        covered += (lo <= p <= hi)
    assert covered / trials >= 0.95

if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items())
           if n.startswith("test_") and callable(v)]
    for fn in fns:
        t0 = time.time()
        fn()
        print(f"PASS  {fn.__name__:48s} {time.time() - t0:6.2f}s")
    print(f"\nall {len(fns)} sampling tests passed   (scipy={have_scipy()})")