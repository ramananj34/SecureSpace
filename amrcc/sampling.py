from __future__ import annotations
import numpy as np

try:
    from scipy.stats import beta as _beta
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def knuth_select_S(n_records: int, n_select: int, rng) -> np.ndarray:
    if not (0 <= n_select <= n_records):
        raise ValueError("require 0 <= n_select <= n_records")
    out = np.empty(n_select, dtype=np.int64)
    t = 0
    m = 0
    while m < n_select:
        if (n_records - t) * rng.random() >= (n_select - m):
            t += 1
        else:
            out[m] = t
            t += 1
            m += 1
    return out

def selection_sample(n_bits: int, w: int, rng, method: str = "choice") -> np.ndarray:
    if not (0 <= w <= n_bits):
        raise ValueError(f"require 0 <= w <= n_bits, got w={w}, n_bits={n_bits}")
    if w == 0:
        return np.empty(0, dtype=np.int64)
    if method == "knuth":
        return knuth_select_S(n_bits, w, rng)
    return np.sort(rng.choice(n_bits, size=w, replace=False)).astype(np.int64)

def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (0.0, 1.0)
    if not _HAVE_SCIPY:
        p = k / n
        se = (p * (1.0 - p) / n) ** 0.5
        z = 1.959963984540054
        return (max(0.0, p - z * se), min(1.0, p + z * se))
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)

def have_scipy() -> bool:
    return _HAVE_SCIPY