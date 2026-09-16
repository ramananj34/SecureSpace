from __future__ import annotations
import numpy as np
from keyed_permutation import chacha20_keystream, fisher_yates as _fy_prototype

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
if _HAVE_NUMBA:
    @njit(cache=True)
    def _durstenfeld_swaps(j_vals, k):
        perm = np.arange(k)
        for idx in range(k - 1):
            i = k - 1 - idx
            j = j_vals[idx]
            tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp
        return perm

def fisher_yates_fast(keystream: bytes, k: int) -> np.ndarray:
    if not _HAVE_NUMBA:
        return _fy_prototype(keystream, k)
    r = np.frombuffer(keystream, dtype="<u8")
    if r.size < k - 1:
        raise ValueError(f"keystream too short: {r.size} u64 < {k - 1} needed")
    mods = np.arange(k, 1, -1, dtype=np.uint64)
    j_vals = (r[: k - 1] % mods).astype(np.int64)
    return _durstenfeld_swaps(j_vals, k)

def permutation_for_frame_fast(key: bytes, t: int, k: int, mission_offset: int = 0) -> np.ndarray:
    ks = chacha20_keystream(key, t, k * 8, mission_offset=mission_offset)
    return fisher_yates_fast(ks, k)

if __name__ == "__main__":
    import time, statistics
    key = b"\x01" * 32
    for k in (1000, 32400):
        ks = chacha20_keystream(key, 0, k * 8)
        assert np.array_equal(_fy_prototype(ks, k), fisher_yates_fast(ks, k)), f"MISMATCH k={k}"
        print(f"k={k:5d}: numba FY == prototype  OK")
    k = 32400; ks = chacha20_keystream(key, 0, k * 8)
    fisher_yates_fast(ks, k)
    def med(fn, n):
        xs = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
        return statistics.median(xs) * 1e3
    mp = med(lambda: _fy_prototype(ks, k), 15)
    mf = med(lambda: fisher_yates_fast(ks, k), 50)
    print(f"prototype FY: {mp:.2f} ms   numba FY: {mf:.3f} ms   speedup: {mp/mf:.0f}x   numba={_HAVE_NUMBA}")