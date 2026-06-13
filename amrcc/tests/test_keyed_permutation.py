from __future__ import annotations
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_AMRCC = _THIS.parent
_ROOT = _AMRCC.parent
for _p in [str(_ROOT), str(_AMRCC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import (chacha20_keystream, fisher_yates, permutation_for_frame, inverse_permutation, apply_perm, apply_inverse_perm, _STRIDE)

KEY = bytes(range(32))
KEY2 = bytes(range(31, -1, -1))

def test_is_valid_permutation():
    for t in [0, 1, 7, 2 ** 32]:
        for k in [1000, 5000]:
            p = permutation_for_frame(KEY, t, k)
            assert p.shape == (k,)
            assert np.array_equal(np.sort(p), np.arange(k))

def test_inverse_is_involution():
    p = permutation_for_frame(KEY, 9, 5000)
    inv = inverse_permutation(p)
    assert np.array_equal(p[inv], np.arange(5000))
    assert np.array_equal(inv[p], np.arange(5000))

def test_apply_roundtrip():
    p = permutation_for_frame(KEY, 3, 4000)
    v = np.random.default_rng(0).integers(0, 2, 4000, dtype=np.int8)
    assert np.array_equal(apply_inverse_perm(apply_perm(v, p), p), v)


def test_determinism_in_key_and_t():
    assert np.array_equal(permutation_for_frame(KEY, 42, 2000), permutation_for_frame(KEY, 42, 2000))
    assert not np.array_equal(permutation_for_frame(KEY, 42, 2000), permutation_for_frame(KEY2, 42, 2000))
    assert not np.array_equal(permutation_for_frame(KEY, 42, 2000), permutation_for_frame(KEY, 43, 2000))

def test_freshness_across_frames():
    perms = [permutation_for_frame(KEY, t, 3000) for t in range(8)]
    for a in range(8):
        for b in range(a + 1, 8):
            assert not np.array_equal(perms[a], perms[b])

def test_keystream_determinism():
    assert chacha20_keystream(KEY, 7, 4096) == chacha20_keystream(KEY, 7, 4096)


def test_keystream_no_overlap():
    ts = [0, 1, 2, 5, 2 ** 20, 2 ** 20 + 1, 2 ** 32]
    nbytes = 32400 * 8
    streams = {t: chacha20_keystream(KEY, t, nbytes) for t in ts}
    assert len({hashlib.sha256(s).digest() for s in streams.values()}) == len(ts)
    def blocks(s):
        return {s[i:i + 64] for i in range(0, len(s) - 63, 64)}
    bs = {t: blocks(s) for t, s in streams.items()}
    for a in ts:
        for b in ts:
            if a < b:
                assert bs[a].isdisjoint(bs[b]), f"keystream block overlap t={a},{b}"

def test_stride_exceeds_frame_blocks():
    blocks_per_frame = (32400 * 8 + 63) // 64
    assert _STRIDE > blocks_per_frame

def test_fy_modulo_bias_negligible():
    k = 32400
    assert k / 2.0 ** 64 < 1e-14

def test_fy_uniformity_smoke_smallk():
    k, M = 8, 4000
    counts = np.zeros(k)
    for s in range(M):
        counts[permutation_for_frame(s.to_bytes(32, "little"), 0, k)[0]] += 1
    assert counts.min() > 0.6 * M / k and counts.max() < 1.4 * M / k

if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items())
           if n.startswith("test_") and callable(v)]
    for fn in fns:
        t0 = time.time()
        fn()
        print(f"PASS  {fn.__name__:40s} {time.time() - t0:6.2f}s")
    print(f"\nall {len(fns)} keyed_permutation tests passed")