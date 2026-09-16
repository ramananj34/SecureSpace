import numpy as np
import pytest
import sys
from pathlib import Path
_THIS = Path(__file__).resolve().parent
_AMRCC = _THIS.parent
_ROOT = _AMRCC.parent
for _p in [str(_ROOT), str(_AMRCC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import chacha20_keystream, fisher_yates, permutation_for_frame
from fy_fast import fisher_yates_fast, permutation_for_frame_fast, _HAVE_NUMBA

KEY = bytes(range(32))

@pytest.mark.parametrize("k", [2, 17, 256, 1000, 4051, 32400])
def test_matches_prototype(k):
    ks = chacha20_keystream(KEY, 0, k * 8)
    p_proto = fisher_yates(ks, k)
    p_fast = fisher_yates_fast(ks, k)
    assert np.array_equal(p_proto, p_fast)

@pytest.mark.parametrize("t", [0, 1, 7, 9999])
def test_matches_per_frame(t):
    k = 32400
    assert np.array_equal(permutation_for_frame(KEY, t, k), permutation_for_frame_fast(KEY, t, k))

def test_is_valid_permutation():
    k = 32400
    p = fisher_yates_fast(chacha20_keystream(KEY, 3, k * 8), k)
    assert p.shape == (k,)
    assert np.array_equal(np.sort(p), np.arange(k))

def test_distinct_keys_distinct_perms():
    k = 4096
    a = fisher_yates_fast(chacha20_keystream(bytes([1]) * 32, 0, k * 8), k)
    b = fisher_yates_fast(chacha20_keystream(bytes([2]) * 32, 0, k * 8), k)
    assert not np.array_equal(a, b)

def test_numba_actually_loaded():
    if not _HAVE_NUMBA:
        pytest.skip("numba unavailable -- fisher_yates_fast fell back to the Python prototype")