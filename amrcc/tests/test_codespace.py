from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_AMRCC = _THIS.parent
_ROOT = _AMRCC.parent
for _p in [str(_ROOT), str(_AMRCC), str(_ROOT / "smap_msl_data"), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "ldpc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from nullspace_attack_utils.ldpc_ops import LDPCCode
from keyed_permutation import permutation_for_frame, inverse_permutation

KEYS = [bytes(range(32)), bytes(range(31, -1, -1)), bytes([7]) * 32]
TS = [0, 1, 2 ** 20]
SLOT_BITS, HEADER_BITS, N_SLOTS = 800, 400, 40

_CODE = None
def code():
    global _CODE
    if _CODE is None:
        _CODE = LDPCCode.dvbs2_long_rate12()
    return _CODE

def _assert_codeword(c, m, key, t, tag, stats):
    perm = permutation_for_frame(key, t, c.k)
    cw = np.asarray(c.encode((np.asarray(m, np.int8)[perm]).astype(np.int8)))
    assert c.is_codeword(cw), f"syndrome != 0: {tag} (t={t})"
    stats["checked"] += 1
    return cw

def test_structured_messages():
    c = code()
    stats = {"checked": 0}
    k = c.k
    msgs = {"all_zero": np.zeros(k, np.int8), "all_one": np.ones(k, np.int8), "header_only": np.r_[np.ones(HEADER_BITS, np.int8), np.zeros(k - HEADER_BITS, np.int8)]}
    for s in range(N_SLOTS):
        m = np.zeros(k, np.int8)
        m[HEADER_BITS + s * SLOT_BITS: HEADER_BITS + (s + 1) * SLOT_BITS] = 1
        msgs[f"slot_{s}"] = m
    for tag, m in msgs.items():
        for key in KEYS:
            for t in TS:
                _assert_codeword(c, m, key, t, tag, stats)
    print(f"      structured: {stats['checked']} codeword checks OK")

def test_weight_sweep_random():
    c = code()
    rng = np.random.default_rng(0)
    stats = {"checked": 0}
    weights = [1, 2, 3, 5, 10, 100, 1000, c.k // 2, c.k - 1]
    for wt in weights:
        for _ in range(20):
            m = np.zeros(c.k, np.int8)
            m[rng.choice(c.k, size=wt, replace=False)] = 1
            _assert_codeword(c, m, KEYS[0], TS[1], f"wt={wt}", stats)
    for _ in range(1000):
        m = rng.integers(0, 2, c.k, dtype=np.int8)
        _assert_codeword(c, m, KEYS[rng.integers(len(KEYS))], int(rng.choice(TS)), "rand", stats)
    print(f"      weight/random: {stats['checked']} codeword checks OK")


def test_codebook_identity_and_decode():
    c = code()
    rng = np.random.default_rng(1)
    for _ in range(200):
        m = rng.integers(0, 2, c.k, dtype=np.int8)
        key, t = KEYS[rng.integers(len(KEYS))], int(rng.choice(TS))
        perm = permutation_for_frame(key, t, c.k)
        pm = (m[perm]).astype(np.int8)
        cw = np.asarray(c.encode(pm))
        assert np.array_equal(np.asarray(c.info_bit_extract(cw)).astype(np.int8), pm)
        assert np.array_equal(np.asarray(c.info_bit_extract(cw)).astype(np.int8)[inverse_permutation(perm)], m)

def test_unit_vector_sweep():
    c = code()
    stats = {"checked": 0}
    full = os.environ.get("FULL") == "1"
    positions = range(c.k) if full else sorted(
        {0, 1, 2, HEADER_BITS - 1, HEADER_BITS, c.k - 1,
         *np.random.default_rng(2).choice(c.k, size=300, replace=False).tolist()})
    t0 = time.time()
    for p in positions:
        m = np.zeros(c.k, np.int8)
        m[p] = 1
        _assert_codeword(c, m, KEYS[0], TS[0], f"e_{p}", stats)
    print(f"      unit vectors ({'FULL' if full else 'sampled'}): "
          f"{stats['checked']} checks OK in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items())
           if n.startswith("test_") and callable(v)]
    for fn in fns:
        t0 = time.time()
        fn()
        print(f"PASS  {fn.__name__:42s} {time.time() - t0:6.2f}s")
    print(f"\nall {len(fns)} codespace tests passed")