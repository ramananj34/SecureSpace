from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import weak_prng as W
from keyed_permutation import permutation_for_frame

def _is_perm(p, k):
    p = np.asarray(p)
    return p.shape == (k,) and p.dtype == np.int64 and np.array_equal(np.sort(p), np.arange(k))

def _mean_fixed(gen, k, M=400, seed0=0):
    return float(np.mean([
        np.sum(gen(np.random.default_rng([seed0, i]).bytes(32), 0, k) == np.arange(k))
        for i in range(M)]))

K = 256

def test_strong_matches_frozen_bitforbit():
    for t in [0, 1, 7, 1_000_000]:
        key = np.random.default_rng([5, t]).bytes(32)
        assert np.array_equal(W.strong(key, t, K), permutation_for_frame(key, t, K))

def test_all_generators_return_valid_permutations():
    for name, gen in W.ladder():
        key = np.random.default_rng(hash(name) & 0x7FFFFFFF).bytes(32)
        p = gen(key, 3, K)
        assert _is_perm(p, K), f"{name} did not return a valid permutation"

def test_chacha20r_core_rfc_faithful():
    key = bytes(range(32)); nonce = bytes.fromhex("000000090000004a00000000")
    blk = W._chacha_block(key, 1, nonce, 20)
    fw = np.frombuffer(blk, dtype="<u4")[:4]
    assert [int(x) for x in fw] == [0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3]

def test_identity_is_identity():
    for t in [0, 9]:
        assert np.array_equal(W.identity(np.random.default_rng(t).bytes(32), t, K), np.arange(K))

def test_reduced_round_chacha_permutations_stay_near_uniform():
    fp = {r: _mean_fixed(W.reduced_round_chacha(r), K) for r in [20, 8, 4, 2]}
    for r, v in fp.items():
        assert v < 5.0, f"chacha{r}r fixed points {v} unexpectedly high"
    assert max(fp.values()) - min(fp.values()) < 3.0

def test_biased_bits_more_fixed_points_as_p_departs_half():
    fp = {p: _mean_fixed(W.biased_bit_fy(p), K) for p in [0.50, 0.75, 0.95]}
    assert fp[0.50] < 5.0
    assert fp[0.95] > fp[0.75] > fp[0.50] - 1e-9

def test_truncated_keyspace_collapses():
    g1 = W.truncated_keyspace(1)
    p_a = g1(np.random.default_rng(1).bytes(32), 4, K)
    p_b = g1(np.random.default_rng(2).bytes(32), 4, K)
    assert np.array_equal(p_a, p_b), "trunc_keyspace(1) must ignore the key"
    g = W.truncated_keyspace(1 << 10)
    perms = {g(np.random.default_rng(i).bytes(32), 4, K).tobytes() for i in range(200)}
    assert len(perms) <= 1 << 10

def test_local_shuffle_bounded_displacement_and_endpoints():
    g = W.local_shuffle(2)
    key = np.random.default_rng(0).bytes(32)
    p = g(key, 0, K)
    assert _is_perm(p, K)
    assert np.max(np.abs(p - np.arange(K))) <= 1
    assert np.array_equal(W.local_shuffle(1)(key, 0, K), np.arange(K))
    fp2 = _mean_fixed(W.local_shuffle(2), K)
    fp64 = _mean_fixed(W.local_shuffle(64), K)
    assert fp2 > fp64

def test_determinism():
    for name, gen in W.ladder():
        key = np.random.default_rng(42).bytes(32)
        assert np.array_equal(gen(key, 11, K), gen(key, 11, K)), f"{name} not deterministic"

def test_ladder_registry_consistent():
    assert set(W.GENERATORS) == {n for n, _ in W.ladder()}
    assert "strong" in W.GENERATORS and "identity" in W.GENERATORS

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")