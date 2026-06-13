from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import nullspace_attack_utils.gf2 as gf2
from theorem_1.secret_integrity import SecretIntegrity
from theorem_1.combined_defense import (CombinedDefense, identity_packed, nullspace_basis_fast)

HEADER, SLOT, NSLOTS = 2, 3, 4
K = HEADER + NSLOTS * SLOT
M = 10
N = K + M
MPRIME = N // 4
SEED = 20250612

class SynCode:
    def __init__(self, k, m, seed):
        self.k, self.m, self.n = k, m, k + m
        rng = np.random.default_rng(seed)
        self.P = rng.integers(0, 2, size=(m, k), dtype=np.uint8)
        Hdense = np.hstack([self.P, np.eye(m, dtype=np.uint8)])
        self.H = csr_matrix(Hdense)

    def encode(self, info):
        info = np.asarray(info, dtype=np.uint8).ravel()
        par = (self.P @ info) % 2
        return np.concatenate([info, par]).astype(np.uint8)

    def info_bit_extract(self, cw):
        return np.asarray(cw)[:self.k]

    def syndrome(self, word):
        w = np.asarray(word, dtype=np.uint8).ravel()
        return ((self.H @ w.astype(np.int64)) % 2).astype(np.uint8)

    def materialize_P_sys(self, packed=True):
        return gf2.pack_bits(self.P) if packed else self.P.copy()

def _build():
    code = SynCode(K, M, SEED)
    secret = SecretIntegrity(N, MPRIME, SEED + 1)
    cd = CombinedDefense.build(secret, code, verbose=False,header_bits=HEADER, slot_bits=SLOT, n_slots=NSLOTS)
    return code, secret, cd

def test_identity_packed():
    I = identity_packed(14)
    Ib = gf2.unpack_bits(I, 14)
    assert np.array_equal(Ib, np.eye(14, dtype=np.uint8))

def test_nullspace_basis_fast_matches_frozen():
    rng = np.random.default_rng(3)
    for (nr, nc) in [(6, 14), (10, 24), (8, 20), (16, 16)]:
        A = gf2.pack_bits(rng.integers(0, 2, size=(nr, nc), dtype=np.uint8))
        Bf = nullspace_basis_fast(A, nc)
        Bg = gf2.nullspace_basis(A, nc)
        assert Bf.shape == Bg.shape
        Bfb = np.atleast_2d(gf2.unpack_bits(Bf, nc))
        Bgb = np.atleast_2d(gf2.unpack_bits(Bg, nc))
        assert {tuple(r) for r in Bfb} == {tuple(r) for r in Bgb}
        for r in range(Bf.shape[0]):
            assert int(gf2.gf2_matvec(A, Bf[r]).sum()) == 0

def test_Mprime_equals_Hprime_times_GLDPC():
    code, secret, cd = _build()
    Mb = np.atleast_2d(gf2.unpack_bits(cd.Mp, K))
    for j in [0, 1, 7, 13]:
        ej = np.zeros(K, dtype=np.uint8); ej[j] = 1
        gj = code.encode(ej)
        hpe = gf2.gf2_matvec(secret.H_packed, gf2.pack_bits(gj))
        assert np.array_equal(Mb[:, j], hpe)

def test_Bcomb_subset_and_dim():
    code, secret, cd = _build()
    rep = cd.verify(n_sample=cd.dim)
    assert rep["all_pass_H"] and rep["all_pass_Hprime"] and rep["basis_independent"]
    Hb = gf2.unpack_bits(gf2.pack_bits(np.hstack([code.P, np.eye(M, dtype=np.uint8)])), N)
    stacked = gf2.pack_bits(np.vstack([Hb, np.atleast_2d(gf2.unpack_bits(secret.H_packed, N))]))
    rank_stacked = gf2.rank(stacked, N)
    assert N - rank_stacked == cd.dim
    assert cd.rank_Mp == K - cd.dim

def test_targetability_full():
    code, secret, cd = _build()
    for slot in range(NSLOTS):
        assert cd.targetability(slot) == SLOT

def test_project_exact_and_passes_both():
    code, secret, cd = _build()
    rng = np.random.default_rng(5)
    for slot in range(NSLOTS):
        for _ in range(8):
            wb = rng.integers(0, 2, size=SLOT, dtype=np.uint8)
            res = cd.project(wb, slot)
            assert res is not None
            v, delta = res
            lo, hi = cd.slot_footprint(slot)
            assert np.array_equal(v[lo:hi], wb)
            assert int(code.syndrome(delta).sum()) == 0
            hp = gf2.gf2_matvec(secret.H_packed, gf2.pack_bits(delta))
            assert int(hp.sum()) == 0
            assert int(gf2.gf2_matvec(cd.Mp, gf2.pack_bits(v)).sum()) == 0

def test_arms_record():
    code, secret, cd = _build()
    rng = np.random.default_rng(7)
    slot = 1
    wb = rng.integers(0, 2, size=SLOT, dtype=np.uint8)
    wb[0] = 1
    rec = cd.arms(wb, slot)
    assert rec["targetable"] and rec["combined_recovered"]
    assert rec["combined_passes_both"]
    assert rec["footprint_exact"]
    assert rec["naive_H_synd"] == 0
    assert rec["naive_flagged_by_combined"] == (rec["naive_Hprime_synd_weight"] > 0)
    assert rec["collateral_info_weight"] >= 0
    assert rec["info_weight_total"] == rec["window_weight"] + rec["collateral_info_weight"]

def test_naive_generically_flagged():
    code, secret, cd = _build()
    rng = np.random.default_rng(11)
    flagged = 0
    trials = 30
    for _ in range(trials):
        wb = rng.integers(0, 2, size=SLOT, dtype=np.uint8)
        if wb.sum() == 0:
            continue
        flagged += int(cd.naive_check(wb, 2) > 0)
    assert flagged >= int(0.7 * trials)

def test_rank_stacked_helper_matches():
    code, secret, cd = _build()
    rep = cd.rank_stacked(verbose=False)
    assert rep["matches_Bnull_dim"] and rep["dim_stacked"] == cd.dim

_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

def main():
    print(f"combined_defense tests  (synthetic n={N} k={K} m={M} m'={MPRIME}; "
          f"frame {HEADER}+{NSLOTS}x{SLOT})\n" + "-" * 64)
    fails = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            fails += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("-" * 64)
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())