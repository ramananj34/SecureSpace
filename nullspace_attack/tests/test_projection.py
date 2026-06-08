import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
import scipy.sparse as sp
import nullspace_attack.gf2 as gf2
from nullspace_attack.ldpc_ops import LDPCCode
from nullspace_attack.projection import (b_transpose_apply, information_set_matrix, is_information_set, project_information_set, normal_form_matrix, normal_form_projection)

def tiny_systematic_code(n=24, k=12, seed=1):
    rng = np.random.default_rng(seed)
    m = n - k
    P = rng.integers(0, 2, size=(m, k), dtype=np.uint8)
    Ppack = gf2.pack_bits(P)
    def enc(info):
        info = np.asarray(info).astype(np.int8).ravel()
        parity = (P.astype(np.int64) @ info.astype(np.int64)) % 2
        return np.concatenate([info, parity.astype(np.int8)])
    H = sp.csr_matrix(np.hstack([P, np.eye(m, dtype=np.uint8)]).astype(np.int8))
    return LDPCCode("tiny-systematic", H, n, k, enc), P, Ppack

def _ok(name):
    print(f"  PASS  {name}")

def test_tiny():
    print("Tiny synthetic systematic code:")
    code, P, Ppack = tiny_systematic_code()
    n, k, m = code.n, code.k, code.m
    rng = np.random.default_rng(2)
    B_dense = np.vstack([np.eye(k, dtype=np.int64), P.astype(np.int64)])
    info = rng.integers(0, 2, size=k, dtype=np.int8)
    cw = code.encode(info)
    assert code.is_codeword(cw) and np.array_equal(cw[:k], info)
    assert np.array_equal(code.parity_of(info), (P.astype(np.int64) @ info) % 2)
    _ok("tiny code systematic, P_sys = P")
    for _ in range(5):
        dp = rng.integers(0, 2, size=n, dtype=np.int8)
        got = b_transpose_apply(code, dp, Ppack)
        ref = (B_dense.T @ dp.astype(np.int64)) % 2
        assert np.array_equal(got, ref), "b_transpose_apply != B^T delta'"
    _ok("b_transpose_apply matches B^T delta'")
    G_sys = gf2.unpack_bits(information_set_matrix(code, range(k), Ppack), k)
    assert np.array_equal(G_sys, np.eye(k, dtype=np.uint8)), "systematic G_S != I_k"
    for _ in range(5):
        S = sorted(rng.choice(n, size=k, replace=False).tolist())
        G_S = gf2.unpack_bits(information_set_matrix(code, S, Ppack), k)
        assert np.array_equal(G_S, B_dense[S, :].astype(np.uint8)), "G_S != G_LDPC[S,:]"
    _ok("information_set_matrix matches G_LDPC rows (and = I_k on systematic set)")
    dp = rng.integers(0, 2, size=n, dtype=np.int8)
    via_set = project_information_set(code, dp, range(k), Ppack)
    via_sys = code.project_systematic(dp)
    assert np.array_equal(via_set, via_sys), "info-set(systematic) != project_systematic"
    assert is_information_set(code, range(k), Ppack)
    # a random information set
    S = None
    for _ in range(50):
        cand = sorted(rng.choice(n, size=k, replace=False).tolist())
        if is_information_set(code, cand, Ppack):
            S = cand
            break
    assert S is not None, "could not find a random information set"
    proj = project_information_set(code, dp, S, Ppack)
    assert code.is_codeword(proj), "info-set projection not a codeword"
    assert np.array_equal(np.asarray(proj)[S], dp[S]), "projection differs from delta' on S"
    bad_S = list(range(k - 1)) + [0]  # position 0 appears twice -> rank k-1
    assert not is_information_set(code, bad_S, Ppack), "rank-deficient set not detected"
    res = project_information_set(code, dp, bad_S, Ppack)
    assert res is None or (code.is_codeword(res) and np.array_equal(np.asarray(res)[bad_S], dp[bad_S])), "rank-deficient-set projection must be None or a valid codeword preserving delta'|_S"
    _ok("project_information_set: systematic==project_systematic, c|_S=delta'|_S, None off-set")
    M_packed = normal_form_matrix(code, Ppack)
    M = gf2.unpack_bits(M_packed, k)
    M_ref = ((P.astype(np.int64).T @ P.astype(np.int64)) % 2 + np.eye(k, dtype=np.int64)) % 2
    assert np.array_equal(M, M_ref.astype(np.uint8)), "normal_form_matrix != I + P^T P"
    M_rank = gf2.rank(M_packed, k)
    dp = rng.integers(0, 2, size=n, dtype=np.int8)
    nf = normal_form_projection(code, dp, M_packed, Ppack)
    if nf is None:
        print(f"        normal-form M is SINGULAR over F2 (rank {M_rank}/{k}) -> "
              f"fallback-1 returns None (the degenerate case)")
    else:
        assert code.is_codeword(nf), "normal-form projection not a codeword"
        rhs = b_transpose_apply(code, dp, Ppack)
        c = code.info_bit_extract(nf)
        assert np.array_equal(gf2.gf2_matvec(M_packed, gf2.pack_bits(c.astype(np.uint8)), k), rhs), \
            "normal-form c does not solve M c = B^T delta'"
        print(f"        normal-form M nonsingular (rank {M_rank}/{k}); projection solves M c = rhs")
    _ok("normal_form_matrix matches I + P^T P; projection consistent (singular -> None)")

def test_short_dvbs2_tie():
    print("\nReal short DVB-S2 code (systematic-equivalence tie):")
    code = LDPCCode.dvbs2_short_rate12()
    P_packed = code.materialize_P_sys(packed=True)
    rng = np.random.default_rng(3)
    dp = rng.integers(0, 2, size=code.n, dtype=np.int8)
    via_set = project_information_set(code, dp, range(code.k), P_packed)
    via_sys = code.project_systematic(dp)
    assert np.array_equal(via_set, via_sys), "short: info-set(systematic) != project_systematic"
    assert code.is_codeword(via_set)
    _ok("project_information_set(systematic) == project_systematic on real short code")

def main():
    test_tiny()
    test_short_dvbs2_tie()
    print("\nprojection tests PASSED.")

if __name__ == "__main__":
    main()