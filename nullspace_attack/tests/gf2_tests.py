import sys
import time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
from nullspace_attack.gf2 import (pack_bits, unpack_bits, gf2_matvec, gf2_matmul, rref, rank, solve, inverse, nullspace_basis)
from ldpc.dvb_s2_short import build_dvbs2_h_short_rate12, encode_dvbs2_short_rate12, RATE_1_2_SHORT

def _ref_rank_nulldim(A_bits: np.ndarray) -> tuple[int, int]:
    A = (np.asarray(A_bits, dtype=np.uint8) & 1).copy()
    m, n = A.shape
    r = 0
    for c in range(n):
        if r >= m:
            break
        piv = next((i for i in range(r, m) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
    return r, n - r

def _ref_matvec(A_bits, x_bits):
    return (A_bits.astype(np.int64) @ x_bits.astype(np.int64)) % 2

def _ref_matmul(A_bits, B_bits):
    return (A_bits.astype(np.int64) @ B_bits.astype(np.int64)) % 2

def _random_invertible(n, rng, tries=200):
    for _ in range(tries):
        A = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
        if rank(pack_bits(A), n) == n:
            return A
    raise RuntimeError("could not draw an invertible matrix")

def _ok(name):
    print(f"  PASS  {name}")

def t1_pack_unpack(rng):
    print("T1  pack/unpack roundtrip")
    for n_cols in (1, 7, 63, 64, 65, 100, 257, 16200):
        v = rng.integers(0, 2, size=n_cols, dtype=np.uint8)
        assert np.array_equal(unpack_bits(pack_bits(v), n_cols), v), f"1-D n={n_cols}"
    for shape in [(1, 64), (5, 63), (9, 65), (13, 200), (40, 800)]:
        M = rng.integers(0, 2, size=shape, dtype=np.uint8)
        assert np.array_equal(unpack_bits(pack_bits(M), shape[1]), M), f"2-D {shape}"
    _ok("roundtrip exact for 1-D and 2-D, on/around 64-bit boundaries")

def t2_matvec(rng):
    print("T2  gf2_matvec vs reference")
    for (m, n) in [(1, 10), (8, 64), (17, 130), (50, 333)]:
        A = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
        x = rng.integers(0, 2, size=n, dtype=np.uint8)
        got = gf2_matvec(pack_bits(A), pack_bits(x), n)
        assert np.array_equal(got, _ref_matvec(A, x)), f"matvec {m}x{n}"
    _ok("matvec matches (A@x)%2 across sizes")

def t3_matmul(rng):
    print("T3  gf2_matmul vs reference")
    for (m, n, p) in [(4, 5, 6), (10, 64, 7), (20, 130, 65), (33, 70, 200)]:
        A = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
        B = rng.integers(0, 2, size=(n, p), dtype=np.uint8)
        got = unpack_bits(gf2_matmul(pack_bits(A), pack_bits(B), n), p)
        assert np.array_equal(got, _ref_matmul(A, B)), f"matmul {m}x{n}x{p}"
    _ok("matmul matches (A@B)%2 across sizes")

def t4_rank(rng):
    print("T4  rank vs independent reference")
    cases = [(10, 10), (30, 50), (50, 30), (64, 64), (100, 130)]
    for (m, n) in cases:
        A = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
        ref_r, _ = _ref_rank_nulldim(A)
        assert rank(pack_bits(A), n) == ref_r, f"full-random {m}x{n}"
    A = rng.integers(0, 2, size=(40, 60), dtype=np.uint8)
    A[5] = A[3]
    A[10] = 0
    A[11] = (A[2] ^ A[7]) & 1
    ref_r, _ = _ref_rank_nulldim(A)
    assert rank(pack_bits(A), 60) == ref_r, "rank-deficient"
    _ok("rank matches reference (full-rank and rank-deficient)")

def t5_solve(rng):
    print("T5  solve (consistent recovers; inconsistent -> None)")
    for (m, n) in [(20, 20), (40, 25), (25, 40), (80, 64)]:
        A = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
        Ap = pack_bits(A)
        x_true = rng.integers(0, 2, size=n, dtype=np.uint8)
        b = _ref_matvec(A, x_true).astype(np.uint8)
        x = solve(Ap, b, n)
        assert x is not None, f"consistent {m}x{n} returned None"
        assert np.array_equal(_ref_matvec(A, x), b), f"A x != b for {m}x{n}"
    A = rng.integers(0, 2, size=(30, 20), dtype=np.uint8)
    A[7] = 0
    b = _ref_matvec(A, rng.integers(0, 2, size=20, dtype=np.uint8)).astype(np.uint8)
    b[7] = 1
    assert solve(pack_bits(A), b, 20) is None, "inconsistent should be None"
    _ok("solve recovers a valid solution; detects inconsistency")

def t6_inverse(rng):
    print("T6  inverse (A A^-1 = I; singular -> None)")
    for n in (5, 8, 16, 33, 64):
        A = _random_invertible(n, rng)
        Ainv = inverse(pack_bits(A), n)
        assert Ainv is not None, f"invertible {n} returned None"
        prod = unpack_bits(gf2_matmul(pack_bits(A), Ainv, n), n)
        assert np.array_equal(prod, np.eye(n, dtype=np.uint8)), f"A A^-1 != I, n={n}"
    A = _random_invertible(10, rng)
    A[4] = 0
    assert inverse(pack_bits(A), 10) is None, "singular should be None"
    _ok("inverse correct; singular detected")

def t7_nullspace(rng):
    print("T7  nullspace_basis (dim = n - rank; A v = 0; independent)")
    for (m, n) in [(10, 16), (30, 50), (64, 100), (20, 20)]:
        A = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
        Ap = pack_bits(A)
        r = rank(Ap, n)
        ref_r, ref_null = _ref_rank_nulldim(A)
        Bp = nullspace_basis(Ap, n)
        n_free = Bp.shape[0]
        assert n_free == n - r == ref_null, f"null dim {n_free} != {n-r}/{ref_null}"
        for row in range(n_free):
            v = unpack_bits(Bp[row], n)
            assert not _ref_matvec(A, v).any(), f"A v != 0 (vec {row}, {m}x{n})"
        if n_free > 0:
            assert rank(Bp, n) == n_free, f"basis not independent, {m}x{n}"
    _ok("null space: correct dimension, A v = 0, independent basis")

def t8_real_short_H():
    print("T8  real short-frame H: pack+matvec syndrome, staircase rank")
    H = build_dvbs2_h_short_rate12()
    n, k = RATE_1_2_SHORT.n, RATE_1_2_SHORT.k
    m = n - k
    rng = np.random.default_rng(7)
    Hd = np.asarray(H.todense(), dtype=np.uint8)
    Hp = pack_bits(Hd)
    for _ in range(3):
        info = rng.integers(0, 2, size=k, dtype=np.int8)
        cw = encode_dvbs2_short_rate12(info)
        syn_ref = (H @ cw.astype(np.int64)) % 2
        syn_gf2 = gf2_matvec(Hp, pack_bits(cw.astype(np.uint8)), n)
        assert not syn_ref.any(), "encoded codeword has nonzero reference syndrome"
        assert np.array_equal(syn_gf2, syn_ref), "gf2 syndrome != reference syndrome"
    _ok("gf2_matvec(H, codeword) == (H@codeword)%2 == 0 on real H")
    P = Hd[:, k:]
    assert P.shape == (m, m)
    assert np.array_equal(np.diag(P), np.ones(m, dtype=np.uint8)), "diag != 1"
    sub = np.array([P[j, j - 1] for j in range(1, m)], dtype=np.uint8)
    assert np.all(sub == 1), "subdiagonal != 1"
    upper_nnz = int(np.triu(P, k=1).sum())
    below_nnz = int((np.tril(P, k=-2)).sum())
    assert upper_nnz == 0 and below_nnz == 0, "parity block is not lower-bidiagonal"
    print(f"        staircase block {m}x{m}: unit diag + unit subdiag, "
          f"nothing else -> invertible -> rank(H)={m}=n-k -> dim N(H)={k}")
    _ok("rank(H)=n-k and dim N(H)=k established structurally (cheap)")

def t9_scaling_probe(rng):
    print("T9  rref scaling probe (informs Day-4 long-frame GE cost)")
    print(f"        {'dim':>6}  {'time(s)':>9}  {'~u64-XOR ops':>14}")
    for nsz in (256, 512, 1024, 2048):
        A = rng.integers(0, 2, size=(nsz, nsz), dtype=np.uint8)
        Ap = pack_bits(A)
        t0 = time.time()
        _, piv = rref(Ap, nsz)
        dt = time.time() - t0
        nwords = (nsz + 63) // 64
        est = len(piv) * nsz * nwords 
        print(f"        {nsz:>6}  {dt:>9.3f}  {est:>14,}")
    print("        (extrapolate to long-frame: rank~32,400, rows~64,800, "
          "nwords~1,013; decides naive-vs-optimized GE for Day 4)")

def main():
    rng = np.random.default_rng(0)
    t1_pack_unpack(rng)
    t2_matvec(rng)
    t3_matmul(rng)
    t4_rank(rng)
    t5_solve(rng)
    t6_inverse(rng)
    t7_nullspace(rng)
    t8_real_short_H()
    t9_scaling_probe(rng)
    print("\nAll gf2 correctness tests PASSED.")

if __name__ == "__main__":
    main()