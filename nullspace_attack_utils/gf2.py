from __future__ import annotations
import numpy as np

__all__ = ["pack_bits", "unpack_bits", "gf2_matvec", "gf2_matmul", "rref", "rank", "solve", "inverse", "nullspace_basis"]
_U1 = np.uint64(1)
_FAST_PATH_LIMIT = 1 << 26

def _u(n: int) -> np.uint64:
    return np.uint64(n)

def pack_bits(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint64)
    squeeze = bits.ndim == 1
    if squeeze:
        bits = bits[None, :]
    nrows, n_cols = bits.shape
    nwords = (n_cols + 63) // 64
    out = np.zeros((nrows, nwords), dtype=np.uint64)
    if nrows * nwords * 64 <= _FAST_PATH_LIMIT:
        padded = np.zeros((nrows, nwords * 64), dtype=np.uint64)
        padded[:, :n_cols] = bits
        padded = padded.reshape(nrows, nwords, 64)
        shifts = np.arange(64, dtype=np.uint64)
        out = (padded << shifts).sum(axis=2, dtype=np.uint64)
    else:
        for c in range(n_cols):
            out[:, c >> 6] |= bits[:, c] << _u(c & 63)
    return out[0] if squeeze else out

def unpack_bits(packed: np.ndarray, n_cols: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint64)
    squeeze = packed.ndim == 1
    if squeeze:
        packed = packed[None, :]
    nrows, nwords = packed.shape
    if nrows * nwords * 64 <= _FAST_PATH_LIMIT:
        shifts = np.arange(64, dtype=np.uint64)
        out = ((packed[:, :, None] >> shifts) & _U1).astype(np.uint8)
        out = out.reshape(nrows, nwords * 64)
    else:
        out = np.zeros((nrows, nwords * 64), dtype=np.uint8)
        for w in range(nwords):
            col = packed[:, w]
            for j in range(64):
                out[:, w * 64 + j] = ((col >> _u(j)) & _U1).astype(np.uint8)
    out = out[:, :n_cols]
    return out[0] if squeeze else out

def _parity_u64(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.uint64)
    v = v ^ (v >> _u(32))
    v = v ^ (v >> _u(16))
    v = v ^ (v >> _u(8))
    v = v ^ (v >> _u(4))
    v = v ^ (v >> _u(2))
    v = v ^ (v >> _u(1))
    return (v & _U1).astype(np.uint8)

def gf2_matvec(A: np.ndarray, x: np.ndarray, n_cols: int | None = None) -> np.ndarray:
    A = np.atleast_2d(np.asarray(A, dtype=np.uint64))
    x = np.asarray(x, dtype=np.uint64).reshape(-1)
    folded = np.bitwise_xor.reduce(A & x.reshape(1, -1), axis=1)
    return _parity_u64(folded)

def _row_support(row_packed: np.ndarray, n_cols: int) -> np.ndarray:
    return np.nonzero(unpack_bits(row_packed, n_cols))[0]

def gf2_matmul(A: np.ndarray, B: np.ndarray, n_cols_A: int) -> np.ndarray:
    A = np.atleast_2d(np.asarray(A, dtype=np.uint64))
    B = np.atleast_2d(np.asarray(B, dtype=np.uint64))
    m, wB = A.shape[0], B.shape[1]
    out = np.zeros((m, wB), dtype=np.uint64)
    for i in range(m):
        cols = _row_support(A[i], n_cols_A)
        if cols.size:
            out[i] = np.bitwise_xor.reduce(B[cols], axis=0)
    return out

def rref(A: np.ndarray, n_cols: int) -> tuple[np.ndarray, list[int]]:
    A = np.array(A, dtype=np.uint64, copy=True)
    A = np.atleast_2d(A)
    nrows = A.shape[0]
    pivot_cols: list[int] = []
    r = 0
    for c in range(n_cols):
        if r >= nrows:
            break
        w = c >> 6
        bit = _u(c & 63)
        colbits = (A[r:, w] >> bit) & _U1
        nz = np.nonzero(colbits)[0]
        if nz.size == 0:
            continue
        pr = r + int(nz[0])
        if pr != r:
            A[[r, pr]] = A[[pr, r]]
        allbits = (A[:, w] >> bit) & _U1
        allbits[r] = 0
        targets = np.nonzero(allbits)[0]
        if targets.size:
            A[targets] ^= A[r]
        pivot_cols.append(c)
        r += 1
    return A, pivot_cols

def rank(A: np.ndarray, n_cols: int) -> int:
    _, piv = rref(A, n_cols)
    return len(piv)

def solve(A: np.ndarray, b: np.ndarray, n_cols: int) -> np.ndarray | None:
    A = np.atleast_2d(np.asarray(A, dtype=np.uint64))
    nrows = A.shape[0]
    b = np.asarray(b, dtype=np.uint8).reshape(-1)
    assert b.shape[0] == nrows, f"b length {b.shape[0]} != nrows {nrows}"
    aug_cols = n_cols + 1
    ab_bits = np.zeros((nrows, aug_cols), dtype=np.uint8)
    ab_bits[:, :n_cols] = unpack_bits(A, n_cols)
    ab_bits[:, n_cols] = b
    R, piv = rref(pack_bits(ab_bits), aug_cols)
    if n_cols in piv:
        return None
    Rbits = unpack_bits(R, aug_cols)
    x = np.zeros(n_cols, dtype=np.uint8)
    for row_idx, c in enumerate(piv):
        x[c] = Rbits[row_idx, n_cols]
    return x

def inverse(A: np.ndarray, n: int) -> np.ndarray | None:
    A = np.atleast_2d(np.asarray(A, dtype=np.uint64))
    assert A.shape[0] == n, f"expected {n} rows, got {A.shape[0]}"
    aug_cols = 2 * n
    aug_bits = np.zeros((n, aug_cols), dtype=np.uint8)
    aug_bits[:, :n] = unpack_bits(A, n)
    aug_bits[:, n:] = np.eye(n, dtype=np.uint8)
    R, piv = rref(pack_bits(aug_bits), aug_cols)
    if len(piv) != n or any(p >= n for p in piv):
        return None
    return pack_bits(unpack_bits(R, aug_cols)[:, n:])

def nullspace_basis(A: np.ndarray, n_cols: int) -> np.ndarray:
    R, piv = rref(A, n_cols)
    Rbits = unpack_bits(R, n_cols)
    pivset = set(piv)
    free = [c for c in range(n_cols) if c not in pivset]
    nwords = (n_cols + 63) // 64
    if not free:
        return np.zeros((0, nwords), dtype=np.uint64)
    basis = np.zeros((len(free), n_cols), dtype=np.uint8)
    for bi, f in enumerate(free):
        v = basis[bi]
        v[f] = 1
        for row_idx, p in enumerate(piv):
            v[p] = Rbits[row_idx, f]
    return pack_bits(basis)