from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import nullspace_attack_utils.gf2 as gf2

__all__ = ["SecretIntegrity", "nwords", "basis_vec_packed", "random_matrix_packed", "random_unit_lower_triangular_packed", "recover_readout", "recover_inverse", "recover_general", "witnesses_at", "nonuniqueness_log2_count", "qpochhammer_span_prob"]
_U1 = np.uint64(1)

def _u(x) -> np.uint64:
    return np.uint64(int(x))

def nwords(n: int) -> int:
    return (int(n) + 63) // 64

def _mask_last_word(packed: np.ndarray, n_cols: int) -> np.ndarray:
    nw = packed.shape[-1]
    valid = int(n_cols) - 64 * (nw - 1)
    if valid < 64:
        packed[..., -1] &= (_U1 << _u(valid)) - _U1
    return packed

def basis_vec_packed(i: int, n: int) -> np.ndarray:
    e = np.zeros(n, dtype=np.uint8)
    e[int(i)] = 1
    return gf2.pack_bits(e)

def random_matrix_packed(nrows: int, ncols: int, rng: np.random.Generator) -> np.ndarray:
    nw = nwords(ncols)
    M = rng.integers(0, 1 << 64, size=(nrows, nw), dtype=np.uint64)
    _mask_last_word(M, ncols)
    return M

def random_unit_lower_triangular_packed(n: int, rng: np.random.Generator) -> np.ndarray:
    X = random_matrix_packed(n, n, rng)
    for r in range(n):
        w, b = r >> 6, r & 63
        if w + 1 < X.shape[1]:
            X[r, w + 1:] = 0
        keep = ((1 << (b + 1)) - 1) & ((1 << 64) - 1)
        X[r, w] = (X[r, w] & np.uint64(keep)) | (_U1 << _u(b))
    return X

class SecretIntegrity:
    
    def __init__(self, n: int, m: int, seed: int):
        self.n = int(n)
        self.m = int(m)
        self.seed = int(seed)
        self.H_packed = random_matrix_packed(self.m, self.n, np.random.default_rng(self.seed))

    def measure(self, X_packed: np.ndarray) -> np.ndarray:
        return gf2.gf2_matmul(self.H_packed, X_packed, self.n)

    def measure_vec(self, x_packed: np.ndarray) -> np.ndarray:
        return gf2.gf2_matvec(self.H_packed, x_packed)

    def column(self, i: int) -> np.ndarray:
        return ((self.H_packed[:, int(i) >> 6] >> _u(int(i) & 63)) & _U1).astype(np.uint8)

    def full_row_rank(self) -> bool:
        return gf2.rank(self.H_packed, self.n) == self.m

def recover_readout(secret: SecretIntegrity, queried_indices) -> np.ndarray:
    n, m = secret.n, secret.m
    Hrec = np.zeros((m, nwords(n)), dtype=np.uint64)
    for i in queried_indices:
        i = int(i)
        col = secret.column(i).astype(np.uint64)
        Hrec[:, i >> 6] |= col << _u(i & 63)
    return Hrec

def recover_inverse(secret: SecretIntegrity, X_packed: np.ndarray, t: int):
    n = secret.n
    assert int(t) == n, f"recover_inverse needs t == n == {n}, got t={t}"
    S = secret.measure(X_packed)
    Xinv = gf2.inverse(X_packed, n)
    if Xinv is None:
        return None, False
    Hrec = gf2.gf2_matmul(S, Xinv, n)
    return Hrec, True

def recover_general(secret: SecretIntegrity, X_packed: np.ndarray, t: int):
    n = secret.n
    R, piv = gf2.rref(X_packed, int(t))
    if len(piv) < n:
        return None, False
    sel = list(piv[:n])
    Xbits = np.atleast_2d(gf2.unpack_bits(X_packed, int(t)))
    Sbits = np.atleast_2d(gf2.unpack_bits(secret.measure(X_packed), int(t)))
    Xsub = gf2.pack_bits(Xbits[:, sel])
    Ssub = gf2.pack_bits(Sbits[:, sel])
    Xinv = gf2.inverse(Xsub, n)
    if Xinv is None:
        return None, False
    return gf2.gf2_matmul(Ssub, Xinv, n), True

def nonuniqueness_log2_count(t: int, n: int, m: int) -> int:
    return int(m) * (int(n) - int(t))

def witnesses_at(secret: SecretIntegrity, t: int, alter_index: int | None = None):
    n, m = secret.n, secret.m
    r = n - int(t)
    assert r >= 1, "t < n required for non-uniqueness"
    if alter_index is None:
        alter_index = int(t)
    alter_index = int(alter_index)
    assert alter_index >= t, "altered column must be unobserved (index >= t)"
    A = secret.H_packed
    B = A.copy()
    B[0, alter_index >> 6] ^= (_U1 << _u(alter_index & 63))
    observed = np.arange(int(t), dtype=np.intp)
    return A, B, observed, nonuniqueness_log2_count(t, n, m)

def qpochhammer_span_prob(n: int, t: int) -> float:
    n, t = int(n), int(t)
    if t < n:
        return 0.0
    logp = 0.0
    for i in range(n):
        e = t - i
        logp += np.log1p(-(2.0 ** (-e)))
    return float(np.exp(logp))