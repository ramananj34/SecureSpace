from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import nullspace_attack_utils.gf2 as gf2
from theorem_1.secret_integrity import (SecretIntegrity, nwords, basis_vec_packed, random_matrix_packed, recover_readout, recover_inverse, recover_general, witnesses_at, nonuniqueness_log2_count, qpochhammer_span_prob)

N, M, SEED = 200, 60, 20250612

def _dense(secret):
    return gf2.unpack_bits(secret.H_packed, secret.n)

def test_shape_padding_rank():
    s = SecretIntegrity(N, M, SEED)
    assert s.H_packed.shape == (M, nwords(N))
    last_valid = N - 64 * (nwords(N) - 1)
    if last_valid < 64:
        garbage = s.H_packed[:, -1] >> np.uint64(last_valid)
        assert int(garbage.sum()) == 0
    assert s.full_row_rank()
    assert np.array_equal(s.H_packed, SecretIntegrity(N, M, SEED).H_packed)

def test_column_equals_oracle():
    s = SecretIntegrity(N, M, SEED)
    for i in [0, 1, 63, 64, 65, 127, 128, 199]:
        via_readout = s.column(i)
        via_oracle = s.measure_vec(basis_vec_packed(i, N))
        assert np.array_equal(via_readout, via_oracle), f"mismatch at column {i}"

def test_measure_batch_matches_columnwise():
    s = SecretIntegrity(N, M, SEED)
    rng = np.random.default_rng(7)
    cols = [3, 10, 64, 130, 199]
    Xb = np.zeros((N, len(cols)), dtype=np.uint8)
    Xb[cols, np.arange(len(cols))] = 1
    S = gf2.unpack_bits(s.measure(gf2.pack_bits(Xb)), len(cols))
    for j, c in enumerate(cols):
        assert np.array_equal(S[:, j], s.column(c))
    t = 17
    X = random_matrix_packed(N, t, rng)
    S = np.atleast_2d(gf2.unpack_bits(s.measure(X), t))
    Xb = np.atleast_2d(gf2.unpack_bits(X, t))
    for j in range(t):
        xj = gf2.pack_bits(Xb[:, j])
        assert np.array_equal(S[:, j], s.measure_vec(xj))

def test_recover_readout_exact_full():
    s = SecretIntegrity(N, M, SEED)
    Hrec = recover_readout(s, range(N))
    assert np.array_equal(Hrec, s.H_packed)

def test_recover_readout_partial_leaves_free():
    s = SecretIntegrity(N, M, SEED)
    t = N - 5
    Hrec = recover_readout(s, range(t))
    Hd, Rd = _dense(s), gf2.unpack_bits(Hrec, N)
    assert np.array_equal(Rd[:, :t], Hd[:, :t])
    assert int(Rd[:, t:].sum()) == 0

def test_recover_inverse_square_exact():
    s = SecretIntegrity(N, M, SEED)
    rng = np.random.default_rng(11)
    tries = 0
    while True:
        tries += 1
        X = random_matrix_packed(N, N, rng)
        Hrec, ok = recover_inverse(s, X, N)
        if ok:
            break
        assert tries < 50
    assert np.array_equal(Hrec, s.H_packed)

def test_recover_inverse_singular_returns_none():
    s = SecretIntegrity(N, M, SEED)
    rng = np.random.default_rng(13)
    Xb = np.atleast_2d(gf2.unpack_bits(random_matrix_packed(N, N, rng), N))
    Xb[:, 1] = Xb[:, 0]
    Hrec, ok = recover_inverse(s, gf2.pack_bits(Xb), N)
    assert (Hrec is None) and (ok is False)

def test_recover_general_overdetermined():
    s = SecretIntegrity(N, M, SEED)
    rng = np.random.default_rng(17)
    Hrec, ok = None, False
    for _ in range(50):
        X = random_matrix_packed(N, N + 8, rng)
        Hrec, ok = recover_general(s, X, N + 8)
        if ok:
            break
    assert ok and np.array_equal(Hrec, s.H_packed)

def test_nonuniqueness_witnesses():
    s = SecretIntegrity(N, M, SEED)
    t = N - 1
    A, B, observed, log2c = witnesses_at(s, t)
    assert log2c == nonuniqueness_log2_count(t, N, M) == M * 1
    for i in [0, 50, 100, t - 1]:
        ei = basis_vec_packed(i, N)
        assert np.array_equal(gf2.gf2_matvec(A, ei), gf2.gf2_matvec(B, ei))
    eu = basis_vec_packed(t, N)
    assert not np.array_equal(gf2.gf2_matvec(A, eu), gf2.gf2_matvec(B, eu))
    Ad, Bd = gf2.unpack_bits(A, N), gf2.unpack_bits(B, N)
    diff_cols = np.where((Ad ^ Bd).any(axis=0))[0]
    assert diff_cols.tolist() == [t]

def test_nonuniqueness_witnesses_r_general():
    s = SecretIntegrity(N, M, SEED)
    for r in [1, 3, 7]:
        t = N - r
        A, B, observed, log2c = witnesses_at(s, t, alter_index=N - 1)
        assert log2c == M * r
        assert observed.size == t

def test_unit_lower_triangular_invertible():
    from theorem_1.secret_integrity import random_unit_lower_triangular_packed
    rng = np.random.default_rng(3)
    for n in [16, 63, 64, 65, 100, 200]:
        X = random_unit_lower_triangular_packed(n, rng)
        Xb = gf2.unpack_bits(X, n)
        assert np.array_equal(np.diag(Xb), np.ones(n, dtype=np.uint8))
        assert int(np.triu(Xb, 1).sum()) == 0
        assert gf2.inverse(X, n) is not None
        s = SecretIntegrity(n, max(1, n // 4), 7)
        Hrec, ok = recover_inverse(s, X, n)
        assert ok and np.array_equal(Hrec, s.H_packed)

def test_qpochhammer_formula_vs_empirical():
    assert abs(qpochhammer_span_prob(64800, 64800) - 0.288788) < 1e-6
    assert abs(qpochhammer_span_prob(20, 20) - 0.288788) < 1e-5
    for c in range(0, 8):
        p = qpochhammer_span_prob(20, 20 + c)
        assert (1.0 - p) <= 2.0 ** (-c) + 1e-12
    n_small, trials = 16, 4000
    s = SecretIntegrity(n_small, n_small // 2 or 1, SEED)
    rng = np.random.default_rng(99)
    for c in [0, 2]:
        t = n_small + c
        spans = 0
        recovered_ok_implies_exact = True
        for _ in range(trials):
            X = random_matrix_packed(n_small, t, rng)
            full = gf2.rank(X, t) == n_small
            spans += int(full)
            if full and (_ % 23 == 0):
                Hrec, ok = recover_general(s, X, t)
                if not (ok and np.array_equal(Hrec, s.H_packed)):
                    recovered_ok_implies_exact = False
        emp = spans / trials
        exact = qpochhammer_span_prob(n_small, t)
        assert abs(emp - exact) < 0.04, f"t=n+{c}: emp {emp:.3f} vs exact {exact:.3f}"
        assert recovered_ok_implies_exact

_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

def main():
    print(f"secret_integrity tests  (n={N}, m={M})\n" + "-" * 52)
    fails = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            fails += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print("-" * 52)
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())