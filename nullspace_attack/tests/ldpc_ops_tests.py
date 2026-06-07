import sys
import time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
from nullspace_attack.ldpc_ops import LDPCCode, p_sys_column

def _representative_indices(k: int, M: int = 360) -> list[int]:
    cand = [0, 1, M - 1, M, M + 1, k // 4, k // 2, 3 * k // 4, k - 2, k - 1]
    return sorted({i for i in cand if 0 <= i < k})

def _ok(name: str):
    print(f"    PASS  {name}")

def check_code(code: LDPCCode, rng: np.random.Generator):
    n, k, m = code.n, code.k, code.m
    print(f"  [{code.name}]  n={n} k={k} m={m}")
    info = rng.integers(0, 2, size=k, dtype=np.int8)
    cw = code.encode(info)
    assert cw.shape == (n,), f"codeword shape {cw.shape}"
    assert np.array_equal(cw[:k], info), "encode not systematic (cw[:k] != info)"
    assert code.is_codeword(cw), "encoded word fails syndrome check"
    _ok("encode is systematic and lands in N(H)")
    assert np.array_equal(code.info_bit_extract(cw), info), "info_bit_extract roundtrip"
    assert np.array_equal(code.parity_of(info), cw[k:]), "parity_of != cw[k:]"
    _ok("info_bit_extract / parity_of consistent with encode")
    a = rng.integers(0, 2, size=k, dtype=np.int8)
    b = rng.integers(0, 2, size=k, dtype=np.int8)
    assert np.array_equal(code.apply_B(a), code.encode(a)), "apply_B != encode"
    lhs = code.apply_B((a ^ b).astype(np.int8))
    rhs = (code.apply_B(a) ^ code.apply_B(b)).astype(np.int8)
    assert np.array_equal(lhs, rhs), "apply_B not linear: B(a^b) != B(a)^B(b)"
    _ok("apply_B = encode and is F2-linear")
    for i in _representative_indices(k):
        col = code.nullspace_column(i)
        assert code.is_codeword(col), f"B column {i} not a codeword"
        e = np.zeros(k, dtype=np.int8); e[i] = 1
        assert np.array_equal(col[:k], e), f"B column {i} info part != e_i"
    _ok("nullspace_column(i) is a codeword with info part e_i")
    for pos in (0, k):
        bad = cw.copy()
        bad[pos] ^= 1
        assert code.syndrome(bad).any(), f"syndrome missed flip at {pos}"
        assert not code.is_codeword(bad), f"is_codeword wrong at {pos}"
    _ok("syndrome / is_codeword flag single-bit corruption")
    t0 = time.time()
    P_packed = code.materialize_P_sys(packed=True)
    t_mat = time.time() - t0
    assert P_packed.shape == (m, (k + 63) // 64), f"P_sys packed shape {P_packed.shape}"
    for i in _representative_indices(k):
        ref = code.encode(np.eye(1, k, i, dtype=np.int8).ravel())[k:]
        got = p_sys_column(P_packed, i)
        assert np.array_equal(got, ref), f"P_sys column {i} != encode(e_i)[k:]"
    print(f"        materialize_P_sys: {t_mat:.3f}s, packed {P_packed.nbytes/1e6:.0f} MB")
    _ok("materialize_P_sys = prefix-XOR(H_info) matches encoder column-by-column")
    dprime = rng.integers(0, 2, size=n, dtype=np.int8)
    proj = code.project_systematic(dprime)
    assert code.is_codeword(proj), "project_systematic output not a codeword"
    assert np.array_equal(proj[:k], dprime[:k]), "project_systematic changed info bits"
    assert np.array_equal(proj, code.apply_B(dprime[:k])), "project != apply_B(info)"
    assert np.array_equal(proj[k:], code.parity_of(dprime[:k])), "parity not recomputed"
    proj_cw = code.project_systematic(cw)
    assert np.array_equal(proj_cw, cw), "project_systematic not identity on codewords"
    _ok("project_systematic: exact codeword, info-preserving, idempotent on codewords")

def main():
    rng = np.random.default_rng(0)
    print("Short frame:")
    check_code(LDPCCode.dvbs2_short_rate12(), rng)
    print("\nLong frame:")
    try:
        check_code(LDPCCode.dvbs2_long_rate12(), rng)
    except ImportError as exc:
        print(f"  SKIPPED long frame (apply Edit A first): {exc}")
    print("\nldpc_ops tests PASSED.")

if __name__ == "__main__":
    main()