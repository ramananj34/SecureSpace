import sys
import time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
import nullspace_attack.gf2 as gf2
from nullspace_attack.ldpc_ops import LDPCCode

def pack_sparse(M, n_cols: int) -> np.ndarray:
    coo = M.tocoo()
    nwords = (n_cols + 63) // 64
    P = np.zeros((coo.shape[0], nwords), dtype=np.uint64)
    np.bitwise_or.at(P, (coo.row.astype(np.intp), (coo.col >> 6).astype(np.intp)), np.uint64(1) << (coo.col & 63).astype(np.uint64))
    return P

def main():
    code = LDPCCode.dvbs2_long_rate12()
    n, k, m = code.n, code.k, code.m
    print(f"[{code.name}]  n={n} k={k} m={m}")
    Hp = pack_sparse(code.H, n)
    print(f"  H packed: {Hp.shape} uint64, {Hp.nbytes / 1e6:.0f} MB")
    print(f"  running Gauss-Jordan rank(H) (~1e12 word-ops; this is the slow one)...")
    t0 = time.time()
    r = gf2.rank(Hp, n)
    dt = time.time() - t0
    print(f"  GE rank(H) = {r}  ({dt/60:.1f} min)")
    assert r == m, f"rank(H)={r} != n-k={m}  -- production H is NOT full row rank!"
    print(f"  rank(H) = n-k = {m}  ->  dim N(H) = n - rank(H) = {n - r} = k = {k}")
    print("\nLong-frame GE rank confirmation PASSED.")

if __name__ == "__main__":
    main()