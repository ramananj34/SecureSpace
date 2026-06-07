import sys
import time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
import nullspace.gf2 as gf2
from nullspace.ldpc_ops import LDPCCode

def pack_sparse(M, n_cols: int) -> np.ndarray:
    coo = M.tocoo()
    nwords = (n_cols + 63) // 64
    P = np.zeros((coo.shape[0], nwords), dtype=np.uint64)
    np.bitwise_or.at(P, (coo.row.astype(np.intp), (coo.col >> 6).astype(np.intp)), np.uint64(1) << (coo.col & 63).astype(np.uint64))
    return P

def gpu_gf2_rank(A_packed_cpu: np.ndarray, n_cols: int, xp) -> int:
    A = xp.asarray(A_packed_cpu)
    nrows = A.shape[0]
    U1 = xp.uint64(1)
    r = 0
    for c in range(n_cols):
        if r >= nrows:
            break
        w = c >> 6
        bit = xp.uint64(c & 63)
        colbits = (A[r:, w] >> bit) & U1
        nz = xp.nonzero(colbits)[0]
        if nz.size == 0:
            continue
        pr = r + int(nz[0])
        if pr != r:
            A[[r, pr]] = A[[pr, r]]
        allbits = (A[:, w] >> bit) & U1
        allbits[r] = 0
        targets = xp.nonzero(allbits)[0]
        if targets.size:
            A[targets] ^= A[r]
        r += 1
    return int(r)

def main():
    try:
        import cupy as xp
        dev = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        backend = f"cupy / {dev}"
        on_gpu = True
    except Exception as exc:
        xp = np
        backend = f"numpy CPU fallback (cupy unavailable: {exc})"
        on_gpu = False
    print(f"Backend: {backend}")
    code = LDPCCode.dvbs2_long_rate12()
    n, k, m = code.n, code.k, code.m
    print(f"[{code.name}]  n={n} k={k} m={m}")
    Hp = pack_sparse(code.H, n)
    print(f"  H packed: {Hp.shape} uint64, {Hp.nbytes / 1e6:.0f} MB")
    print("  running GF(2) Gauss-Jordan rank(H)...")
    t0 = time.time()
    if on_gpu:
        r = gpu_gf2_rank(Hp, n, xp)
        xp.cuda.Stream.null.synchronize()
    else:
        r = gf2.rank(Hp, n)
    dt = time.time() - t0
    print(f"  GE rank(H) = {r}  ({dt:.1f}s)")
    assert r == m, f"rank(H)={r} != n-k={m}  -- production H is NOT full row rank!"
    print(f"  rank(H) = n-k = {m}  ->  dim N(H) = n - rank(H) = {n - r} = k = {k}")
    if on_gpu:
        short = LDPCCode.dvbs2_short_rate12()
        Sp = pack_sparse(short.H, short.n)
        r_gpu = gpu_gf2_rank(Sp, short.n, xp)
        r_cpu = gf2.rank(Sp, short.n)
        assert r_gpu == r_cpu == short.m, f"GPU/CPU short-frame rank mismatch: {r_gpu} vs {r_cpu}"
        print(f"  GPU/CPU agree on short-frame rank(H)={r_gpu} (=n-k {short.m})")
    print("\nLong-frame GE rank confirmation PASSED.")

if __name__ == "__main__":
    main()