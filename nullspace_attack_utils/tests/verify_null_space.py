import sys
import time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
import nullspace_attack_utils.gf2 as gf2
from nullspace_attack_utils.ldpc_ops import LDPCCode

def pack_sparse(M, n_cols: int) -> np.ndarray:
    coo = M.tocoo()
    nwords = (n_cols + 63) // 64
    P = np.zeros((coo.shape[0], nwords), dtype=np.uint64)
    np.bitwise_or.at(P, (coo.row.astype(np.intp), (coo.col >> 6).astype(np.intp)), np.uint64(1) << (coo.col & 63).astype(np.uint64))
    return P

def exhaustive_hb_zero(code: LDPCCode, chunk: int = 1000):
    n, k, H = code.n, code.k, code.H
    bad = 0
    info_ok = True
    t0 = time.time()
    i = 0
    while i < k:
        cols = list(range(i, min(i + chunk, k)))
        C = np.zeros((n, len(cols)), dtype=np.int8)
        for j, ci in enumerate(cols):
            cw = code.nullspace_column(ci)
            C[:, j] = cw
            if cw[ci] != 1 or int(cw[:k].sum()) != 1:
                info_ok = False
        S = (H @ C.astype(np.int64)) % 2
        bad += int(np.count_nonzero(S))
        i += chunk
    return bad, info_ok, time.time() - t0

def rank_h_structural(code: LDPCCode):
    n, k, m = code.n, code.k, code.m
    coo = code.H[:, k:].tocoo()
    rows, cols = coo.row, coo.col
    is_diag = cols == rows
    is_sub = cols == rows - 1
    invertible = bool(np.all(is_diag | is_sub) and int(is_diag.sum()) == m and int(is_sub.sum()) == m - 1 and coo.nnz == 2 * m - 1)
    return invertible

def route_b_ge(code: LDPCCode):
    n, k, m = code.n, code.k, code.m
    Hp = pack_sparse(code.H, n)
    t0 = time.time()
    r = gf2.rank(Hp, n)
    t_rank = time.time() - t0
    Bp = gf2.nullspace_basis(Hp, n)
    nullity = Bp.shape[0]
    cw_ok = all(not gf2.gf2_matvec(Hp, Bp[row], n).any() for row in range(nullity))
    basis_indep = gf2.rank(Bp, n) == nullity
    return dict(rank=r, rank_ok=(r == m), nullity=nullity, nullity_ok=(nullity == k), cw_ok=cw_ok, basis_indep=basis_indep, t_rank=t_rank)

def main():
    results = {}
    short = LDPCCode.dvbs2_short_rate12()
    long = LDPCCode.dvbs2_long_rate12()
    for code in (short, long):
        n, k, m = code.n, code.k, code.m
        print(f"[{code.name}]  n={n} k={k} m={m}")
        bad, info_ok, dt = exhaustive_hb_zero(code)
        assert bad == 0, f"{bad} nonzero syndrome entries across B columns"
        assert info_ok, "some B column has info part != e_i"
        print(f"  HB=0 exhaustive: all {k} columns are codewords; "
              f"info parts = I_k  ({dt:.1f}s)")
        inv = rank_h_structural(code)
        assert inv, "parity block is not an invertible staircase"
        print(f"  rank(H)=n-k={m} (staircase invertible) -> dim N(H)=k={k}")
        results[code.name] = dict(hb=bad == 0, ik=info_ok, struct=inv)
    print(f"\n[{short.name}] GE cross-check (independent rank / null space):")
    ge = route_b_ge(short)
    assert ge["rank_ok"] and ge["nullity_ok"] and ge["cw_ok"] and ge["basis_indep"]
    print(f"  GE rank(H)={ge['rank']} (=n-k {short.m}); "
          f"nullity={ge['nullity']} (=k {short.k}); "
          f"all {ge['nullity']} basis vecs are codewords; basis rank={ge['nullity']}")
    print(f"  -> Route-A (encoder) and Route-B (GE) are both bases of the same "
          f"{short.k}-dim N(H)  (GE rank: {ge['t_rank']:.1f}s)")
    s_ops = short.m * short.m * ((short.n + 63) // 64)
    l_ops = long.m * long.m * ((long.n + 63) // 64)
    est = ge["t_rank"] * (l_ops / s_ops)
    print(f"\nCalibrated long-frame GE rank estimate: ~{est/60:.0f} min single-core "
          f"(scale factor {l_ops/s_ops:.0f}x). Run verify_nullspace_long_ge.py "
          f"on the laptop, or on the cluster for a fast version.")
    print("\nVerification summary")
    print(f"  {'frame':<18}{'HB=0':<8}{'rank(B)=k':<12}{'rank(H)=n-k':<14}{'dim N(H)=k'}")
    for name in (short.name, long.name):
        r = results[name]
        mark = lambda b: "OK" if b else "FAIL"
        print(f"  {name:<18}{mark(r['hb']):<8}{mark(r['ik']):<12}"
              f"{mark(r['struct']):<14}{mark(r['struct'])}")
    print(f"  {short.name:<18}GE: rank(H)={short.m} OK, nullity={short.k} OK, "
          f"basis subset N(H) OK")
    print(f"  {long.name:<18}GE: see verify_nullspace_long_ge.py")
    print("\nDay-4 verification PASSED (long-frame GE pending separate run).")

if __name__ == "__main__":
    main()