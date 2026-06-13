from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import nullspace_attack_utils.gf2 as gf2
from secret_integrity import SecretIntegrity, nwords

__all__ = ["CombinedDefense", "identity_packed", "nullspace_basis_fast", "HEADER_BITS", "SLOT_BITS", "N_SLOTS"]
_U1 = np.uint64(1)
HEADER_BITS = 400
SLOT_BITS = 800
N_SLOTS = 40

def _u(x) -> np.uint64:
    return np.uint64(int(x))

def identity_packed(k: int) -> np.ndarray:
    nwk = nwords(k)
    I = np.zeros((k, nwk), dtype=np.uint64)
    idx = np.arange(k)
    I[idx, idx >> 6] |= (_U1 << (idx & 63).astype(np.uint64))
    return I

def nullspace_basis_fast(A_packed: np.ndarray, n_cols: int) -> np.ndarray:
    R, piv = gf2.rref(A_packed, n_cols)
    Rbits = np.atleast_2d(gf2.unpack_bits(R, n_cols))
    pivset = set(piv)
    free = [c for c in range(n_cols) if c not in pivset]
    if not free:
        return np.zeros((0, nwords(n_cols)), dtype=np.uint64)
    piv_arr = np.asarray(piv, dtype=np.intp)
    rank = len(piv)
    basis = np.zeros((len(free), n_cols), dtype=np.uint8)
    for bi, f in enumerate(free):
        basis[bi, f] = 1
        if rank:
            basis[bi, piv_arr] = Rbits[:rank, f]
    return gf2.pack_bits(basis)

class CombinedDefense:

    def __init__(self, code, secret, Mp, rank_Mp, Bnull, header_bits=HEADER_BITS, slot_bits=SLOT_BITS, n_slots=N_SLOTS):
        self.code = code
        self.secret = secret
        self.k, self.n, self.m = code.k, code.n, code.m
        self.Mp = Mp
        self.rank_Mp = rank_Mp
        self.Bnull = Bnull
        self.dim = Bnull.shape[0]
        self.header_bits, self.slot_bits, self.n_slots = header_bits, slot_bits, n_slots
        assert header_bits + n_slots * slot_bits == self.k, (
            f"frame layout {header_bits}+{n_slots}x{slot_bits} != k={self.k}")
        self._Bnull_bits = None

    @classmethod
    def build(cls, secret: SecretIntegrity, code, verbose: bool = True, header_bits=HEADER_BITS, slot_bits=SLOT_BITS, n_slots=N_SLOTS) -> "CombinedDefense":
        k, n, m = code.k, code.n, code.m
        assert secret.n == n, f"H' has n={secret.n}, code has n={n}"
        t0 = time.time()
        P_sys = code.materialize_P_sys(packed=True)
        G = np.vstack([identity_packed(k), P_sys])
        if verbose:
            print(f"[build] G_LDPC {G.shape} ({G.nbytes/1e6:.0f} MB)  "
                  f"({time.time()-t0:.1f}s)")
        t1 = time.time()
        Mp = gf2.gf2_matmul(secret.H_packed, G, n)
        if verbose:
            print(f"[build] M' = H' G_LDPC {Mp.shape}  ({time.time()-t1:.1f}s)")
        t2 = time.time()
        Bnull = nullspace_basis_fast(Mp, k)
        rank_Mp = k - Bnull.shape[0]
        if verbose:
            print(f"[build] rank(M')={rank_Mp}  dim N(M')={Bnull.shape[0]}  "
                  f"(= dim N([H;H']))  ({time.time()-t2:.1f}s)")
        if verbose:
            print(f"[build] total {time.time()-t0:.1f}s")
        return cls(code, secret, Mp, rank_Mp, Bnull, header_bits=header_bits, slot_bits=slot_bits, n_slots=n_slots)

    def slot_footprint(self, slot: int):
        lo = self.header_bits + slot * self.slot_bits
        return lo, lo + self.slot_bits

    def _bnull_bits(self):
        if self._Bnull_bits is None:
            self._Bnull_bits = np.atleast_2d(gf2.unpack_bits(self.Bnull, self.k))
        return self._Bnull_bits

    def _slot_matrix(self, slot: int):
        lo, hi = self.slot_footprint(slot)
        AF = self._bnull_bits()[:, lo:hi].T
        return gf2.pack_bits(AF)

    def targetability(self, slot: int) -> int:
        return gf2.rank(self._slot_matrix(slot), self.dim)

    def project(self, window_bits: np.ndarray, slot: int):
        wb = np.asarray(window_bits, dtype=np.uint8).ravel()
        assert wb.size == self.slot_bits, f"window needs {self.slot_bits} bits, got {wb.size}"
        AF = self._slot_matrix(slot)
        c = gf2.solve(AF, wb, self.dim)
        if c is None:
            return None
        v_packed = gf2.gf2_matmul(gf2.pack_bits(c), self.Bnull, self.dim)
        v = gf2.unpack_bits(np.atleast_2d(v_packed), self.k)[0]
        delta = np.asarray(self.code.encode(v.astype(np.int8))).astype(np.uint8)
        return v, delta

    def naive_check(self, window_bits: np.ndarray, slot: int) -> int:
        lo, hi = self.slot_footprint(slot)
        v = np.zeros(self.k, dtype=np.uint8)
        v[lo:hi] = np.asarray(window_bits, dtype=np.uint8).ravel()
        s = gf2.gf2_matvec(self.Mp, gf2.pack_bits(v))
        return int(s.sum())

    def arms(self, window_bits: np.ndarray, slot: int) -> dict:
        wb = np.asarray(window_bits, dtype=np.uint8).ravel()
        lo, hi = self.slot_footprint(slot)
        rec = {"slot": int(slot), "window_weight": int(wb.sum()), "naive_H_synd": 0, "naive_Hprime_synd_weight": self.naive_check(wb, slot)}
        rec["naive_flagged_by_combined"] = bool(rec["naive_Hprime_synd_weight"] > 0)
        proj = self.project(wb, slot)
        if proj is None:
            rec.update(targetable=False, combined_recovered=False)
            return rec
        v, delta = proj
        h_synd = int(self.code.syndrome(delta).sum())
        hp_synd = int(gf2.gf2_matvec(self.secret.H_packed, gf2.pack_bits(delta)).sum())
        footprint_exact = bool(np.array_equal(v[lo:hi], wb))
        coll_weight = int(v.sum()) - int(v[lo:hi].sum())
        hb, sb = self.header_bits, self.slot_bits
        slots_touched = [s for s in range(self.n_slots) if v[hb + s*sb: hb + (s+1)*sb].any()]
        header_touched = bool(v[:hb].any())
        rec.update(targetable=True, combined_recovered=True, combined_H_synd=h_synd, combined_Hprime_synd=hp_synd, combined_passes_both=bool(h_synd == 0 and hp_synd == 0), footprint_exact=footprint_exact, info_weight_total=int(v.sum()), collateral_info_weight=coll_weight, n_slots_touched=len(slots_touched), header_touched=header_touched)
        return rec

    def verify(self, n_sample: int = 16, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.dim, size=min(n_sample, self.dim), replace=False)
        ok_h = ok_hp = True
        for j in idx:
            v = gf2.unpack_bits(np.atleast_2d(self.Bnull[j]), self.k)[0]
            delta = np.asarray(self.code.encode(v.astype(np.int8))).astype(np.uint8)
            ok_h &= (int(self.code.syndrome(delta).sum()) == 0)
            ok_hp &= (int(gf2.gf2_matvec(self.secret.H_packed, gf2.pack_bits(delta)).sum()) == 0)
        indep = (gf2.rank(self.Bnull, self.k) == self.dim)
        return {"sampled": int(idx.size), "all_pass_H": bool(ok_h), "all_pass_Hprime": bool(ok_hp), "basis_independent": bool(indep), "dim": int(self.dim)}

    def rank_stacked(self, verbose: bool = True) -> dict:
        from scipy.sparse import vstack as spvstack
        t0 = time.time()
        Hp = _pack_sparse(self.code.H, self.n)
        stacked = np.vstack([Hp, self.secret.H_packed])
        r = gf2.rank(stacked, self.n)
        dim = self.n - r
        if verbose:
            print(f"[rank_stacked] rank([H;H'])={r}  dim N([H;H'])={dim}  "
                  f"(expect {self.dim})  ({(time.time()-t0)/60:.1f} min)")
        return {"rank_stacked": int(r), "dim_stacked": int(dim), "matches_Bnull_dim": bool(dim == self.dim), "seconds": round(time.time() - t0, 1)}

def _pack_sparse(M, n_cols: int) -> np.ndarray:
    coo = M.tocoo()
    P = np.zeros((coo.shape[0], nwords(n_cols)), dtype=np.uint64)
    np.bitwise_or.at(P, (coo.row.astype(np.intp), (coo.col >> 6).astype(np.intp)), _U1 << (coo.col & 63).astype(np.uint64))
    return P

def build_long(seed: int = 20250612, m_prime: int | None = None, verbose: bool = True):
    from ldpc_ops import LDPCCode
    code = LDPCCode.dvbs2_long_rate12()
    mp = m_prime if m_prime is not None else code.n // 4
    secret = SecretIntegrity(code.n, mp, seed)
    cd = CombinedDefense.build(secret, code, verbose=verbose)
    return cd

if __name__ == "__main__":
    import json
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"], default="build", nargs="?")
    ap.add_argument("--seed", type=int, default=20250612)
    ap.add_argument("--out", default=str(_THIS / "runs_e8"))
    ap.add_argument("--rank-stacked", action="store_true", help="also run the heavy ~30min stacked rank cross-check")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cd = build_long(seed=args.seed)
    rec = {"experiment": "E8_build_Bcomb", "seed": args.seed, "n": cd.n, "k": cd.k, "m_ldpc": cd.m, "m_prime": cd.secret.m, "rank_Mprime": cd.rank_Mp, "dim_N_combined": cd.dim}
    print("\nverify B_comb subset N([H;H'])...")
    rec["verify"] = cd.verify(n_sample=16)
    print(" ", rec["verify"])
    rec["targetability"] = {str(s): cd.targetability(s) for s in (0, 20, 39)}
    print("  targetability (slots 0/20/39):", rec["targetability"])
    if args.rank_stacked:
        rec["rank_stacked_check"] = cd.rank_stacked()
    tmp = out / "e8_build.json.tmp"
    json.dump(rec, open(tmp, "w"), indent=2, default=float)
    tmp.replace(out / "e8_build.json")
    print(f"\nsaved -> {out/'e8_build.json'}")