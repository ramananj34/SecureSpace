import sys
import time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
import nullspace_attack.gf2 as gf2
from nullspace_attack.ldpc_ops import LDPCCode
from nullspace_attack.projection import (b_transpose_apply, normal_form_matrix, project_information_set)

def _time(fn, repeat=1):
    t0 = time.time()
    for _ in range(repeat):
        out = fn()
    return (time.time() - t0) / repeat, out

def profile_frame(code: LDPCCode, rng: np.random.Generator):
    n, k, m = code.n, code.k, code.m
    print(f"\n[{code.name}]  n={n} k={k} m={m}")
    infos = [rng.integers(0, 2, size=k, dtype=np.int8) for _ in range(20)]
    t_enc, _ = _time(lambda: [code.encode(i) for i in infos])
    t_enc /= len(infos)
    print(f"  encode:                 {t_enc*1e3:7.2f} ms/codeword   ({1/t_enc:6.0f} cw/s)")
    cw = code.encode(infos[0])
    t_ext, _ = _time(lambda: code.info_bit_extract(cw), repeat=1000)
    print(f"  info_bit_extract:       {t_ext*1e6:7.2f} us/call")
    B = 100
    C = np.stack([code.encode(i) for i in [rng.integers(0, 2, size=k, dtype=np.int8) for _ in range(B)]], axis=1).astype(np.int64)
    t_syn, _ = _time(lambda: (code.H @ C) % 2)
    print(f"  syndrome (batch {B}):    {t_syn*1e3:7.2f} ms   ({B/t_syn:6.0f} words/s)")
    t_mat, P_packed = _time(lambda: code.materialize_P_sys(packed=True))
    print(f"  materialize_P_sys:      {t_mat*1e3:7.2f} ms   (packed {P_packed.nbytes/1e6:.0f} MB)")
    dp = rng.integers(0, 2, size=n, dtype=np.int8)
    t_bt, _ = _time(lambda: b_transpose_apply(code, dp, P_packed), repeat=10)
    print(f"  B^T delta' (rhs):       {t_bt*1e3:7.2f} ms")
    t_ps, _ = _time(lambda: code.project_systematic(dp), repeat=5)
    print(f"  project_systematic:     {t_ps*1e3:7.2f} ms   (cheap default; O(encode), no GE)")
    return P_packed

def profile_ge_scaling(rng):
    print("\nGE scaling (rref / solve / inverse), random dense F2:")
    print(f"  {'dim':>6}  {'rref(s)':>9}  {'solve(s)':>9}  {'inverse(s)':>11}")
    for nsz in (256, 512, 1024, 2048):
        A = rng.integers(0, 2, size=(nsz, nsz), dtype=np.uint8)
        Ap = gf2.pack_bits(A)
        b = rng.integers(0, 2, size=nsz, dtype=np.uint8)
        t_r, _ = _time(lambda: gf2.rref(Ap, nsz))
        t_s, _ = _time(lambda: gf2.solve(Ap, b, nsz))
        t_i, _ = _time(lambda: gf2.inverse(Ap, nsz))
        print(f"  {nsz:>6}  {t_r:>9.3f}  {t_s:>9.3f}  {t_i:>11.3f}")

def profile_expensive_short(code_short, P_short, code_long, rng):
    ks, kl = code_short.k, code_long.k
    ms, ml = code_short.m, code_long.m
    print("\nNormal-form fallback (forming M = I + P_sys^T P_sys), short frame:")
    t0 = time.time()
    M_packed = normal_form_matrix(code_short, P_short)
    t_form = time.time() - t0
    M_rank = gf2.rank(M_packed, ks)
    singular = M_rank < ks
    print(f"  form M (k={ks}): {t_form:.1f}s ;  rank(M) = {M_rank}/{ks} -> "
          f"{'SINGULAR over F2' if singular else 'nonsingular'}")
    factor = (kl * kl * ml) / (ks * ks * ms)
    print(f"  long-frame form M extrapolation: ~{t_form*factor/60:.0f} min "
          f"(O(k^2 m), {factor:.0f}x) -> PROHIBITIVE; fallback-1 deprioritized (§6.13)")
    if singular:
        print(f"  => B^T B is degenerate over F2 (§6.13): normal-form returns None here;")
        print(f"     this is a first data point that fallback-1 is unreliable, as warned.")
    print("\nGeneral information-set projection (k x k solve), per-call cost:")
    A = rng.integers(0, 2, size=(ks, ks), dtype=np.uint8)
    Ap = gf2.pack_bits(A)
    b = rng.integers(0, 2, size=ks, dtype=np.uint8)
    t_solve, _ = _time(lambda: gf2.solve(Ap, b, ks))
    print(f"  short per-call (k={ks} solve proxy): {t_solve:.1f}s")
    factor = (kl * kl * kl) / (ks * ks * ks)
    print(f"  long per-call extrapolation: ~{t_solve*factor/60:.0f} min "
          f"(O(k^3), {factor:.0f}x) -> PROHIBITIVE per Algorithm-1 iteration")

def main():
    rng = np.random.default_rng(0)
    short = LDPCCode.dvbs2_short_rate12()
    long = LDPCCode.dvbs2_long_rate12()
    P_short = profile_frame(short, rng)
    _ = profile_frame(long, rng)
    profile_ge_scaling(rng)
    profile_expensive_short(short, P_short, long, rng)
    print("\n" + "=" * 70)
    print("Week-4 budgeting implications (long frame):")
    print("  * project_systematic is the ONLY cheap projection (~tens of ms).")
    print("  * general info-set decoding and normal-form are ~k^3 / O(k^2 m)")
    print("    per call -> NOT usable per Algorithm-1 iteration on the long frame.")
    print("  * Attack on long should use project_systematic + the precomputed")
    print("    low-weight codeword library (ldpc/codeword_enumeration.py), not")
    print("    per-iteration GE. Info-set / normal-form remain short-frame tools.")
    print("=" * 70)
    print("\nProfile complete.")

if __name__ == "__main__":
    main()