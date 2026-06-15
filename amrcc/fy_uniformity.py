from __future__ import annotations
import sys
import math
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_AMRCC = _THIS
_ROOT = _AMRCC.parent
for _p in [str(_ROOT), str(_AMRCC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import chacha20_keystream, fisher_yates

KEY = bytes(range(32))

def _fy_L(keystream: bytes, k: int, L_bytes: int) -> np.ndarray:
    perm = np.arange(k, dtype=np.int64)
    pos = 0
    for i in range(k - 1, 0, -1):
        r = int.from_bytes(keystream[pos:pos + L_bytes], "little")
        pos += L_bytes
        j = r % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def _perm_rank(perm) -> int:
    avail = list(range(len(perm)))
    rank = 0
    for i, p in enumerate(perm):
        idx = avail.index(int(p))
        rank = rank * (len(perm) - i) + idx
        avail.pop(idx)
    return rank

def small_k_tvd(k: int = 6, M: int = 200_000, L_bytes: int = 8) -> dict:
    kfac = math.factorial(k)
    counts = np.zeros(kfac, dtype=np.int64)
    for t in range(M):
        ks = chacha20_keystream(KEY, t, k * L_bytes)
        counts[_perm_rank(_fy_L(ks, k, L_bytes))] += 1
    p = counts / M
    tvd = 0.5 * np.abs(p - 1.0 / kfac).sum()
    noise_floor = math.sqrt((kfac - 1) / (2.0 * math.pi * M))
    return {"k": k, "k_factorial": kfac, "M": M, "L_bits": 8 * L_bytes, "tvd": float(tvd), "approx_noise_floor": float(noise_floor)}

def production_marginals(k: int = 32_400, M: int = 4_000, positions=(0, 16_200, 32_399), nbins: int = 40) -> dict:
    perms = np.empty((M, len(positions)), dtype=np.int64)
    fixed = 0
    for t in range(M):
        ks = chacha20_keystream(KEY, t, k * 8)
        perm = fisher_yates(ks, k)
        perms[t] = perm[list(positions)]
        fixed += int(np.count_nonzero(perm == np.arange(k)))
    out = {"k": k, "M": M, "positions": list(positions), "nbins": nbins, "by_position": []}
    edges = np.linspace(0, k, nbins + 1)
    exp = M / nbins
    for c in range(len(positions)):
        hist, _ = np.histogram(perms[:, c], bins=edges)
        chi2 = float(((hist - exp) ** 2 / exp).sum())
        out["by_position"].append({"position": int(positions[c]), "mean": float(perms[:, c].mean()), "theory_mean": (k - 1) / 2.0, "chi2": chi2, "dof": nbins - 1})
    out["fixed_points_mean"] = fixed / M
    out["fixed_points_theory"] = 1.0
    return out

def _predicted_mod_tvd(L_bits: int, m: int) -> float:
    twoL = 1 << L_bits
    base = twoL // m
    rem = twoL % m
    hi = (base + 1) / twoL
    loo = base / twoL
    u = 1.0 / m
    return 0.5 * (rem * abs(hi - u) + (m - rem) * abs(loo - u))

def modulo_bias_demo(pairs=((8, 100), (16, 25_000)), n_draws: int = 4_000_000, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    rows = []
    for L_bits, m in pairs:
        vals = rng.integers(0, 1 << L_bits, size=n_draws, dtype=np.uint64) % np.uint64(m)
        hist = np.bincount(np.asarray(vals, dtype=np.int64), minlength=m).astype(np.float64)
        p = hist / n_draws
        emp = 0.5 * np.abs(p - 1.0 / m).sum()
        rows.append({"L_bits": L_bits, "m": m, "n_draws": n_draws, "tvd_empirical": float(emp), "tvd_predicted": _predicted_mod_tvd(L_bits, m)})
    return rows

def union_bound_table(k: int = 32_400, L_list=(32, 64)) -> list:
    return [{"L_bits": L, "k": k, "tvd_bound": (k * k) / (2.0 ** L), "tvd_bound_log2": math.log2((k * k) / (2.0 ** L))} for L in L_list]

def _print_report():
    print("=" * 72)
    print("FISHER-YATES UNIFORMITY  (D.6.11: L=64 vs L=32)")
    print("=" * 72)
    print("\n[1] small-k full-distribution TVD @ L=64")
    r = small_k_tvd()
    print(f"    k={r['k']} (k!={r['k_factorial']}), M={r['M']}, L={r['L_bits']} bits")
    print(f"    TVD = {r['tvd']:.5f}   (noise floor ~ {r['approx_noise_floor']:.5f})"
          f"  -> {'UNIFORM' if r['tvd'] < 3 * r['approx_noise_floor'] else 'CHECK'}")
    print("\n[2] production-k position marginals @ L=64")
    r = production_marginals()
    crit = 73.0
    for bp in r["by_position"]:
        flag = "ok" if bp["chi2"] < crit else "CHECK"
        print(f"    pos {bp['position']:6d}: mean={bp['mean']:9.1f} (theory {bp['theory_mean']:.1f})"
              f"  chi2={bp['chi2']:6.1f}/{bp['dof']}  [{flag}]")
    print(f"    fixed points/perm: {r['fixed_points_mean']:.4f} (theory {r['fixed_points_theory']:.1f})")
    print("\n[3] modulo-bias mechanism  (bias ~ m / 2^L)")
    for row in modulo_bias_demo():
        print(f"    L={row['L_bits']:2d} bits, m={row['m']:6d}: "
              f"TVD emp={row['tvd_empirical']:.5f}  pred={row['tvd_predicted']:.5f}")
    print("\n    => per-permutation union bound  TVD <= k^2 / 2^L  at k=32,400:")
    for row in union_bound_table():
        print(f"       L={row['L_bits']:2d}: TVD <= {row['tvd_bound']:.3e}  (2^{row['tvd_bound_log2']:.1f})")
    print("\n    L=32 -> ~0.245 (unacceptable);  L=64 -> ~2^-34 (folded into eps_PRG).  L=64 chosen.")
    print("=" * 72)

if __name__ == "__main__":
    _print_report()