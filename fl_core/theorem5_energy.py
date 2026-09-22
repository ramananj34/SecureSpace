from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "adv_subspace")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from theorem4_energy import verify_theorem4

def y_max_bound(w, delta_msb):
    return float(w) * float(delta_msb) ** 2

def row_block_bound(d_grad, b=8, delta_lsb=None):
    dl = delta_lsb if delta_lsb is not None else 2.0 ** -(b - 1)
    delta_max = (2 ** b - 1) * dl
    return float(d_grad * delta_max ** 2)


def verify_theorem5_link(Pi_Vtheta, D, g_q_bits, w, k_g, c_quant, d_adv_prime_eff,
                         eps_prg=0.0, delta_msb=1.0, d_grad=None, b=8,
                         n_mc=20000, seed=0, tol=0.20):
    base = verify_theorem4(Pi_Vtheta, D, g_q_bits, w, k_g, c_quant, d_adv_prime_eff,
                           n_mc=n_mc, seed=seed, tol=1.0)
    Ymax = y_max_bound(w, delta_msb)
    Ymax_rowblock = row_block_bound(d_grad, b) if d_grad is not None else None
    rhs_core = base["rhs"]["total"]
    rhs_with_prg = rhs_core + float(eps_prg) * Ymax
    lhs = base["lhs_mc"]
    rel_core = abs(lhs - rhs_core) / rhs_core if rhs_core > 0 else (0.0 if lhs == 0 else np.inf)
    return {"w": int(w), "k_g": int(k_g), "d_adv_prime_eff": float(d_adv_prime_eff),
            "c_quant": float(c_quant), "eps_prg": float(eps_prg),
            "cross_term_raw": base["cross_term_raw"], "energy_trace": base["energy_trace"],
            "rhs_core": float(rhs_core), "rhs_with_prg": float(rhs_with_prg),
            "y_max": float(Ymax), "y_max_rowblock": Ymax_rowblock,
            "lhs_mc": float(lhs), "rel_error_core": float(rel_core),
            "within_tol": bool(rel_core <= tol),
            "lhs_le_rhs_with_prg": bool(lhs <= rhs_with_prg + 1e-12),
            "tol": float(tol), "n_mc": int(n_mc)}