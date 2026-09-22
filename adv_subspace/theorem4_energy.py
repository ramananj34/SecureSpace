from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import permutation_for_frame, inverse_permutation, apply_inverse_perm
from sampling import selection_sample

def closed_form_rhs(w, k, c_quant, d_adv_eff, cross_term):
    coef_main = (w * (k - w)) / (k * (k - 1))
    coef_cross = (w * (w - 1)) / (k * (k - 1))
    main = coef_main * c_quant * d_adv_eff
    cross = coef_cross * cross_term
    return {"main_term": float(main), "cross_term_scaled": float(cross), "total": float(main + cross),
            "coef_main": float(coef_main), "coef_cross": float(coef_cross)}

def cross_term_value(Pi_V, D, x_q_bits):
    s = 1.0 - 2.0 * np.asarray(x_q_bits, np.float64)
    Ds = np.asarray(D) @ s
    PiDs = np.asarray(Pi_V) @ Ds
    return float(PiDs @ PiDs)

def mc_projected_energy(Pi_V, D, x_q_bits, w, k, n_mc=20000, seed=0, use_keyed=False, key=None):
    Pi_V = np.asarray(Pi_V, np.float64); D = np.asarray(D, np.float64)
    s = 1.0 - 2.0 * np.asarray(x_q_bits, np.float64)
    DS = D * s[None, :]
    rng = np.random.default_rng(seed)
    supp = selection_sample(k, w, rng)
    delta = np.zeros(k, np.uint8); delta[supp] = 1
    key_rng = np.random.default_rng([seed, 0xE15])
    acc = 0.0
    for i in range(int(n_mc)):
        if use_keyed:
            kk = (key if key is not None else key_rng.bytes(32))
            perm = permutation_for_frame(kk, i, k)
        else:
            perm = rng.permutation(k)
        v = delta[inverse_permutation(np.asarray(perm, np.int64))]
        vec = DS @ v.astype(np.float64)
        pv = Pi_V @ vec
        acc += float(pv @ pv)
    return acc / float(n_mc)

def verify_theorem4(Pi_V, D, x_q_bits, w, k, c_quant, d_adv_eff, n_mc=20000, seed=0,
                    tol=0.20, use_keyed=False):
    Pi_V = np.asarray(Pi_V, np.float64); D = np.asarray(D, np.float64)
    M = Pi_V.T @ Pi_V
    tr_energy = float(np.trace(D.T @ M @ D)) / c_quant if c_quant > 0 else 0.0
    s = 1.0 - 2.0 * np.asarray(x_q_bits, np.float64)
    Ds = D @ s
    cross_raw = float(Ds @ (M @ Ds))
    coef_main = (w * (k - w)) / (k * (k - 1))
    coef_cross = (w * (w - 1)) / (k * (k - 1))
    main = coef_main * c_quant * tr_energy
    cross = coef_cross * cross_raw
    rhs_total = main + cross
    lhs = mc_projected_energy(Pi_V, D, x_q_bits, w, k, n_mc=n_mc, seed=seed, use_keyed=use_keyed)
    rel = abs(lhs - rhs_total) / rhs_total if rhs_total > 0 else (0.0 if lhs == 0 else np.inf)
    return {"w": int(w), "k": int(k), "d_adv_eff_reported": float(d_adv_eff),
            "energy_trace": float(tr_energy), "c_quant": float(c_quant),
            "cross_term_raw": float(cross_raw),
            "rhs": {"main_term": float(main), "cross_term_scaled": float(cross), "total": float(rhs_total),
                    "coef_main": float(coef_main), "coef_cross": float(coef_cross)},
            "lhs_mc": float(lhs), "rel_error": float(rel),
            "within_tol": bool(rel <= tol), "tol": float(tol), "n_mc": int(n_mc),
            "use_keyed": bool(use_keyed)}

def cross_equals_trace_on_average(Pi_V, D, k, c_quant, d_adv_eff, n_xq=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [cross_term_value(Pi_V, D, rng.integers(0, 2, k)) for _ in range(int(n_xq))]
    emp = float(np.mean(vals))
    trace_pred = float(c_quant * d_adv_eff)
    return {"empirical_cross_mean": emp, "trace_prediction": trace_pred,
            "rel_error": abs(emp - trace_pred) / trace_pred if trace_pred > 0 else float("nan"),
            "n_xq": int(n_xq)}

def unsigned_xq0_cross(Pi_V, D):
    d, k = np.asarray(D).shape
    return cross_term_value(Pi_V, D, np.zeros(k, np.uint8))