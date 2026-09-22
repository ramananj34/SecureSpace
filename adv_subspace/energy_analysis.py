from __future__ import annotations
import numpy as np

def reduction_factor(k, d_adv, delta_msb=1.0, c_quant=None, b=8, delta_lsb=None):
    if c_quant is None:
        dl = delta_lsb if delta_lsb is not None else 2.0 ** -(b - 1)
        c_quant = (4 ** b - 1) / 3.0 * dl ** 2
    return float((k / d_adv) * (delta_msb ** 2 / c_quant))

def worst_case_cross_main_ratio(k, d_adv, w, d=None, b=8):
    if d is None:
        d = k / b
    return float(3.0 * w * d / (d_adv * k))

def average_cross_main_ratio(k, w):
    return float((w - 1) / (k - w)) if k > w else float("nan")

def energy_regime_table(k, weights, d_adv_values, delta_msb=1.0, b=8, delta_lsb=None):
    dl = delta_lsb if delta_lsb is not None else 2.0 ** -(b - 1)
    c_quant = (4 ** b - 1) / 3.0 * dl ** 2
    rows = []
    for d_adv in d_adv_values:
        rf = reduction_factor(k, d_adv, delta_msb, c_quant)
        for w in weights:
            rows.append({
                "d_adv": float(d_adv), "w": int(w),
                "reduction_factor": rf,
                "l2_reduction_factor": float(np.sqrt(rf)),
                "avg_cross_main_ratio": average_cross_main_ratio(k, w),
                "worst_cross_main_ratio": worst_case_cross_main_ratio(k, d_adv, w, b=b),
            })
    return {"c_quant": float(c_quant), "k": int(k), "rows": rows}

def cross_term_distribution(Pi_V, D, k, n_xq=4000, seed=0):
    Pi_V = np.asarray(Pi_V, np.float64); D = np.asarray(D, np.float64)
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_xq))
    for i in range(int(n_xq)):
        s = 1.0 - 2.0 * rng.integers(0, 2, k)
        Ds = D @ s; PiDs = Pi_V @ Ds
        vals[i] = float(PiDs @ PiDs)
    trace_pred = float(np.trace(D.T @ Pi_V @ D))
    return {"mean": float(vals.mean()), "trace_prediction": trace_pred,
            "mean_matches_trace_rel": abs(vals.mean() - trace_pred) / trace_pred if trace_pred > 0 else float("nan"),
            "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max()),
            "p05": float(np.percentile(vals, 5)), "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)), "n_xq": int(n_xq)}

def worst_case_bound_on_cross(D, k, b=8):
    D = np.asarray(D, np.float64); d = D.shape[0]
    dlsb = 2.0 ** -(b - 1)
    delta_max = (2 ** b - 1) * dlsb
    return {"upper_bound_Ds_sq": float(d * delta_max ** 2), "delta_max": float(delta_max)}