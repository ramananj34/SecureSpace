from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "fl_core"), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from grad_channel import GradientQuantizer, poison_delta_info, receiver_perturbation, isl_key  # frozen W13

def uplink_clean(theta, gq: GradientQuantizer, key, t=0):
    bits, rng_tuple = gq.encode(np.asarray(theta, np.float64))
    return gq.decode(bits, rng_tuple)

def uplink_attacked(theta, gq: GradientQuantizer, key, w, sensitivity=None, perm_on=True, t=0, seed=0):
    theta = np.asarray(theta, np.float64)
    bits, rng_tuple = gq.encode(theta)
    kg = bits.size
    b = int(gq.b)
    d = theta.size
    if perm_on:
        di = poison_delta_info(kg, int(w), np.random.default_rng([seed, 0xE17]))
        v = receiver_perturbation(di, key, t)
        m_hat = bits ^ v
    else:
        if sensitivity is None:
            sensitivity = np.ones(d)
        target_coords = np.argsort(-np.abs(sensitivity))[: int(w)]
        v = np.zeros(kg, np.uint8)
        v[target_coords * b + (b - 1)] = 1
        m_hat = bits ^ v
    return gq.decode(m_hat, rng_tuple)

def l2_integrity(received, true_theta):
    return float(np.linalg.norm(np.asarray(received) - np.asarray(true_theta)))

def targeted_damage(received, true_theta, sensitivity):
    corr = np.asarray(received) - np.asarray(true_theta)
    s = np.asarray(sensitivity, np.float64)
    s = s / (np.linalg.norm(s) + 1e-12)
    return float((s @ corr) ** 2)

def uplink_experiment(theta, gq, key, w, sensitivity, seed=0):
    clean = uplink_clean(theta, gq, key)
    undef = uplink_attacked(theta, gq, key, w, sensitivity=sensitivity, perm_on=False, seed=seed)
    amrcc = uplink_attacked(theta, gq, key, w, sensitivity=sensitivity, perm_on=True, seed=seed)
    return {
        "l2": {"clean": l2_integrity(clean, theta), "undefended": l2_integrity(undef, theta),
               "amrcc": l2_integrity(amrcc, theta)},
        "targeted_damage": {"clean": targeted_damage(clean, theta, sensitivity),
                            "undefended": targeted_damage(undef, theta, sensitivity),
                            "amrcc": targeted_damage(amrcc, theta, sensitivity)},
        "w": int(w), "d": int(theta.size),
    }