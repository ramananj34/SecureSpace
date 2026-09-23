from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "fl_core"), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from grad_channel import GradientQuantizer, poison_delta_info, receiver_perturbation, isl_key


def byzantine_mask(M, B, seed=0):
    rng = np.random.default_rng([seed, 0xB1])
    idx = rng.choice(M, int(B), replace=False) if B > 0 else np.array([], dtype=int)
    mask = np.zeros(M, dtype=bool)
    mask[idx] = True
    return mask

def unbounded_aligned_gradient(honest_updates, scale=50.0, direction=None):
    hs = np.mean(np.asarray(honest_updates, np.float64), 0)
    d = direction if direction is not None else -hs
    n = np.linalg.norm(d)
    return (scale * d / n) if n > 1e-12 else np.zeros_like(hs)

def bounded_scattered_gradient(honest_update_i, gq: GradientQuantizer, key, w, t=0, seed=0):
    bits, rng_tuple = gq.encode(np.asarray(honest_update_i, np.float64))
    kg = bits.size
    di = poison_delta_info(kg, int(w), np.random.default_rng([seed, 0x5C]))
    v = receiver_perturbation(di, key, t)
    m_hat = bits ^ v
    return gq.decode(m_hat, rng_tuple)

def apply_attack(honest_updates, regime, B_mask, gq=None, base_seed=0, t=0,
                 unbounded_scale=50.0, scatter_w=100, v_theta_dir=None):
    U = [np.asarray(u, np.float64).copy() for u in honest_updates]
    M = len(U)
    perm_on = regime.endswith("_perm")
    honest_only = [U[i] for i in range(M) if not B_mask[i]]
    for i in range(M):
        if not B_mask[i]:
            continue
        if perm_on:
            if gq is None:
                raise ValueError("perm regime needs a GradientQuantizer")
            key = isl_key(base_seed, link_id=i)
            U[i] = bounded_scattered_gradient(U[i], gq, key, w=scatter_w, t=t, seed=base_seed * 1000 + i)
        else:
            U[i] = unbounded_aligned_gradient(honest_only, scale=unbounded_scale, direction=v_theta_dir)
    return np.array(U)

def regime_aggregator(regime):
    base = regime.replace("_perm", "")
    agg = {"fedavg": "fedavg", "median": "coord_median", "krum": "krum"}[base]
    return agg, regime.endswith("_perm")