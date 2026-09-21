from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from prng_distinguish import eps_v, eps_v_mc_floor

np.seterr(over="ignore")

def perm_distribution_entropy(gen, k, t=0, n_keys=400, seed=0):
    from collections import Counter
    c = Counter()
    for i in range(int(n_keys)):
        key = np.random.default_rng([seed, 21, i]).bytes(32)
        c[gen(key, t, k).tobytes()] += 1
    counts = np.array(list(c.values()), float)
    p = counts / counts.sum()
    H = float(-(p * np.log(p + 1e-12)).sum())
    H_max = float(np.log(int(n_keys)))
    return {"n_distinct": int(len(c)), "n_keys": int(n_keys),
            "entropy_nats": H, "entropy_ratio": (H / H_max) if H_max > 0 else 0.0}

def recover_score(gen, k, t=0, n_keys=400, seed=0):
    d = perm_distribution_entropy(gen, k, t=t, n_keys=n_keys, seed=seed)
    return {"recover_score": 1.0 - d["n_distinct"] / d["n_keys"], **d}

def nonuniformity_score(gen, k, w, n_perm=3000, seed=0):
    e = eps_v(gen, k, w, n_perm=n_perm, seed=seed)
    floor = eps_v_mc_floor(k, w, n_perm=min(n_perm, 2000), reps=5, seed=seed)["mc_floor_mean"]
    return {"eps_v": e, "eps_v_mc_floor": floor,
            "eps_v_above_floor": max(0.0, e - floor)}

def attribute(gen, k, w, n_keys=400, n_perm=3000, seed=0):
    ri = recover_score(gen, k, n_keys=n_keys, seed=seed)
    rii = nonuniformity_score(gen, k, w, n_perm=n_perm, seed=seed)
    rec, nu = ri["recover_score"], rii["eps_v_above_floor"]
    hi_rec, hi_nu = rec > 0.30, nu > 0.10
    if hi_rec and hi_nu:
        label = "both"
    elif hi_rec:
        label = "recovery"
    elif hi_nu:
        label = "nonuniformity"
    else:
        label = "neither"
    return {"recover_score": rec, "eps_v": rii["eps_v"], "eps_v_above_floor": nu,
            "n_distinct_perms": ri["n_distinct"], "perm_entropy_ratio": ri["entropy_ratio"],
            "route_label": label}