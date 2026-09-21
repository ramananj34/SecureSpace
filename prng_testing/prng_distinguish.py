from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import inverse_permutation
from sampling import selection_sample

np.seterr(over="ignore")

def induced_v_hit_probs(gen, k, w, n_perm=4000, seed=0, fixed_support=True):
    rng = np.random.default_rng([seed, 1])
    if fixed_support:
        D = np.sort(rng.choice(k, w, replace=False))
    q = np.zeros(k)
    for i in range(int(n_perm)):
        if not fixed_support:
            D = np.sort(np.random.default_rng([seed, 2, i]).choice(k, w, replace=False))
        di = np.zeros(k, np.uint8); di[D] = 1
        perm = gen(np.random.default_rng([seed, 3, i]).bytes(32), 0, k)
        inv = inverse_permutation(np.asarray(perm, np.int64))
        v = di[inv]
        q[v == 1] += 1
    return q / float(n_perm)

def eps_v(gen, k, w, n_perm=4000, seed=0, fixed_support=True):
    q = induced_v_hit_probs(gen, k, w, n_perm=n_perm, seed=seed, fixed_support=fixed_support)
    return float(np.sum(np.abs(q - w / k)) / (2.0 * w))

def eps_v_mc_floor(k, w, n_perm=4000, seed=0, reps=8):
    def uni(key, t, kk): return np.random.default_rng(key[:8]).permutation(kk)
    vals = []
    for r in range(reps):
        rng = np.random.default_rng([seed, 100, r])
        D = np.sort(rng.choice(k, w, replace=False)); di = np.zeros(k, np.uint8); di[D] = 1
        q = np.zeros(k)
        for i in range(int(n_perm)):
            perm = rng.permutation(k)
            inv = inverse_permutation(perm.astype(np.int64))
            q[di[inv] == 1] += 1
        q /= n_perm
        vals.append(np.sum(np.abs(q - w / k)) / (2.0 * w))
    return {"mc_floor_mean": float(np.mean(vals)), "mc_floor_std": float(np.std(vals))}

def _perm_stats(sample):
    M, k = sample.shape
    fp = float(np.sum(sample == np.arange(k), axis=1).mean())
    colmean = sample.mean(0)
    marg = float(np.sqrt(np.mean((colmean - (k - 1) / 2.0) ** 2)))
    a = sample[:, :-1].astype(float) - sample[:, :-1].mean(0)
    b = sample[:, 1:].astype(float) - sample[:, 1:].mean(0)
    adj = float(np.mean(np.abs((a * b).mean(0) / (np.sqrt((a * a).mean(0) * (b * b).mean(0)) + 1e-12))))
    disp = float(np.mean(np.abs(sample - np.arange(k))))
    return np.array([fp, marg, adj, disp])

_STAT_NAMES = ["fixed_points", "marginal_dev", "adjacency_corr", "displacement"]

def _uniform_null(k, M, reps=40, seed=555):
    V = np.array([_perm_stats(np.stack([np.random.default_rng([seed, r, i]).permutation(k)
                                        for i in range(M)])) for r in range(reps)])
    return V.mean(0), V.std(0) + 1e-12

def perm_nonuniformity_report(gen, k, M=200, reps_null=40, seed=0):
    mu, sd = _uniform_null(k, M, reps=reps_null, seed=seed + 555)
    s = np.stack([gen(np.random.default_rng([seed, 7, i]).bytes(32), 0, k) for i in range(M)])
    z = (_perm_stats(s) - mu) / sd
    return {"stat_names": _STAT_NAMES, "z": [float(x) for x in z],
            "max_abs_z": float(np.max(np.abs(z))),
            "detectably_nonuniform": bool(np.max(np.abs(z)) > 4.0)}

def keystream_bias(stream_bits):
    b = np.asarray(stream_bits, np.float64).ravel()
    if b.size < 2:
        return {"p1": float("nan"), "bias_abs": float("nan"), "lag1_autocorr": float("nan")}
    p1 = float(b.mean())
    a = b[:-1] - b[:-1].mean(); c = b[1:] - b[1:].mean()
    denom = np.sqrt((a * a).mean() * (c * c).mean()) + 1e-12
    return {"p1": p1, "bias_abs": abs(p1 - 0.5), "lag1_autocorr": float((a * c).mean() / denom)}