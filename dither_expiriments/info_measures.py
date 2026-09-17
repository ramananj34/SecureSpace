from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from dither import dithered_levels

def _to_base(x, base):
    return x / np.log(2) if base == "bit" else x

def mi_from_joint(joint, base="nat"):
    P = np.asarray(joint, np.float64)
    tot = P.sum()
    if tot <= 0:
        return 0.0
    P = P / tot
    Px = P.sum(1, keepdims=True); Py = P.sum(0, keepdims=True)
    m = P > 0
    r = P[m] / (Px @ Py)[m]
    return float(max(_to_base(np.sum(P[m] * np.log(r)), base), 0.0))

def kl_categorical(p, q, eps=1e-12, base="nat"):
    p = np.asarray(p, np.float64); q = np.asarray(q, np.float64)
    p = p / p.sum()
    q = np.clip(q, eps, None); q = q / q.sum()
    m = p > 0
    return float(max(_to_base(np.sum(p[m] * np.log(p[m] / q[m])), base), 0.0))

def _joint2x2(x, y):
    idx = (np.asarray(x).astype(np.int64) << 1) | np.asarray(y).astype(np.int64)
    return np.bincount(idx, minlength=4).reshape(2, 2)

def _mi_two_binary(x, y, base="nat", miller_madow=True):
    c = _joint2x2(x, y).astype(np.float64)
    mi = mi_from_joint(c, base)
    if miller_madow:
        N = c.sum()
        if N > 0:
            mx = int((c.sum(1) > 0).sum()); my = int((c.sum(0) > 0).sum()); mxy = int((c > 0).sum())
            corr = _to_base((mx + my - mxy - 1) / (2.0 * N), base)
            mi = max(mi + corr, 0.0)
    return float(mi)

def adjacency_mi(s, base="nat", miller_madow=True):
    s = np.asarray(s).astype(np.int64).ravel()
    if s.size < 2:
        return 0.0
    return _mi_two_binary(s[:-1], s[1:], base, miller_madow)

def level_hist(levels, n_levels):
    c = np.bincount(np.asarray(levels).astype(np.int64).ravel(), minlength=n_levels).astype(np.float64)
    s = c.sum()
    return c / s if s > 0 else c

def bitplane_bernoulli(bits):
    bits = np.asarray(bits)
    return bits.reshape(-1, bits.shape[-1]).mean(0).astype(np.float64)

def dither_info_curve(tele, quantizer, sigmas, n_reps=8, seed=0, base="nat", miller_madow=True):
    q = quantizer; b = q.b
    tele = np.asarray(tele, np.float64)
    levels0 = q.quantize(tele)
    bits0 = np.asarray(q.levels_to_bits(levels0), np.uint8)
    adj0 = [adjacency_mi(bits0[:, p], base, miller_madow) for p in range(b)]
    hist0 = level_hist(levels0, q.n_levels)
    bern0 = bitplane_bernoulli(bits0)
    out = []
    for i_s, sigma in enumerate(sigmas):
        adj = np.zeros(b); sig = np.zeros(b); planekl = np.zeros(b); bern = np.zeros(b)
        lvlkl = []
        for rep in range(n_reps):
            rng = np.random.default_rng([seed, i_s, rep])
            lv = dithered_levels(tele, sigma, rng, q)
            bits = np.asarray(q.levels_to_bits(lv), np.uint8)
            adj += np.array([adjacency_mi(bits[:, p], base, miller_madow) for p in range(b)])
            sig += np.array([_mi_two_binary(bits[:, p], bits0[:, p], base, miller_madow) for p in range(b)])
            h = level_hist(lv, q.n_levels)
            lvlkl.append(kl_categorical(hist0, h, base=base))
            bpp = bitplane_bernoulli(bits)
            for p in range(b):
                planekl[p] += kl_categorical([1 - bern0[p], bern0[p]], [1 - bpp[p], bpp[p]], base=base)
            bern += bpp
        adj /= n_reps; sig /= n_reps; planekl /= n_reps; bern /= n_reps
        out.append({
            "sigma": float(sigma), "sigma_lsb": float(sigma) / q.delta_lsb,
            "adjacency_mi_per_plane": [float(x) for x in adj], "adjacency_mi_total": float(adj.sum()),
            "signal_mi_per_plane": [float(x) for x in sig], "signal_mi_total": float(sig.sum()),
            "level_kl_mean": float(np.mean(lvlkl)), "level_kl_std": float(np.std(lvlkl)),
            "bitplane_kl_per_plane": [float(x) for x in planekl], "bitplane_kl_total": float(planekl.sum()),
            "bern_p1_per_plane": [float(x) for x in bern], "base": base,
        })
    return {
        "no_dither": {
            "adjacency_mi_per_plane": [float(x) for x in adj0], "adjacency_mi_total": float(sum(adj0)),
            "level_hist": [float(x) for x in hist0], "bern_p1_per_plane": [float(x) for x in bern0],
            "base": base,
        },
        "by_sigma": out,
    }