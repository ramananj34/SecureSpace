from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from fgsm_pgd_attacks import detect as _detect, missed as _missed, footprint_mask
from sampling import selection_sample, clopper_pearson

def _flippable(label, T, cfg, window=None, b=8):
    if window is not None:
        start, length = window
        ts = np.arange(int(start), min(int(start) + int(length), T))
    else:
        F = footprint_mask(label, cfg.l_s, cfg.error_buffer, T)
        ts = np.where(F > 0)[0]
    return ts.astype(np.int64), int(ts.size * b)

def eta_w_estimate(chan_id, model, train_features, tele, cmds, labels, label, cfg, quantizer, w, n_samples, window=None, seed=0, alpha=0.05, sampler="choice", detect_fn=None, missed_fn=None):
    detect_fn = detect_fn or _detect
    missed_fn = missed_fn or _missed
    rng = np.random.default_rng(seed)
    tele = np.asarray(tele, np.float32)
    T = len(tele)
    b = int(getattr(quantizer, "b", 8))
    ts, n_bits = _flippable(label, T, cfg, window=window, b=b)
    if not (0 <= w <= n_bits):
        raise ValueError(f"w={w} out of range [0,{n_bits}] (region has {ts.size} timesteps x {b} bits)")
    levels0 = quantizer.quantize(tele)
    bits0 = np.asarray(quantizer.levels_to_bits(levels0))
    tele_q = np.asarray(quantizer.dequantize(levels0), np.float32)
    missed_clean = bool(missed_fn(detect_fn(chan_id, model, train_features, tele_q, cmds, cfg), label))
    misses = 0
    for _ in range(int(n_samples)):
        bits = bits0.copy()
        if w > 0:
            flat = selection_sample(n_bits, w, rng, method=sampler)
            bits[ts[flat // b], flat % b] ^= 1
        tele_pert = np.asarray(quantizer.dequantize(quantizer.bits_to_levels(bits)), np.float32)
        if missed_fn(detect_fn(chan_id, model, train_features, tele_pert, cmds, cfg), label):
            misses += 1
    eta = misses / float(n_samples)
    lo, hi = clopper_pearson(misses, int(n_samples), alpha)
    return {"chan": chan_id, "label": [int(label[0]), int(label[1])], "w": int(w), "n_bits_region": int(n_bits), "region_timesteps": int(ts.size), "n_samples": int(n_samples), "misses": int(misses), "eta_hat": float(eta), "ci_lo": float(lo), "ci_hi": float(hi), "ci_alpha": float(alpha), "missed_clean": missed_clean, "sampler": sampler}

def eta_curve(chan_id, model, train_features, tele, cmds, labels, label, cfg, quantizer, weights, n_samples, **kw):
    return [eta_w_estimate(chan_id, model, train_features, tele, cmds, labels, label, cfg, quantizer, int(w), n_samples, **kw) for w in weights]

def max_eta(curve):
    if not curve:
        return {"max_eta_hat": 0.0, "max_ci_hi": 0.0, "argmax_w": None}
    j = int(np.argmax([c["eta_hat"] for c in curve]))
    return {"max_eta_hat": float(curve[j]["eta_hat"]), "max_ci_hi": float(curve[j]["ci_hi"]), "argmax_w": int(curve[j]["w"])}