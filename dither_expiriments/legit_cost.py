from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "baseline_fgsm_pgd")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from dither import dithered_levels

try:
    from fgsm_pgd_attacks import detect as _detect, missed as _missed
    from pipeline import evaluate_anomalies as _evaluate
except Exception:
    _detect = _missed = _evaluate = None

def reconstruction_cost(tele, quantizer, sigmas, n_reps=8, seed=0):
    q = quantizer
    dlsb = q.delta_lsb
    b = q.b
    tele = np.asarray(tele, np.float64)
    levels0 = q.quantize(tele)
    bits0 = np.asarray(q.levels_to_bits(levels0), np.uint8)
    recon0 = np.asarray(q.dequantize(levels0), np.float64)
    mse_q = float(np.mean((recon0 - tele) ** 2))
    out = []
    for i_s, sigma in enumerate(sigmas):
        added, mse_qd, lvlchg = [], [], []
        plane_flip = np.zeros(b)
        for rep in range(n_reps):
            rng = np.random.default_rng([seed, i_s, rep])
            lv = dithered_levels(tele, sigma, rng, q)
            recon = np.asarray(q.dequantize(lv), np.float64)
            bits = np.asarray(q.levels_to_bits(lv), np.uint8)
            added.append(np.mean((recon - recon0) ** 2))
            mse_qd.append(np.mean((recon - tele) ** 2))
            lvlchg.append(np.mean(lv.astype(np.int64) != levels0.astype(np.int64)))
            plane_flip += np.mean(bits != bits0, axis=0)
        plane_flip /= n_reps
        m_qd = float(np.mean(mse_qd)); m_add = float(np.mean(added))
        s_lsb = float(sigma) / dlsb
        out.append({
            "sigma": float(sigma), "sigma_lsb": s_lsb,
            "added_mse_vs_nodither": m_add,
            "added_rms_lsb": float(np.sqrt(m_add) / dlsb),
            "mse_recon_nodither": mse_q, "mse_recon_dither": m_qd,
            "noise_floor_db": float(10.0 * np.log10(m_qd / mse_q)) if mse_q > 0 else float("nan"),
            "rms_err_lsb_nodither": float(np.sqrt(mse_q) / dlsb),
            "rms_err_lsb_dither": float(np.sqrt(m_qd) / dlsb),
            "level_change_rate": float(np.mean(lvlchg)),
            "level_change_rate_pred": float(np.sqrt(2.0 / np.pi) * s_lsb),
            "lsb_flip_rate": float(plane_flip[0]),
            "msb_flip_rate": float(plane_flip[b - 1]),
            "plane_flip_rate": [float(x) for x in plane_flip],
        })
    return out

def detection_cost(chan_id, model, train_features, tele, cmds, labels, cfg, quantizer,
                   sigmas, n_reps=8, seed=0, detect_fn=None, missed_fn=None, evaluate_fn=None):
    detect_fn = detect_fn or _detect
    missed_fn = missed_fn or _missed
    evaluate_fn = evaluate_fn or _evaluate
    if detect_fn is None or evaluate_fn is None:
        raise RuntimeError("detect_fn/evaluate_fn unavailable (torch/data absent); pass them explicitly")
    q = quantizer
    tele = np.asarray(tele, np.float64)
    levels0 = q.quantize(tele)
    recon0 = np.asarray(q.dequantize(levels0), np.float64)
    E0 = detect_fn(chan_id, model, train_features, recon0, cmds, cfg)
    ev0 = evaluate_fn(list(E0), list(labels))
    clean0 = {"f0_5": ev0["f0_5"], "tp": ev0["tp"], "fp": ev0["fp"], "fn": ev0["fn"], "per_label_missed": [bool(missed_fn(E0, l)) for l in labels]}
    out = []
    for i_s, sigma in enumerate(sigmas):
        f0s, tps, fps, fns = [], [], [], []
        lab_miss = np.zeros(len(labels))
        for rep in range(n_reps):
            rng = np.random.default_rng([seed, i_s, rep])
            recon = np.asarray(q.dequantize(dithered_levels(tele, sigma, rng, q)), np.float64)
            E = detect_fn(chan_id, model, train_features, recon, cmds, cfg)
            ev = evaluate_fn(list(E), list(labels))
            f0s.append(ev["f0_5"]); tps.append(ev["tp"]); fps.append(ev["fp"]); fns.append(ev["fn"])
            lab_miss += np.array([missed_fn(E, l) for l in labels], float)
        lab_miss /= n_reps
        out.append({
            "sigma": float(sigma), "sigma_lsb": float(sigma) / q.delta_lsb,
            "f0_5_mean": float(np.mean(f0s)), "f0_5_min": float(np.min(f0s)),
            "tp_mean": float(np.mean(tps)), "fp_mean": float(np.mean(fps)), "fn_mean": float(np.mean(fns)),
            "per_label_miss_rate": [float(x) for x in lab_miss],
            "any_label_ever_missed": bool((lab_miss > 0).any()),
        })
    return {"clean": clean0, "by_sigma": out}

def dither_cost(chan_id, model, train_features, tele, cmds, labels, cfg, quantizer, sigmas, n_reps=8, seed=0, **det_kw):
    rc = reconstruction_cost(tele, quantizer, sigmas, n_reps=n_reps, seed=seed)
    dc = detection_cost(chan_id, model, train_features, tele, cmds, labels, cfg, quantizer, sigmas, n_reps=n_reps, seed=seed, **det_kw)
    by_sigma = []
    for r, d in zip(rc, dc["by_sigma"]):
        by_sigma.append({**r, **{k: v for k, v in d.items() if k not in ("sigma", "sigma_lsb")}})
    return {"chan": chan_id, "clean_detection": dc["clean"], "by_sigma": by_sigma}