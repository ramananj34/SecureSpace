from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from ceil_attack import (CeilingConfig, _ewma_span, _region_and_baseline, _pgd_band, _square_polish, _proxy_loss, _es_over, _deanomalized_init, _band_loss)
from fgsm_pgd_attacks import (detect, missed, collateral_fp, footprint_mask, _label_seed, _rand_init)

def lift_to_bits(g_R, x_q_levels, quantizer):
    g_R = np.asarray(g_R, dtype=np.float64)
    bits = quantizer.levels_to_bits(np.asarray(x_q_levels))
    signs = 1.0 - 2.0 * bits.astype(np.float64)
    weights = (2.0 ** np.arange(quantizer.b)) * quantizer.delta_lsb
    return g_R[:, None] * signs * weights[None, :]

def snap_to_lattice(tele_q, delta_star, quantizer, x_q_levels=None):
    tele_q = np.asarray(tele_q, dtype=np.float64)
    delta_star = np.asarray(delta_star, dtype=np.float64)
    if x_q_levels is None:
        x_q_levels = quantizer.quantize(tele_q)
    x_q_levels = np.asarray(x_q_levels)
    levels_adv = quantizer.quantize(tele_q + delta_star)
    bits_clean = quantizer.levels_to_bits(x_q_levels)
    bits_adv = quantizer.levels_to_bits(levels_adv)
    delta_info_bits = (bits_clean ^ bits_adv).astype(np.uint8)
    tele_recv = quantizer.dequantize(levels_adv)
    tele_q_dq = quantizer.dequantize(x_q_levels)
    delta_snap = tele_recv - tele_q_dq
    dlsb = quantizer.delta_lsb
    sz = delta_snap.size
    return {"delta_info_bits": delta_info_bits, "levels_adv": levels_adv, "x_q_levels": x_q_levels, "tele_recv": tele_recv, "delta_snap": delta_snap, "weight": int(delta_info_bits.sum()), "realized_linf_lsb": float(np.max(np.abs(delta_snap)) / dlsb) if sz else 0.0, "requant_loss_lsb": float(np.max(np.abs(delta_star - delta_snap)) / dlsb) if sz else 0.0}

def _alpha_K(cfg):
    span = _ewma_span(cfg)
    return span, 2.0 / (span + 1), 8 * span

def c2_ceiling_attack(chan_id, model, train_features, tele, cmds, labels, label, cfg, ccfg, eps, quantizer, return_arrays=False):
    device = next(model.parameters()).device
    T = len(tele)
    l_s = cfg.l_s
    span, alpha, K = _alpha_K(cfg)
    x_q_stream = quantizer.quantize(np.asarray(tele, np.float64))
    tele_q = quantizer.dequantize(x_q_stream).astype(np.float64)
    tau_t = torch.tensor(tele_q.astype(np.float32), device=device)
    cmds_t = torch.tensor(cmds.astype(np.float32), device=device)
    F = footprint_mask(label, l_s, cfg.error_buffer, T)
    F_t = torch.tensor(F, device=device)
    fnz = np.where(F > 0)[0]
    F_lo, F_hi = int(fnz[0]), int(fnz[-1])
    b, b_low, R_loss_t, inwin_t, info = _region_and_baseline(model, tau_t, cmds_t, label, cfg, alpha, K, ccfg.batch_windows, F_lo, F_hi, T, ccfg.max_grad_windows)
    proxy = lambda d: _proxy_loss(model, tau_t, d, cmds_t, R_loss_t, inwin_t, l_s, alpha, K, b, b_low, ccfg.batch_windows)
    def _snap(d_np):
        return snap_to_lattice(tele_q, d_np, quantizer, x_q_levels=x_q_stream)
    def true_miss(delta_t):
        s = _snap(delta_t.detach().cpu().numpy())
        E = detect(chan_id, model, train_features, s["tele_recv"], cmds, cfg)
        return missed(E, label)
    zeros = torch.zeros(T, device=device)
    out = {"chan": chan_id, "label": [int(label[0]), int(label[1])], "eps": float(eps), "span": int(span), "baseline_b": float(b), "baseline_b_low": float(b_low), "loss_clean": proxy(zeros), **{f"region_{k}": v for k, v in info.items()}}
    gen = torch.Generator().manual_seed(_label_seed(ccfg.seed, chan_id, label))
    deanom = torch.tensor(_deanomalized_init(tele_q, label, eps, F, T), device=device)
    best, best_l, band_missed, ru = zeros, out["loss_clean"], False, 0
    for r in range(ccfg.band_restarts):
        if r == 0:
            init = deanom.clone()
        elif r == 1:
            init = zeros.clone()
        else:
            init = _rand_init(eps, F_t, T, gen)
        d = _pgd_band(model, tau_t, cmds_t, F_t, R_loss_t, inwin_t, eps, ccfg.band_T_steps, l_s, alpha, K, b, b_low, init, ccfg.batch_windows)
        ru = r + 1
        if true_miss(d):
            band_missed, best, best_l = True, d, proxy(d)
            break
        ll = proxy(d)
        if ll < best_l:
            best, best_l = d, ll
    out["ewma_pgd_missed"] = bool(band_missed)
    out["ewma_pgd_restarts_used"] = int(ru)
    out["ewma_pgd_loss"] = float(best_l)
    if band_missed:
        final, ceil_missed, nq = best, True, 0
    else:
        final, ceil_missed, nq = _square_polish(F_lo, F_hi, eps, best, proxy, true_miss, ccfg.query_budget, ccfg.square_iters, gen)
    out["ceiling_missed"] = bool(ceil_missed)
    out["ceiling_queries_used"] = int(nq)
    out["ceiling_loss"] = float(proxy(final))
    snap = _snap(final.detach().cpu().numpy())
    E = detect(chan_id, model, train_features, snap["tele_recv"], cmds, cfg)
    out["ceiling_collateral"] = int(collateral_fp(E, labels))
    out["ceiling_missed_lattice"] = bool(missed(E, label))
    out["requant_loss_lsb"] = float(snap["requant_loss_lsb"])
    out["realized_linf_lsb"] = float(snap["realized_linf_lsb"])
    out["delta_info_weight_stream"] = int(snap["weight"])
    if return_arrays:
        out["_delta_info_bits"] = snap["delta_info_bits"]
        out["_delta_snap"] = snap["delta_snap"]
        out["_x_q_levels"] = snap["x_q_levels"]
        out["_tele_q"] = tele_q
    return out