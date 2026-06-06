from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_THIS), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from fgsm_pgd_attacks import (DEVICE, prepare_model, load_streams, detect, missed, collateral_fp, footprint_mask, _label_seed, _rand_init)

def _ewma_span(cfg) -> int:
    try:
        return max(1, int(cfg.batch_size * cfg.window_size * cfg.smoothing_perc))
    except Exception:
        return 105

def _causal_ewma(e: torch.Tensor, alpha: float, K: int) -> torch.Tensor:
    device, dtype = e.device, e.dtype
    j = torch.arange(K, device=device, dtype=dtype)
    k = alpha * (1.0 - alpha) ** j
    w = torch.flip(k, dims=[0]).view(1, 1, K)
    e_pad = torch.cat([e.new_zeros(K - 1), e]).view(1, 1, -1)
    return torch.nn.functional.conv1d(e_pad, w).view(-1)

def _windows_errors(model, tau, delta, cmds, R_t, l_s, batch_windows, requires_grad):
    device = tau.device
    arange_ls = torch.arange(l_s, device=device)
    def body():
        parts = []
        for s in range(0, len(R_t), batch_windows):
            idx = R_t[s:s + batch_windows]
            win = idx[:, None] + arange_ls[None, :]
            tele_win = tau[win] + delta[win]
            x = torch.cat([tele_win[..., None], cmds[win]], dim=-1)
            pred = model(x)[:, 0]
            tgt = tau[idx + l_s] + delta[idx + l_s]
            parts.append((pred - tgt).abs())
        return torch.cat(parts)
    if requires_grad:
        return body()
    with torch.no_grad():
        return body()

def _es_over(model, tau, delta, cmds, R_t, l_s, alpha, K, batch_windows, requires_grad):
    e = _windows_errors(model, tau, delta, cmds, R_t, l_s, batch_windows, requires_grad)
    return _causal_ewma(e, alpha, K)

def _region_and_baseline(model, tau_t, cmds_t, label, cfg, alpha, K, batch_windows, F_lo_t, F_hi_t, T, max_grad_windows):
    l_s, buf = cfg.l_s, cfg.error_buffer
    n_windows = T - l_s - cfg.n_predictions
    p0, p1 = label[0] - l_s, label[1] - l_s
    lo_i, hi_i = max(0, p0 - buf), min(n_windows - 1, p1 + buf)
    bs, W = cfg.batch_size, cfg.window_size * cfg.batch_size
    peak = (lo_i + hi_i) // 2
    i_win = max(0, int((peak - (W - bs)) // bs))
    ws, we = i_win * bs, min(n_windows, i_win * bs + W)
    f0, f1 = max(0, F_lo_t - l_s), min(n_windows - 1, F_hi_t - l_s)
    Rb0 = max(0, ws - K)
    Rb = np.arange(Rb0, we)
    Rb_t = torch.tensor(Rb, device=tau_t.device, dtype=torch.long)
    es_b = _es_over(model, tau_t, torch.zeros(T, device=tau_t.device), cmds_t, Rb_t, l_s, alpha, K, batch_windows, requires_grad=False).cpu().numpy()
    in_win = (Rb >= ws) & (Rb < we)
    out_foot = (Rb < f0) | (Rb > f1)
    base_vals = es_b[in_win & out_foot]
    b = float(np.median(base_vals)) if base_vals.size else float(np.median(es_b))
    b_low = float(np.percentile(base_vals, 10)) if base_vals.size else max(0.0, 0.5 * b)
    Rlab = np.arange(max(0, lo_i - K), hi_i + 1)
    Rlab_t = torch.tensor(Rlab, device=tau_t.device, dtype=torch.long)
    es_lab = _es_over(model, tau_t, torch.zeros(T, device=tau_t.device), cmds_t, Rlab_t, l_s, alpha, K, batch_windows, requires_grad=False).cpu().numpy()
    lab_mask = (Rlab >= lo_i) & (Rlab <= hi_i)
    lab_pos = Rlab[lab_mask]
    exc = np.clip(es_lab[lab_mask] - b, 0.0, None)
    if exc.max() > 0:
        core = lab_pos[exc >= 0.5 * exc.max()]
        peak_pos = int(np.median(core))
    else:
        peak_pos = int(lab_pos[len(lab_pos) // 2])
    W_in = min(hi_i - lo_i + 1, max(1, max_grad_windows - K))
    c_lo = max(lo_i, min(peak_pos - W_in // 2, hi_i - W_in + 1))
    c_hi = min(hi_i, c_lo + W_in - 1)
    R_loss = np.arange(max(0, c_lo - K), c_hi + 1)
    inwin = np.where((R_loss >= c_lo) & (R_loss <= c_hi))[0]
    R_loss_t = torch.tensor(R_loss, device=tau_t.device, dtype=torch.long)
    inwin_t = torch.tensor(inwin, device=tau_t.device, dtype=torch.long)
    return b, b_low, R_loss_t, inwin_t, {"lo_i": int(lo_i), "hi_i": int(hi_i), "peak_pos": int(peak_pos), "c_lo": int(c_lo), "c_hi": int(c_hi)}

def _band_loss(es_sel, b, b_low):
    over = torch.clamp(es_sel - b, min=0.0)
    under = torch.clamp(b_low - es_sel, min=0.0)
    return (over ** 2).mean() + 0.5 * (under ** 2).mean()

def _proxy_loss(model, tau, delta, cmds, R_loss_t, inwin_t, l_s, alpha, K, b, b_low, batch_windows):
    es = _es_over(model, tau, delta, cmds, R_loss_t, l_s, alpha, K, batch_windows, requires_grad=False)
    return float(_band_loss(es[inwin_t], b, b_low))

def _grad_band(model, tau, delta, cmds, R_loss_t, inwin_t, l_s, alpha, K, b, b_low, batch_windows):
    delta.grad = None
    with torch.backends.cudnn.flags(enabled=False):
        es = _es_over(model, tau, delta, cmds, R_loss_t, l_s, alpha, K, batch_windows, requires_grad=True)
        loss = _band_loss(es[inwin_t], b, b_low)
        loss.backward()
    return float(loss.detach()), delta.grad.detach().clone()

def _pgd_band(model, tau, cmds, F_mask, R_loss_t, inwin_t, eps, T_steps, l_s, alpha, K, b, b_low, init, batch_windows, momentum=0.9):
    delta = init.clone().requires_grad_(True)
    best_d, best_l = init.detach().clone(), float("inf")
    g_acc = torch.zeros_like(delta)
    for t in range(T_steps):
        loss_val, g = _grad_band(model, tau, delta, cmds, R_loss_t, inwin_t, l_s, alpha, K, b, b_low, batch_windows)
        if loss_val < best_l:
            best_l, best_d = loss_val, delta.detach().clone()
        with torch.no_grad():
            g_acc = momentum * g_acc + g / (g.abs().mean() + 1e-12)
            step = (2.5 * eps / T_steps) * (0.15 ** (t / max(1, T_steps - 1)))
            delta = torch.clamp(delta - step * torch.sign(g_acc), -eps, eps) * F_mask
        delta.requires_grad_(True)
    lf, _ = _grad_band(model, tau, delta, cmds, R_loss_t, inwin_t, l_s, alpha, K, b, b_low, batch_windows)
    if lf < best_l:
        best_d = delta.detach().clone()
    return best_d

def _square_polish(F_lo, F_hi, eps, delta0, proxy_fn, true_miss_fn, query_budget, max_iters, gen, p_init=0.3):
    delta = delta0.clone()
    best = proxy_fn(delta)
    nq = 0
    Flen = F_hi - F_lo + 1
    if true_miss_fn(delta):
        return delta, True, nq
    for it in range(max_iters):
        frac = p_init * max(0.1, 1.0 - it / max(1, max_iters))
        s = max(1, int(round(frac * Flen)))
        start = F_lo + int(torch.randint(0, max(1, Flen - s + 1), (1,), generator=gen).item())
        sign = 1.0 if torch.rand(1, generator=gen).item() < 0.5 else -1.0
        cand = delta.clone()
        cand[start:start + s] = sign * eps
        sc = proxy_fn(cand)
        if sc < best:
            delta, best = cand, sc
            if nq < query_budget:
                nq += 1
                if true_miss_fn(delta):
                    return delta, True, nq
        if nq >= query_budget:
            break
    return delta, False, nq

@dataclass
class CeilingConfig:
    band_T_steps: int = 100
    band_restarts: int = 5
    band_alpha_frac: float = 0.2
    square_iters: int = 200
    query_budget: int = 80
    batch_windows: int = 512
    max_grad_windows: int = 2000
    seed: int = 0

def _make_true_miss(chan_id, model, train_features, tele, cmds, cfg, label):
    def f(delta):
        E = detect(chan_id, model, train_features, tele + delta.detach().cpu().numpy(), cmds, cfg)
        return missed(E, label)
    return f

def _deanomalized_init(tele, label, eps, F_np, T):
    x = np.asarray(tele, np.float32)
    a, b = int(label[0]), int(label[1])
    pre_lo, post_hi = max(0, a - 20), min(T, b + 21)
    has_pre, has_post = a > pre_lo, post_hi > b + 1
    left = float(np.median(x[pre_lo:a])) if has_pre else None
    right = float(np.median(x[b + 1:post_hi])) if has_post else None
    if left is None:
        left = right
    if right is None:
        right = left
    if left is None and right is None:
        left = right = float(np.median(x))
    idx = np.arange(a, b + 1)
    target = x.copy()
    target[a:b + 1] = left + (right - left) * (idx - a) / max(1, b - a)
    delta = np.clip(target - x, -eps, eps) * F_np
    return delta.astype(np.float32)

def ceiling_attack(chan_id, model, train_features, tele, cmds, labels, label, cfg, ccfg, eps, return_delta=False):
    device = next(model.parameters()).device
    T = len(tele)
    l_s = cfg.l_s
    span = _ewma_span(cfg)
    alpha = 2.0 / (span + 1)
    K = 8 * span
    tau_t = torch.tensor(np.asarray(tele, np.float32), device=device)
    cmds_t = torch.tensor(cmds.astype(np.float32), device=device)
    F = footprint_mask(label, l_s, cfg.error_buffer, T)
    F_t = torch.tensor(F, device=device)
    fnz = np.where(F > 0)[0]
    F_lo, F_hi = int(fnz[0]), int(fnz[-1])
    b, b_low, R_loss_t, inwin_t, info = _region_and_baseline(model, tau_t, cmds_t, label, cfg, alpha, K, ccfg.batch_windows, F_lo, F_hi, T, ccfg.max_grad_windows)
    proxy = lambda d: _proxy_loss(model, tau_t, d, cmds_t, R_loss_t, inwin_t, l_s, alpha, K, b, b_low, ccfg.batch_windows)
    true_miss = _make_true_miss(chan_id, model, train_features, tele, cmds, cfg, label)
    zeros = torch.zeros(T, device=device)
    out = {"chan": chan_id, "label": [int(label[0]), int(label[1])], "eps": float(eps), "span": int(span), "baseline_b": float(b), "baseline_b_low": float(b_low), "loss_clean": proxy(zeros), **{f"region_{k}": v for k, v in info.items()}}
    gen = torch.Generator().manual_seed(_label_seed(ccfg.seed, chan_id, label))
    deanom = torch.tensor(_deanomalized_init(tele, label, eps, F, T), device=device)
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
    E = detect(chan_id, model, train_features, tele + final.detach().cpu().numpy(), cmds, cfg)
    out["ceiling_collateral"] = int(collateral_fp(E, labels))
    if return_delta:
        with torch.no_grad():
            esc = _es_over(model, tau_t, torch.zeros(T, device=device), cmds_t, R_loss_t, l_s, alpha, K, ccfg.batch_windows, requires_grad=False)
            esp = _es_over(model, tau_t, best, cmds_t, R_loss_t, l_s, alpha, K, ccfg.batch_windows, requires_grad=False)
        out["best_delta"] = best.detach().cpu().numpy().astype(float).tolist()
        out["es_clean"] = esc.detach().cpu().numpy().astype(float).tolist()
        out["es_pert"] = esp.detach().cpu().numpy().astype(float).tolist()
        out["es_R0"] = int(R_loss_t[0].item())
    return out