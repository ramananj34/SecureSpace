from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "baseline_fgsm_pgd"),
           str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def predict_targets_torch(model, tele, cmds, P, l_s, batch_windows=512):
    import torch
    from fgsm_pgd_attacks import predict_windows
    device = next(model.parameters()).device
    tau = torch.tensor(np.asarray(tele, np.float32), device=device)
    cmds_t = torch.tensor(np.asarray(cmds, np.float32), device=device)
    P = np.asarray(P, np.int64)
    preds = predict_windows(model, tau, cmds_t, P, int(l_s), int(batch_windows))
    return preds.detach().cpu().numpy()


def _target_windows(block_start, block_len, l_s, T):
    t_lo = max(int(block_start), int(l_s))
    t_hi = min(int(block_start) + int(block_len), int(T))
    if t_hi <= t_lo:
        return np.zeros(0, np.int64)
    return np.arange(t_lo - l_s, t_hi - l_s, dtype=np.int64)


def window_reorder_error(model, tele, cmds, block_start, block_len, order, l_s, predict_fn=None, batch_windows=512):
    predict_fn = predict_fn or predict_targets_torch
    tele = np.asarray(tele, np.float64).copy()
    T = tele.size
    order = np.asarray(order, np.int64)
    block = tele[block_start:block_start + block_len].copy()
    tele[block_start:block_start + block_len] = block[order]
    P = _target_windows(block_start, block_len, l_s, T)
    if P.size == 0:
        return 0.0, 0
    preds = np.asarray(predict_fn(model, tele, cmds, P, l_s, batch_windows), np.float64).ravel()
    tgt = tele[P + l_s]
    return float(np.mean((preds - tgt) ** 2)), int(P.size)


def lstm_plausibility_fitness_factory(model, tele, cmds, block_start, block_len, l_s, predict_fn=None, batch_windows=512): 
    def fitness(order):
        err, _ = window_reorder_error(model, tele, cmds, block_start, block_len, order, l_s, predict_fn=predict_fn, batch_windows=batch_windows)
        return -err
    return fitness


def shuffle_sensitivity(model, tele, cmds, block_start, block_len, l_s, n_shuffles=200, seed=0, predict_fn=None, batch_windows=512):
    true_err, npred = window_reorder_error(model, tele, cmds, block_start, block_len, np.arange(block_len), l_s, predict_fn=predict_fn, batch_windows=batch_windows)
    rng = np.random.default_rng(seed)
    errs = np.empty(n_shuffles)
    for i in range(n_shuffles):
        errs[i], _ = window_reorder_error(model, tele, cmds, block_start, block_len, rng.permutation(block_len), l_s, predict_fn=predict_fn, batch_windows=batch_windows)
    mu, sd = float(errs.mean()), float(errs.std())
    return {
        "block_start": int(block_start), "block_len": int(block_len), "n_pred_windows": int(npred),
        "true_err": float(true_err), "shuffle_err_mean": mu, "shuffle_err_std": sd,
        "frac_shuffles_worse_than_true": float(np.mean(errs > true_err)),
        "z_true_below_shuffles": float((mu - true_err) / sd) if sd > 0 else float("nan"),
        "n_shuffles": int(n_shuffles),
    }