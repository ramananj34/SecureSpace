from __future__ import annotations
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
_paths_to_add = [str(_THIS), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction")]
for _p in _paths_to_add:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from smap_msl_dataset_api import SMAPMSLChannelDataset
from VENDOR_telemanom import VendoredConfig, Channel, batch_predict, Errors
from pipeline import load_trained_model, evaluate_anomalies, _mask_to_sequences  # noqa: F401
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_model(model):
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model

def load_streams(chan_id: str, data_dir: Path):
    tr = SMAPMSLChannelDataset(chan_id, "train", data_dir=data_dir)
    te = SMAPMSLChannelDataset(chan_id, "test", data_dir=data_dir)
    train_features = np.concatenate([tr.telemetry_scaled[:, None], tr._commands], axis=1).astype(np.float32)
    tele = te.telemetry_scaled.astype(np.float64)
    cmds = te._commands.astype(np.float32)
    labels = _mask_to_sequences(te.anomaly_mask)
    return tele, cmds, train_features, labels, te.n_timesteps

def make_channel(chan_id, train_features, tele_test, cmds_test, cfg, seed=42) -> Channel:
    test_features = np.concatenate([np.asarray(tele_test, np.float32)[:, None], cmds_test.astype(np.float32)],axis=1).astype(np.float32)
    return Channel.from_arrays(chan_id=chan_id, train_arr=train_features, test_arr=test_features, config=cfg, seed=seed)

def detect(chan_id, model, train_features, tele_test, cmds_test, cfg):
    ch = make_channel(chan_id, train_features, tele_test, cmds_test, cfg)
    batch_predict(model, ch, cfg, method="first",device=next(model.parameters()).device)
    errs = Errors(ch, cfg)
    errs.process_batches(ch)
    return list(errs.E_seq)

def _overlap(x, y):
    return not (x[1] < y[0] or y[1] < x[0])

def missed(E_seq, label):
    return not any(_overlap(e, label) for e in E_seq)

def collateral_fp(E_seq, labels):
    return sum(1 for e in E_seq if not any(_overlap(e, l) for l in labels))

def suppression_windows(label, l_s, buf, n_windows):
    a, b = label
    lo = max(0, a - buf - l_s)
    hi = min(n_windows - 1, b + buf - l_s)
    return np.arange(lo, hi + 1) if hi >= lo else np.arange(0, 0)

def footprint_mask(label, l_s, buf, T):
    a, b = label
    lo = max(0, a - l_s - buf)
    hi = min(T - 1, b + buf)
    m = np.zeros(T, dtype=np.float32)
    m[lo:hi + 1] = 1.0
    return m

def _windows_idx(P, l_s, device):
    P_t = torch.as_tensor(P, device=device, dtype=torch.long)
    return P_t, P_t[:, None] + torch.arange(l_s, device=device)[None, :]

@torch.no_grad()
def predict_windows(model, tau, cmds, P, l_s, batch_windows=512, delta=None):
    device = tau.device
    P_t, _ = _windows_idx(P, l_s, device)
    arange_ls = torch.arange(l_s, device=device)
    out = []
    for s in range(0, len(P), batch_windows):
        idx = P_t[s:s + batch_windows]
        win = idx[:, None] + arange_ls[None, :]
        tele_win = tau[win] + (delta[win] if delta is not None else 0.0)
        x = torch.cat([tele_win[..., None], cmds[win]], dim=-1)
        out.append(model(x)[:, 0])
    return torch.cat(out)

@torch.no_grad()
def surrogate_value(model, tau, delta, cmds, P, l_s, batch_windows=512):
    device = tau.device
    preds = predict_windows(model, tau, cmds, P, l_s, batch_windows, delta)
    P_t = torch.as_tensor(P, device=device, dtype=torch.long)
    tgt = tau[P_t + l_s] + (delta[P_t + l_s] if delta is not None else 0.0)
    return float(((preds - tgt) ** 2).mean())

def grad_loss(model, tau, delta, cmds, P, l_s, batch_windows=512):
    delta.grad = None
    device = tau.device
    P_t = torch.as_tensor(P, device=device, dtype=torch.long)
    arange_ls = torch.arange(l_s, device=device)
    nP = len(P)
    total = 0.0
    with torch.backends.cudnn.flags(enabled=False):
        for s in range(0, nP, batch_windows):
            idx = P_t[s:s + batch_windows]
            win = idx[:, None] + arange_ls[None, :]
            tele_win = tau[win] + delta[win]
            x = torch.cat([tele_win[..., None], cmds[win]], dim=-1)
            pred = model(x)[:, 0]
            tgt = tau[idx + l_s] + delta[idx + l_s]
            partial = ((pred - tgt) ** 2).sum() / nP
            partial.backward()
            total += float(partial.detach())
    return total, delta.grad.detach().clone()

def fgsm(model, tau, cmds, F_mask, P, eps, l_s, batch_windows=512):
    T = tau.shape[0]
    delta = torch.zeros(T, device=tau.device, requires_grad=True)
    _, g = grad_loss(model, tau, delta, cmds, P, l_s, batch_windows)
    with torch.no_grad():
        return ((-eps * torch.sign(g)) * F_mask).detach()

def pgd(model, tau, cmds, F_mask, P, eps, T_steps, alpha, l_s, init, batch_windows=512):
    delta = init.clone().requires_grad_(True)
    for _ in range(T_steps):
        _, g = grad_loss(model, tau, delta, cmds, P, l_s, batch_windows)
        with torch.no_grad():
            delta = torch.clamp(delta - alpha * torch.sign(g), -eps, eps) * F_mask
        delta.requires_grad_(True)
    return delta.detach()

def _label_seed(base, chan_id, label):
    h = hashlib.md5(f"{chan_id}|{label[0]}|{label[1]}".encode()).hexdigest()
    return (base + int(h[:8], 16)) % (2 ** 31)

def _rand_init(eps, F_mask, T, gen):
    u = (torch.rand(T, generator=gen) * 2 - 1) * eps
    return (u.to(F_mask.device) * F_mask)

@dataclass
class AttackConfig:
    T_steps: int = 40
    n_restarts: int = 5
    batch_windows: int = 512
    seed: int = 0
    eps_grid: tuple = (2 ** -8, 2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3)

def attack_anomaly(chan_id, model, train_features, tele, cmds, labels, label, cfg: VendoredConfig, atk: AttackConfig, eps: float):
    device = next(model.parameters()).device
    T = len(tele)
    l_s, buf, n_pred = cfg.l_s, cfg.error_buffer, cfg.n_predictions
    n_windows = T - l_s - n_pred
    P = suppression_windows(label, l_s, buf, n_windows)
    F = footprint_mask(label, l_s, buf, T)
    tau_t = torch.tensor(np.asarray(tele, np.float32), device=device)
    cmds_t = torch.tensor(cmds.astype(np.float32), device=device)
    F_t = torch.tensor(F, device=device)
    def eval_delta(d_t):
        tele_p = tele + d_t.cpu().numpy()
        return detect(chan_id, model, train_features, tele_p, cmds, cfg)
    L0 = surrogate_value(model, tau_t, torch.zeros(T, device=device), cmds_t, P, l_s, atk.batch_windows)
    out = {"chan": chan_id, "label": tuple(label), "eps": float(eps), "nP": int(len(P)), "surrogate_clean": L0}
    d = fgsm(model, tau_t, cmds_t, F_t, P, eps, l_s, atk.batch_windows)
    E = eval_delta(d)
    out["fgsm_missed"] = bool(missed(E, label))
    out["fgsm_surrogate"] = surrogate_value(model, tau_t, d, cmds_t, P, l_s, atk.batch_windows)
    out["fgsm_collateral"] = int(collateral_fp(E, labels))
    gen = torch.Generator().manual_seed(_label_seed(atk.seed, chan_id, label))
    pgd_hit_miss, used, last_E, d_best = False, 0, None, None
    for r in range(atk.n_restarts):
        init = torch.zeros(T, device=device) if r == 0 else _rand_init(eps, F_t, T, gen)
        d = pgd(model, tau_t, cmds_t, F_t, P, eps, atk.T_steps, eps / 4.0, l_s,
                init, atk.batch_windows)
        last_E = eval_delta(d)
        d_best = d
        used = r + 1
        if missed(last_E, label):
            pgd_hit_miss = True
            break
    out["pgd_missed"] = bool(pgd_hit_miss)
    out["pgd_restarts_used"] = used
    out["pgd_surrogate"] = surrogate_value(model, tau_t, d_best, cmds_t, P, l_s, atk.batch_windows)
    out["pgd_collateral"] = int(collateral_fp(last_E, labels))
    return out