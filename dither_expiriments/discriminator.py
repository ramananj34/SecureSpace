from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

def _avg_ranks(a):
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); start = csum - counts
    avg = (start + csum - 1) / 2.0 + 1.0
    return avg[inv]

def roc_auc(scores, labels):
    scores = np.asarray(scores, np.float64); labels = np.asarray(labels, np.int64)
    n_pos = int(labels.sum()); n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _avg_ranks(scores)
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def slots_from_stream(tele_scaled, quantizer, window=100):
    q = quantizer
    x = np.asarray(tele_scaled, np.float64)
    n = (x.size // window)
    if n == 0:
        return np.zeros((0, window * q.b), np.uint8)
    x = x[:n * window].reshape(n, window)
    out = np.zeros((n, window * q.b), np.uint8)
    for i in range(n):
        out[i] = np.asarray(q.levels_to_bits(q.quantize(x[i])), np.uint8).reshape(-1)
    return out

def position_marginals(bit_matrix, smoothing=0.5):
    B = np.asarray(bit_matrix, np.float64)
    if B.ndim != 2 or B.shape[0] == 0:
        raise ValueError("bit_matrix must be (n_examples, L) with n_examples>0")
    n = B.shape[0]
    return (B.sum(0) + smoothing) / (n + 2.0 * smoothing)

def marginal_llr_scores(bit_matrix, p):
    p = np.clip(np.asarray(p, np.float64), 1e-9, 1 - 1e-9)
    logit = np.log(p / (1.0 - p))
    return np.asarray(bit_matrix, np.float64) @ logit

def _perm_rows(B, rng):
    B = np.asarray(B, np.uint8)
    out = np.empty_like(B)
    for i in range(B.shape[0]):
        out[i] = B[i, rng.permutation(B.shape[1])]
    return out

def marginal_discriminator_auc(real_train, real_test, seed=0, smoothing=0.5):
    p = position_marginals(real_train, smoothing=smoothing)
    rng = np.random.default_rng(seed)
    neg = _perm_rows(real_test, rng)
    s_pos = marginal_llr_scores(real_test, p)
    s_neg = marginal_llr_scores(neg, p)
    scores = np.concatenate([s_pos, s_neg])
    labels = np.concatenate([np.ones(s_pos.size, int), np.zeros(s_neg.size, int)])
    return {"auc_marginal": roc_auc(scores, labels),
            "n_pos": int(s_pos.size), "n_neg": int(s_neg.size)}

def marginal_recovery_reconstruction(m_obs, p):
    m_obs = np.asarray(m_obs, np.uint8).ravel()
    L = m_obs.size; w = int(m_obs.sum())
    order = np.argsort(-np.asarray(p, np.float64), kind="mergesort")
    rec = np.zeros(L, np.uint8); rec[order[:w]] = 1
    return rec

def marginal_recovery_permutation(m_obs, p, rng=None):
    m_obs = np.asarray(m_obs, np.uint8).ravel()
    L = m_obs.size; w = int(m_obs.sum())
    p = np.asarray(p, np.float64)
    if rng is None:
        order = np.argsort(-p, kind="mergesort")
    else:
        order = np.lexsort((rng.random(L), -p))
    dst_ones = np.sort(order[:w]); dst_zeros = np.sort(order[w:])
    src_ones = np.where(m_obs == 1)[0]; src_zeros = np.where(m_obs == 0)[0]
    inv_pi = np.empty(L, np.int64)
    inv_pi[dst_ones] = src_ones
    inv_pi[dst_zeros] = src_zeros
    pi = np.empty(L, np.int64); pi[inv_pi] = np.arange(L, dtype=np.int64)
    return pi

if _HAVE_TORCH:
    class FrameDiscriminator(nn.Module):
        def __init__(self, L, hidden=(256, 64)):
            super().__init__()
            layers = []
            d = L
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def _make_pos_neg(real_bits, seed):
        rng = np.random.default_rng(seed)
        pos = np.asarray(real_bits, np.float32)
        neg = _perm_rows(pos.astype(np.uint8), rng).astype(np.float32)
        X = np.concatenate([pos, neg], 0)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
        return X, y

    def train_discriminator(real_bits, hidden=(256, 64), epochs=40, lr=1e-3, batch=256, train_frac=0.7, seed=0, device=None, center=True, verbose=False):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        real_bits = np.asarray(real_bits, np.uint8)
        n = real_bits.shape[0]
        n_tr = int(n * train_frac)
        Xtr, ytr = _make_pos_neg(real_bits[:n_tr], seed)
        Xte, yte = _make_pos_neg(real_bits[n_tr:], seed + 1)
        L = real_bits.shape[1]
        if center:
            Xtr = Xtr - 0.5; Xte = Xte - 0.5
        Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
        Xte_t = torch.tensor(Xte, device=device); yte_t = torch.tensor(yte, device=device)
        model = FrameDiscriminator(L, hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
        idx = np.arange(Xtr.shape[0])
        rng = np.random.default_rng(seed)
        best_auc, best_state = -1.0, None
        for ep in range(epochs):
            model.train(); rng.shuffle(idx)
            for s in range(0, idx.size, batch):
                b = idx[s:s + batch]
                opt.zero_grad()
                loss = bce(model(Xtr_t[b]), ytr_t[b]); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                ste = model(Xte_t).cpu().numpy()
            auc_te = roc_auc(ste, yte)
            if auc_te > best_auc:
                best_auc = auc_te
                best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  ep{ep+1}/{epochs} auc_test={auc_te:.4f}")
        with torch.no_grad():
            model.eval()
            auc_tr = roc_auc(model(Xtr_t).cpu().numpy(), ytr)
        return best_state, {"auc_joint_test": float(best_auc), "auc_joint_train": float(auc_tr),
                            "L": int(L), "n_real": int(n), "n_train_real": int(n_tr),
                            "hidden": list(hidden), "epochs": int(epochs)}

    def save_discriminator(out_dir, name, state, metrics):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        if state is not None:
            torch.save(state, out / f"{name}.pt")
        tmp = (out / f"{name}.json.tmp")
        json.dump(metrics, open(tmp, "w"), indent=2); tmp.replace(out / f"{name}.json")

def necessary_condition_verdict(auc_marginal, auc_joint=None, joint_gap_thresh=0.03, marginal_signal_thresh=0.55):
    v = {
        "auc_marginal": float(auc_marginal),
        "marginal_signal": bool(auc_marginal > marginal_signal_thresh),
    }
    if auc_joint is not None:
        gap = float(auc_joint) - float(auc_marginal)
        v.update({"auc_joint": float(auc_joint), "joint_minus_marginal": gap,
                  "usable_joint_beyond_marginal": bool(gap > joint_gap_thresh)})
    return v