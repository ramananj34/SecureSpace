from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

def fedavg(updates, weights=None):
    U = np.asarray(updates, np.float64)
    if weights is None:
        return U.mean(0)
    w = np.asarray(weights, np.float64); w = w / w.sum()
    return (w[:, None] * U).sum(0)

def coord_median(updates):
    return np.median(np.asarray(updates, np.float64), axis=0)

def krum(updates, f, return_score=False):
    U = np.asarray(updates, np.float64)
    n = U.shape[0]
    if n < 2 * f + 3:
        raise ValueError(f"Krum needs n >= 2f+3 (n={n}, f={f})")
    m = n - f - 2
    sq = np.zeros((n, n))
    for i in range(n):
        d = U - U[i]
        sq[i] = np.einsum("ij,ij->i", d, d)
        sq[i, i] = np.inf
    scores = np.array([np.sort(sq[i])[:m].sum() for i in range(n)])
    sel = int(np.argmin(scores))
    return (U[sel], sel, scores) if return_score else U[sel]

AGGREGATORS = {
    "fedavg": lambda U, f=0: fedavg(U),
    "coord_median": lambda U, f=0: coord_median(U),
    "krum": lambda U, f=0: krum(U, f),
}

def aggregate(name, updates, f=0):
    if name not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; choose from {list(AGGREGATORS)}")
    return AGGREGATORS[name](np.asarray(updates, np.float64), f)

def flatten_params(state_dict):
    keys = sorted(state_dict.keys())
    parts, shapes = [], []
    for k in keys:
        v = state_dict[k].detach().cpu().numpy().ravel()
        parts.append(v); shapes.append((k, state_dict[k].shape))
    return np.concatenate(parts).astype(np.float64), shapes

def unflatten_params(vec, shapes, like_state_dict):
    import torch
    out = {}
    off = 0
    for k, shape in shapes:
        n = int(np.prod(shape))
        arr = np.asarray(vec[off:off + n], np.float32).reshape([int(x) for x in shape])
        out[k] = torch.tensor(arr, dtype=like_state_dict[k].dtype, device=like_state_dict[k].device)
        off += n
    return out

def temporal_partitions(n_timesteps, n_sats, window):
    total = n_timesteps
    per = total // n_sats
    parts = []
    for s in range(n_sats):
        lo = s * per
        hi = (s + 1) * per if s < n_sats - 1 else total
        parts.append((lo, hi))
    return parts

def local_update(model_ctor, global_state, features_slice, cfg, local_epochs=1, lr=1e-3,
                 device=None, seed=0):
    if not _HAVE_TORCH:
        raise RuntimeError("torch required for local_update")
    import torch, torch.nn as nn
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = model_ctor().to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    g0, shapes = flatten_params(model.state_dict())
    X, y = _windows(features_slice, cfg.sequence_length)
    if X.shape[0] == 0:
        return np.zeros_like(g0), shapes
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    with torch.backends.cudnn.flags(enabled=False):
        for _ in range(int(local_epochs)):
            opt.zero_grad()
            pred = model(Xt)[:, 0]
            loss = crit(pred, yt)
            loss.backward(); opt.step()
    g1, _ = flatten_params(model.state_dict())
    return (g1 - g0), shapes

def _windows(features, l_s):
    f = np.asarray(features, np.float32)
    T = f.shape[0]
    n = T - l_s
    if n <= 0:
        return np.zeros((0, l_s, f.shape[1]), np.float32), np.zeros((0,), np.float32)
    X = np.stack([f[i:i + l_s] for i in range(n)])
    y = f[l_s:l_s + n, 0]
    return X, y