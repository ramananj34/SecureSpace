from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "telemanom_reproduction"),
           str(_ROOT / "smap_msl_data"), str(_ROOT / "baseline_fgsm_pgd")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

def default_f_score(model, x):
    return model(x)[:, :1]

def jacobian_fscore(model, window, f_score=None, device=None):
    if not _HAVE_TORCH:
        raise RuntimeError("torch required for jacobian_fscore")
    f_score = f_score or default_f_score
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            try:
                device = next(model.buffers()).device
            except StopIteration:
                device = torch.device("cpu")
    x = torch.tensor(np.asarray(window, np.float32)[None], device=device, requires_grad=True)
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        out = f_score(model, x)
        n_out = out.shape[1]
        rows = []
        for j in range(n_out):
            g = torch.autograd.grad(out[0, j], x, retain_graph=(j < n_out - 1))[0]
            rows.append(g[0].reshape(-1).detach().cpu().numpy())
    return np.stack(rows)

def telemetry_columns(seq_len, input_dim, telem_feature=0):
    return (np.arange(seq_len) * input_dim + telem_feature).astype(np.int64)

def adversarial_subspace(J, cols, tau=0.1):
    cols = np.asarray(cols, np.int64)
    Jr = np.asarray(J, np.float64)[:, cols]
    d = Jr.shape[1]
    U, S, Vt = np.linalg.svd(Jr, full_matrices=False)
    s1 = S[0] if S.size and S[0] > 0 else 0.0
    d_adv_int = int(np.sum(S > tau * s1)) if s1 > 0 else 0
    d_adv_int = max(d_adv_int, 1) if s1 > 0 else 0
    Vbasis = Vt[:d_adv_int].T if d_adv_int > 0 else np.zeros((d, 0))
    Pi_V = Vbasis @ Vbasis.T if d_adv_int > 0 else np.zeros((d, d))
    return {"singular_values": [float(x) for x in S], "d_adv_int": d_adv_int,
            "Vbasis": Vbasis, "Pi_V": Pi_V, "d_restricted": d, "tau": tau}

def dadv_effective_trace(Pi_V, region_idx=None):
    Pi_V = np.asarray(Pi_V, np.float64)
    if region_idx is None:
        return float(np.trace(Pi_V))
    region_idx = np.asarray(region_idx, np.int64)
    return float(np.sum(np.diag(Pi_V)[region_idx]))

def dadv_geometric_dim(Pi_V, region_idx, tol=1e-8):
    region_idx = np.asarray(region_idx, np.int64)
    sub = np.asarray(Pi_V, np.float64)[np.ix_(region_idx, region_idx)]
    return int(np.linalg.matrix_rank(sub, tol=tol))

def dadv_frobenius(Pi_V, region_idx):
    region_idx = np.asarray(region_idx, np.int64)
    sub = np.asarray(Pi_V, np.float64)[np.ix_(region_idx, region_idx)]
    return float(np.sum(sub * sub))

def dadv_all_three(Pi_V, region_idx=None):
    if region_idx is None:
        region_idx = np.arange(np.asarray(Pi_V).shape[0])
    return {"effective_trace": dadv_effective_trace(Pi_V, region_idx),
            "geometric_dim": dadv_geometric_dim(Pi_V, region_idx),
            "frobenius_sq": dadv_frobenius(Pi_V, region_idx)}

def c_quant_from_quantizer(quantizer):
    return float(quantizer.c_quant())

def bit_to_coord_matrix(quantizer, d_coords):
    b = int(quantizer.b); dlsb = float(quantizer.delta_lsb)
    weights = (2.0 ** np.arange(b)) * dlsb
    D = np.zeros((d_coords, d_coords * b))
    for j in range(d_coords):
        D[j, j * b:(j + 1) * b] = weights
    return D