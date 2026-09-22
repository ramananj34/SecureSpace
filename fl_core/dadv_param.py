from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "adv_subspace"), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from jacobian_dadv import adversarial_subspace, dadv_all_three, c_quant_from_quantizer

try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

def _flat_grad(params_grads):
    return np.concatenate([g.reshape(-1).detach().cpu().numpy() for g in params_grads])

def param_jacobian(model, x_windows, f_score=None, device=None):
    if not _HAVE_TORCH:
        raise RuntimeError("torch required for param_jacobian")
    import torch
    f_score = f_score or (lambda m, x: m(x)[:, :1])
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = [p for p in model.parameters()]
    X = np.asarray(x_windows, np.float32)
    if X.ndim == 2:
        X = X[None]
    rows = []
    prev_flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(True)
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            for a in range(X.shape[0]):
                xa = torch.tensor(X[a][None], device=device, requires_grad=False)
                out = f_score(model, xa)
                g = torch.autograd.grad(out[0, 0], params, retain_graph=False)
                rows.append(_flat_grad(g))
    finally:
        for p, flag in zip(params, prev_flags):
            p.requires_grad_(flag)
    return np.stack(rows)

def v_theta_single_anchor(model, x_window, tau=0.1, f_score=None, device=None):
    J = param_jacobian(model, x_window, f_score=f_score, device=device)
    d_grad = J.shape[1]
    sub = adversarial_subspace(J, cols=np.arange(d_grad), tau=tau)
    return {"J_shape": list(J.shape), "d_adv_prime_int": sub["d_adv_int"],
            "d_adv_prime_effective_trace": float(np.trace(sub["Pi_V"])),
            "singular_values_top5": [float(x) for x in sub["singular_values"][:5]],
            "anchors": 1}

def v_theta_stacked(model, x_windows, tau=0.1, f_score=None, device=None):
    J = param_jacobian(model, x_windows, f_score=f_score, device=device)
    d_grad = J.shape[1]
    sub = adversarial_subspace(J, cols=np.arange(d_grad), tau=tau)
    return {"J_shape": list(J.shape), "d_adv_prime_int": sub["d_adv_int"],
            "d_adv_prime_effective_trace": float(np.trace(sub["Pi_V"])),
            "singular_values": [float(x) for x in sub["singular_values"]],
            "Pi_V": sub["Pi_V"], "anchors": int(J.shape[0])}

def dadv_prime_from_jacobian(J, tau=0.1):
    J = np.asarray(J, np.float64)
    d_grad = J.shape[1]
    sub = adversarial_subspace(J, cols=np.arange(d_grad), tau=tau)
    return {"d_adv_prime_int": sub["d_adv_int"],
            "d_adv_prime_effective_trace": float(np.trace(sub["Pi_V"])),
            "Pi_V": sub["Pi_V"], "singular_values": [float(x) for x in sub["singular_values"]]}