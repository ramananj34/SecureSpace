from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "adv_subspace")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import dadv_param as DP
import theorem5_energy as T5

def test_dadv_prime_rank1_single_anchor():
    J = np.random.default_rng(0).normal(size=(1, 200))
    r = DP.dadv_prime_from_jacobian(J, tau=0.1)
    assert r["d_adv_prime_int"] == 1

def test_dadv_prime_multi_anchor_higher_rank():
    J = np.random.default_rng(1).normal(size=(50, 200))
    r = DP.dadv_prime_from_jacobian(J, tau=0.1)
    assert r["d_adv_prime_int"] > 1
    assert r["d_adv_prime_int"] <= 50

def test_param_jacobian_matches_finite_diff():
    try:
        import torch, torch.nn as nn
    except Exception:
        print("  (torch absent: param-Jacobian finite-diff test skipped)"); return
    d = 30
    class LinInParams(nn.Module):
        def __init__(self):
            super().__init__(); self.theta = nn.Parameter(torch.randn(d))
        def forward(self, x):
            return (x.reshape(x.shape[0], -1) * self.theta).sum(1, keepdim=True)
    m = LinInParams().eval()
    for p in m.parameters(): p.requires_grad_(True)
    x = np.random.default_rng(2).normal(size=(1, 1, d)).astype(np.float32)
    J = DP.param_jacobian(m, x[0], f_score=lambda mm, xx: mm(xx))
    assert J.shape == (1, d)
    assert np.allclose(J[0], x.reshape(-1), atol=1e-5)

def test_param_jacobian_through_lstm_eval():
    try:
        import torch, torch.nn as nn
    except Exception:
        print("  (torch absent: LSTM param-Jacobian test skipped)"); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class TinyLSTM(nn.Module):
        def __init__(self): super().__init__(); self.lstm=nn.LSTM(3,6,2,batch_first=True); self.fc=nn.Linear(6,10)
        def forward(self, x): o,_=self.lstm(x); return self.fc(o[:,-1,:])
    m = TinyLSTM().to(dev).eval()
    x = np.random.default_rng(3).normal(size=(2, 15, 3)).astype(np.float32)
    J = DP.param_jacobian(m, x, device=dev)
    n_params = sum(p.numel() for p in m.parameters())
    assert J.shape == (2, n_params) and np.isfinite(J).all()

def test_theorem5_link_matches_brute_force():
    d, b = 3, 2; k_g = d * b; dlsb = 0.5
    weights = (2.0 ** np.arange(b)) * dlsb
    D = np.zeros((d, k_g))
    for j in range(d): D[j, j*b:(j+1)*b] = weights
    c_quant = float((weights ** 2).sum())
    rng = np.random.default_rng(0)
    Q_, _ = np.linalg.qr(rng.normal(size=(d, d))); Vb = Q_[:, :2]; Pi = Vb @ Vb.T
    g_q = rng.integers(0, 2, k_g)
    res = T5.verify_theorem5_link(Pi, D, g_q, w=3, k_g=k_g, c_quant=c_quant, d_adv_prime_eff=2.0,
                                  eps_prg=0.0, delta_msb=1.0, d_grad=d, b=b, n_mc=60000, seed=1)
    assert res["within_tol"] and res["rel_error_core"] < 0.05

def test_theorem5_eps_prg_term_is_additive_slack():
    d, b = 3, 2; k_g = d * b; dlsb = 0.5
    weights = (2.0 ** np.arange(b)) * dlsb
    D = np.zeros((d, k_g))
    for j in range(d): D[j, j*b:(j+1)*b] = weights
    c_quant = float((weights ** 2).sum())
    Pi = np.outer(*(lambda v: (v, v))(np.linalg.qr(np.random.default_rng(1).normal(size=(d, d)))[0][:, 0]))
    g_q = np.random.default_rng(2).integers(0, 2, k_g)
    res = T5.verify_theorem5_link(Pi, D, g_q, w=3, k_g=k_g, c_quant=c_quant, d_adv_prime_eff=1.0,
                                  eps_prg=0.1, delta_msb=1.0, d_grad=d, b=b, n_mc=20000, seed=3)
    assert res["rhs_with_prg"] >= res["rhs_core"]
    assert res["lhs_le_rhs_with_prg"]

def test_y_max_and_rowblock_bounds():
    assert T5.y_max_bound(100, 1.0) == 100.0
    rb = T5.row_block_bound(5, b=8)
    assert abs(rb - 5 * (255/128)**2) < 1e-6

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")