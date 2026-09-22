from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import jacobian_dadv as JD

def test_dadv_recovers_known_rank():
    d, n_out = 50, 5
    rng = np.random.default_rng(0)
    Ur, _ = np.linalg.qr(rng.normal(size=(n_out, n_out)))
    Vr, _ = np.linalg.qr(rng.normal(size=(d, d)))
    S = np.array([10.0, 8.0, 5.0, 0.3, 0.05])
    J = Ur @ np.diag(S) @ Vr[:, :n_out].T
    J = (Ur * S) @ Vr[:, :n_out].T
    res = JD.adversarial_subspace(J, cols=np.arange(d), tau=0.1)
    assert res["d_adv_int"] == 3, res["singular_values"][:5]

def test_pi_v_is_projector_with_correct_trace():
    d, n_out = 40, 4
    rng = np.random.default_rng(1)
    J = rng.normal(size=(n_out, d))
    res = JD.adversarial_subspace(J, cols=np.arange(d), tau=0.1)
    Pi = res["Pi_V"]
    assert np.allclose(Pi, Pi.T) and np.allclose(Pi @ Pi, Pi)
    assert abs(np.trace(Pi) - res["d_adv_int"]) < 1e-9

def test_scalar_fscore_gives_rank1():
    d = 100
    J = np.random.default_rng(2).normal(size=(1, d))
    res = JD.adversarial_subspace(J, cols=np.arange(d), tau=0.1)
    assert res["d_adv_int"] == 1
    assert res["Vbasis"].shape == (d, 1)

def test_d66_three_quantities_distinct():
    d = 60; d_adv = 5
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V = Q[:, :d_adv]; Pi = V @ V.T
    region = np.arange(0, 20)
    q = JD.dadv_all_three(Pi, region)
    # full trace = d_adv exactly
    assert abs(JD.dadv_effective_trace(Pi, None) - d_adv) < 1e-9
    assert 0 < q["effective_trace"] < d_adv
    assert q["geometric_dim"] == d_adv
    assert abs(q["frobenius_sq"] - q["effective_trace"]) > 1e-6

def test_effective_trace_expectation_over_random_region():
    d = 200; d_adv = 10
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Pi = Q[:, :d_adv] @ Q[:, :d_adv].T
    m = 50
    vals = [JD.dadv_effective_trace(Pi, rng.choice(d, m, replace=False)) for _ in range(300)]
    expected = (m / d) * d_adv
    assert abs(np.mean(vals) - expected) < 0.15 * expected

def test_D_DT_equals_cquant_I():
    from smap_msl_dataset_api import Quantizer
    q = Quantizer()
    d = 10
    D = JD.bit_to_coord_matrix(q, d)
    cq = JD.c_quant_from_quantizer(q)
    assert np.allclose(D @ D.T, cq * np.eye(d))
    assert abs(cq - (q.n_levels**2 - 1) / 3.0 * q.delta_lsb**2) < 1e-12

def test_telemetry_columns():
    cols = JD.telemetry_columns(seq_len=250, input_dim=25, telem_feature=0)
    assert cols.shape == (250,)
    assert cols[0] == 0 and cols[1] == 25 and cols[-1] == 249 * 25

def test_autograd_jacobian_matches_finite_diff():
    try:
        import torch
    except Exception:
        print("  (torch absent: autograd Jacobian test skipped)"); return
    L, F = 5, 3
    a = torch.tensor(np.random.default_rng(5).normal(size=(L * F,)), dtype=torch.float32)
    class Lin(torch.nn.Module):
        def forward(self, x):
            return (x.reshape(x.shape[0], -1) @ a)[:, None]
    m = Lin()
    def f_score(model, x): return model(x)
    win = np.random.default_rng(6).normal(size=(L, F)).astype(np.float32)
    J = JD.jacobian_fscore(m, win, f_score=f_score)
    assert J.shape == (1, L * F)
    assert np.allclose(J[0], a.numpy(), atol=1e-5)

def test_jacobian_through_lstm_in_eval_mode():
    try:
        import torch, torch.nn as nn
    except Exception:
        print("  (torch absent: LSTM-eval Jacobian test skipped)"); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, F, H = 20, 4, 8
    class TinyLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(F, H, num_layers=2, batch_first=True)
            self.fc = nn.Linear(H, 3)
        def forward(self, x):
            o, _ = self.lstm(x); return self.fc(o[:, -1, :])
    m = TinyLSTM().to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    win = np.random.default_rng(0).normal(size=(L, F)).astype(np.float32)
    def f_score(model, x): return model(x)[:, :1]
    J = JD.jacobian_fscore(m, win, f_score=f_score, device=dev)
    assert J.shape == (1, L * F) and np.isfinite(J).all()

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")