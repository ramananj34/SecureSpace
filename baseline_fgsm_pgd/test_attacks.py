from __future__ import annotations
import sys
from pathlib import Path
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
_paths_to_add = [str(_THIS), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction")]
for _p in _paths_to_add:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from VENDOR_telemanom import VendoredConfig, batch_predict
from pipeline import load_trained_model, evaluate_anomalies
from fgsm_pgd_attacks import (DEVICE, prepare_model, load_streams, make_channel, detect, missed, suppression_windows, footprint_mask, predict_windows, surrogate_value, grad_loss, fgsm, pgd, attack_anomaly, AttackConfig)

CHAN = "A-2"
DATA_DIR = _ROOT / "smap_msl_data"
RUNS_DIR = _ROOT / "runs"
FROZEN_E_SEQ = [(4450, 4659)]

_results = []

def check(name, cond, detail=""):
    ok = bool(cond)
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok

print(f"device={DEVICE}  channel={CHAN}")
tele, cmds, train_features, labels, T = load_streams(CHAN, DATA_DIR)
n_feat = train_features.shape[1]
cfg = VendoredConfig()
model = prepare_model(load_trained_model(RUNS_DIR / CHAN / "model.pt", n_features=n_feat, device=DEVICE))
l_s, buf, n_pred = cfg.l_s, cfg.error_buffer, cfg.n_predictions
n_windows = T - l_s - n_pred
label = labels[0]
P = suppression_windows(label, l_s, buf, n_windows)
F = footprint_mask(label, l_s, buf, T)
fp = np.where(F > 0)[0]
tau_t = torch.tensor(tele.astype(np.float32), device=DEVICE)
cmds_t = torch.tensor(cmds.astype(np.float32), device=DEVICE)
F_t = torch.tensor(F, device=DEVICE)
print(f"T={T}  n_feat={n_feat}  n_windows={n_windows}  labels={labels}")
print(f"label={label}  |P|={len(P)}  footprint=[{fp[0]}, {fp[-1]}] ({len(fp)} steps)")
print("\n1. differentiable forward == vendored pipeline")
ch = make_channel(CHAN, train_features, tele, cmds, cfg)
batch_predict(model, ch, cfg, method="first", device=DEVICE)
yhat = np.asarray(ch.y_hat).ravel()
Xtest_P = torch.tensor(np.asarray(ch.X_test)[P].astype(np.float32), device=DEVICE)
with torch.no_grad():
    direct = model(Xtest_P)[:, 0].cpu().numpy()
mine = predict_windows(model, tau_t, cmds_t, P, l_s).cpu().numpy()
check("my forward == model(X_test[P])[:,0]", np.allclose(mine, direct, atol=1e-5), f"max|d|={np.max(np.abs(mine - direct)):.2e}")
check("my forward == y_hat[P]", np.allclose(mine, yhat[P], atol=1e-5), f"max|d|={np.max(np.abs(mine - yhat[P])):.2e}")
print("\n2. gradient (chunked == single) and finite-difference")
mid = len(P) // 2
Psmall = P[max(0, mid - 32): mid + 32]
d_a = torch.zeros(T, device=DEVICE, requires_grad=True)
_, g_full = grad_loss(model, tau_t, d_a, cmds_t, Psmall, l_s, batch_windows=100000)
d_b = torch.zeros(T, device=DEVICE, requires_grad=True)
_, g_ch = grad_loss(model, tau_t, d_b, cmds_t, Psmall, l_s, batch_windows=8)
check("chunked grad == single-batch grad", torch.allclose(g_full, g_ch, atol=1e-6), f"max|d|={float((g_full - g_ch).abs().max()):.2e}")
torch.manual_seed(0)
u = torch.randn(T, device=DEVICE) * F_t
u = u / (u.norm() + 1e-12)
h = 1e-2
Lp = surrogate_value(model, tau_t, (h * u), cmds_t, Psmall, l_s)
Lm = surrogate_value(model, tau_t, (-h * u), cmds_t, Psmall, l_s)
fd = (Lp - Lm) / (2 * h)
analytic = float((g_full * u).sum())
rel = abs(fd - analytic) / (abs(analytic) + 1e-9)
check("finite-diff matches autograd (rel<0.1)", rel < 0.1, f"fd={fd:.4e} analytic={analytic:.4e} rel={rel:.3f}")
print("\n3. telemetry-only (commands untouched)")
d3 = fgsm(model, tau_t, cmds_t, F_t, P, eps=2 ** -3, l_s=l_s).cpu().numpy()
tf_clean = np.concatenate([tele.astype(np.float32)[:, None], cmds], axis=1).astype(np.float32)
tf_pert = np.concatenate([(tele + d3).astype(np.float32)[:, None], cmds], axis=1).astype(np.float32)
check("command columns unchanged", np.array_equal(tf_clean[:, 1:], tf_pert[:, 1:]))
changed = np.abs(d3) > 0
check("telemetry perturbed only on footprint", bool(np.all(changed <= (F > 0)) and changed.any()), f"n_changed={int(changed.sum())}")
print("\n4. L_inf projection")
eps = 2 ** -4
d4 = pgd(model, tau_t, cmds_t, F_t, P, eps, T_steps=10, alpha=eps / 4, l_s=l_s, init=torch.zeros(T, device=DEVICE))
mx = float(d4.abs().max())
off = float((d4 * (1 - F_t)).abs().max())
check("||delta||_inf <= eps", mx <= eps + 1e-6, f"max|delta|={mx:.5f} eps={eps:.5f}")
check("delta == 0 outside footprint", off == 0.0, f"max off-F={off:.2e}")
print("\n5. FGSM == 1-step PGD(alpha=eps)")
eps = 2 ** -5
d_f = fgsm(model, tau_t, cmds_t, F_t, P, eps, l_s)
d_p = pgd(model, tau_t, cmds_t, F_t, P, eps, T_steps=1, alpha=eps, l_s=l_s, init=torch.zeros(T, device=DEVICE))
check("FGSM == PGD(T=1, alpha=eps)", torch.allclose(d_f, d_p, atol=1e-6), f"max|d|={float((d_f - d_p).abs().max()):.2e}")
print("\n6. PGD reduces the surrogate")
eps = 2 ** -3
L0 = surrogate_value(model, tau_t, torch.zeros(T, device=DEVICE), cmds_t, P, l_s)
dT = pgd(model, tau_t, cmds_t, F_t, P, eps, T_steps=20, alpha=eps / 4, l_s=l_s, init=torch.zeros(T, device=DEVICE))
LT = surrogate_value(model, tau_t, dT, cmds_t, P, l_s)
check("L(delta_T) < L(0)", LT < L0, f"L0={L0:.4e} -> LT={LT:.4e}")
print("\n7. eval-mode determinism")
p1 = predict_windows(model, tau_t, cmds_t, P[:32], l_s).cpu().numpy()
p2 = predict_windows(model, tau_t, cmds_t, P[:32], l_s).cpu().numpy()
check("two forwards identical", np.array_equal(p1, p2))
print("\n8. clean baseline reproduces frozen result")
E0 = detect(CHAN, model, train_features, tele, cmds, cfg)
m = evaluate_anomalies(E0, labels)
check("f0_5 == 1.000", abs(m["f0_5"] - 1.0) < 1e-9, f"f0_5={m['f0_5']:.3f} E={E0}")
check("E_seq == frozen [(4450,4659)]", [tuple(e) for e in E0] == FROZEN_E_SEQ, f"E={E0}")
print("\n9. end-to-end smoke + oracle sanity")
check("clean (delta=0): label is a HIT (not missed)", not missed(E0, label), f"label={label}")
atk = AttackConfig(T_steps=20, n_restarts=3)
res = attack_anomaly(CHAN, model, train_features, tele, cmds, labels, label, cfg, atk, eps=2 ** -3)
check("attack ran end-to-end", "pgd_missed" in res)
print(f"  result: {res}")
print(f"  -> FGSM missed={res['fgsm_missed']}, PGD missed={res['pgd_missed']} "
      f"(restarts_used={res['pgd_restarts_used']}); surrogate "
      f"{res['surrogate_clean']:.3e} -> pgd {res['pgd_surrogate']:.3e}  [informational]")
print(f"\nsummary: {sum(_results)}/{len(_results)} checks passed")
sys.exit(0 if all(_results) else 1)