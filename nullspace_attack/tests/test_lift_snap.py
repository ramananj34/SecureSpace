import sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "smap_msl_data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smap_msl_dataset_api import Quantizer
from nullspace_attack import lift_to_bits, snap_to_lattice

def test_lift_exact_for_linear_loss():
    q = Quantizer()
    rng = np.random.default_rng(0)
    n = 32
    tele = rng.uniform(-0.9, 0.9, n)
    x_q = q.quantize(tele)
    tele_q = q.dequantize(x_q)
    w = rng.standard_normal(n)
    g_F = lift_to_bits(w, x_q, q)
    max_err = 0.0
    for j in range(n):
        for i in range(q.b):
            lv2 = int(x_q[j]) ^ (1 << i)
            d_dq = q.dequantize(np.array([lv2]))[0] - tele_q[j]
            dL_exact = w[j] * d_dq
            max_err = max(max_err, abs(dL_exact - g_F[j, i]))
    assert max_err < 1e-9, f"linear lift not exact: max err {max_err:.3e}"
    print(f"[ok] linear loss: lift_to_bits exact, max|ΔL − g_F| = {max_err:.2e}")

def test_lift_lsb_accurate_for_nonlinear_loss():
    q = Quantizer()
    rng = np.random.default_rng(1)
    n = 16
    tele = rng.uniform(-0.8, 0.8, n)
    x_q = q.quantize(tele)
    tele_q = q.dequantize(x_q)
    A = rng.standard_normal((n, n)) * 0.3
    H = A @ A.T
    L = lambda t: 0.5 * float(t @ (H @ t))
    g_R = H @ tele_q
    g_F = lift_to_bits(g_R, x_q, q)
    rel = {i: [] for i in range(q.b)}
    for j in range(n):
        for i in range(q.b):
            lv2 = int(x_q[j]) ^ (1 << i)
            t2 = tele_q.copy()
            t2[j] = q.dequantize(np.array([lv2]))[0]
            dL = L(t2) - L(tele_q)
            denom = abs(g_F[j, i]) + 1e-12
            rel[i].append(abs(dL - g_F[j, i]) / denom)
    lsb_med = float(np.median(rel[0]))
    msb_med = float(np.median(rel[q.b - 1]))
    assert lsb_med < 0.05, f"LSB first-order accuracy poor: {lsb_med:.3e}"
    print(f"[ok] nonlinear loss: LSB rel-err median {lsb_med:.2e} (first-order OK); "
          f"MSB rel-err median {msb_med:.2e} (large — §6.13 nonlinearity, expected)")

def test_snap_xor_identity():
    q = Quantizer()
    rng = np.random.default_rng(2)
    n = 200
    tele = rng.uniform(-1.2, 1.2, n)
    x_q = q.quantize(tele)
    tele_q = q.dequantize(x_q)
    delta_star = rng.uniform(-0.5, 0.5, n)
    s = snap_to_lattice(tele_q, delta_star, q, x_q_levels=x_q)
    bits_clean = q.levels_to_bits(x_q)
    recon = q.dequantize(q.bits_to_levels(bits_clean ^ s["delta_info_bits"]))
    err = float(np.max(np.abs(recon - s["tele_recv"])))
    assert err == 0.0, f"XOR identity violated: max err {err:.3e}"
    print(f"[ok] XOR identity: Q⁻¹(x_q ⊕ δ_info) == tele_recv exactly (err {err:.1e})")

def test_snap_rounding_bound():
    q = Quantizer()
    rng = np.random.default_rng(3)
    n = 500
    tele = rng.uniform(-0.5, 0.5, n)
    x_q = q.quantize(tele)
    tele_q = q.dequantize(x_q)
    delta_star = rng.uniform(-0.3, 0.3, n)
    assert np.all(np.abs(tele_q + delta_star) < 1.0)
    s = snap_to_lattice(tele_q, delta_star, q, x_q_levels=x_q)
    assert s["requant_loss_lsb"] <= 0.5 + 1e-6, s["requant_loss_lsb"]
    print(f"[ok] in-range snap rounding ≤ Δ_LSB/2: {s['requant_loss_lsb']:.4f} LSB; "
          f"realized L∞ {s['realized_linf_lsb']:.2f} LSB, wt(δ_info) {s['weight']}")

def test_snap_saturation_is_clamped():
    q = Quantizer()
    n = 100
    tele = np.full(n, 0.95)
    x_q = q.quantize(tele)
    tele_q = q.dequantize(x_q)
    delta_star = np.full(n, 0.4)
    s = snap_to_lattice(tele_q, delta_star, q, x_q_levels=x_q)
    top_center = q.dequantize(np.array([q.n_levels - 1]))[0]
    assert np.allclose(s["tele_recv"], top_center), (s["tele_recv"][0], top_center)
    assert s["requant_loss_lsb"] > 0.5
    print(f"[ok] saturation clamps to top bin center {top_center:.4f}; "
          f"requant_loss {s['requant_loss_lsb']:.1f} LSB reflects the clamp (expected)")

def test_snap_zero_delta():
    q = Quantizer()
    n = 50
    tele = np.linspace(-0.5, 0.5, n)
    x_q = q.quantize(tele)
    tele_q = q.dequantize(x_q)
    s = snap_to_lattice(tele_q, np.zeros(n), q, x_q_levels=x_q)
    assert s["weight"] == 0 and np.allclose(s["delta_snap"], 0.0)
    print("[ok] zero delta → zero δ_info, zero perturbation")

if __name__ == "__main__":
    test_lift_exact_for_linear_loss()
    test_lift_lsb_accurate_for_nonlinear_loss()
    test_snap_xor_identity()
    test_snap_rounding_bound()
    test_snap_saturation_is_clamped()
    test_snap_zero_delta()
    print("\nall lift/snap tests passed.")