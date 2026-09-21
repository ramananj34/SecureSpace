from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc"), str(_ROOT / "smap_msl_data"),
           str(_ROOT / "telemanom_reproduction"), str(_ROOT / "baseline_fgsm_pgd")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from scipy.stats import norm as _norm, beta as _beta
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _phi_inv(p):
    if _HAVE_SCIPY:
        return float(_norm.ppf(p))
    if p <= 0: return -np.inf
    if p >= 1: return np.inf
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    pl=0.02425
    if p<pl:
        q=np.sqrt(-2*np.log(p)); return float((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    if p>1-pl:
        q=np.sqrt(-2*np.log(1-p)); return float(-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    q=p-0.5; r=q*q
    return float((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))


def cp_lower(k, n, alpha):
    if k <= 0:
        return 0.0
    if k >= n:
        return float(alpha ** (1.0 / n))
    if _HAVE_SCIPY:
        return float(_beta.ppf(alpha, k, n - k + 1))
    p = k / n; se = np.sqrt(p * (1 - p) / n)
    return max(0.0, p - 1.6448536269514722 * se)

def smoothed_detect_prob(certify_input, add_noise_and_detect, sigma, n, rng):
    k = 0
    for _ in range(int(n)):
        k += int(add_noise_and_detect(certify_input, sigma, rng))
    return k, int(n)


def certify_point(certify_input, add_noise_and_detect, sigma, n0=100, n=1000, alpha=0.001, rng=None):
    rng = rng or np.random.default_rng(0)
    k0, _ = smoothed_detect_prob(certify_input, add_noise_and_detect, sigma, n0, rng)
    detected_is_top = (k0 > n0 / 2.0)
    if not detected_is_top:
        return {"sigma": float(sigma), "top_is_detected": False, "p_A": None,
                "radius": 0.0, "abstain": True, "reason": "detector not robust to noise (selection)"}
    kA, _ = smoothed_detect_prob(certify_input, add_noise_and_detect, sigma, n, rng)
    pA = cp_lower(kA, n, alpha)
    if pA > 0.5:
        R = float(sigma * _phi_inv(pA))
        return {"sigma": float(sigma), "top_is_detected": True, "p_A": float(pA),
                "kA": int(kA), "n": int(n), "radius": R, "abstain": False}
    return {"sigma": float(sigma), "top_is_detected": True, "p_A": float(pA), "kA": int(kA), "n": int(n),
            "radius": 0.0, "abstain": True, "reason": "p_A <= 1/2"}

def certify_over_sigmas(certify_input, add_noise_and_detect, sigmas, n0=100, n=1000, alpha=0.001, seed=0):
    rows = [certify_point(certify_input, add_noise_and_detect, s, n0=n0, n=n, alpha=alpha,
                          rng=np.random.default_rng([seed, i])) for i, s in enumerate(sigmas)]
    best = max(rows, key=lambda r: r["radius"])
    return {"best_radius": best["radius"], "best_sigma": best["sigma"], "per_sigma": rows}