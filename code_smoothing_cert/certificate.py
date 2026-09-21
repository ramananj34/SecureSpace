from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from run_e14 import r_cert as _r_cert_frozen
    _HAVE_FROZEN_RCERT = True
except Exception:
    _HAVE_FROZEN_RCERT = False
    def _r_cert_frozen(curve):
        rc = 0
        for c in sorted(curve, key=lambda x: x["w"]):
            if c["eta_hat"] < 0.5:
                rc = c["w"]
            else:
                break
        simple = max([c["w"] for c in curve if c["eta_hat"] < 0.5], default=0)
        return rc, simple

def _normalize_curve(curve):
    out = []
    for c in curve:
        eta = c.get("eta_hat", c.get("eta"))
        if eta is None:
            raise ValueError(f"curve entry missing eta/eta_hat: {c}")
        out.append({"w": int(c["w"]), "eta_hat": float(eta),
                    "ci_hi": (float(c["ci_hi"]) if c.get("ci_hi") is not None else None)})
    return out

def r_cert(curve):
    return _r_cert_frozen(_normalize_curve(curve))

def r_cert_ci_aware(curve, half=0.5):
    nc = _normalize_curve(curve)
    if any(c["ci_hi"] is None for c in nc):
        return None
    rc = 0
    for c in sorted(nc, key=lambda x: x["w"]):
        if c["ci_hi"] < half:
            rc = c["w"]
        else:
            break
    return rc

def certified_accuracy_curve(rcerts, grid):
    rcerts = np.asarray(rcerts, float)
    return [{"radius": int(w), "certified_accuracy": float(np.mean(rcerts >= w))} for w in sorted(grid)]

def radius_at_coverage(rcerts, grid, coverage=0.95):
    rcerts = np.asarray(rcerts, float)
    best = 0
    for w in sorted(grid):
        if float(np.mean(rcerts >= w)) >= coverage:
            best = int(w)
    return best

def summarize_rcerts(rcerts_strict, rcerts_simple=None):
    a = np.asarray(rcerts_strict, float)
    out = {"n_points": int(a.size),
           "r_cert_median": float(np.median(a)) if a.size else 0.0,
           "r_cert_mean": float(np.mean(a)) if a.size else 0.0,
           "r_cert_min": float(a.min()) if a.size else 0.0,
           "r_cert_max": float(a.max()) if a.size else 0.0,
           "frac_rcert_ge_50": float(np.mean(a >= 50)) if a.size else 0.0}
    if rcerts_simple is not None:
        s = np.asarray(rcerts_simple, float)
        out["n_strict_lt_simple"] = int(np.sum(a < s))
        out["frac_nonmonotone"] = float(np.mean(a < s)) if a.size else 0.0
    return out

def curve_is_monotone(curve, tol=1e-9):
    nc = sorted(_normalize_curve(curve), key=lambda c: c["w"])
    e = [c["eta_hat"] for c in nc]
    return all(e[i] <= e[i + 1] + tol for i in range(len(e) - 1))