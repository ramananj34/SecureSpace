from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import certificate as C

A2_STORED = {"label": [4450, 4560], "r_cert": 512, "r_cert_simple": 512,
             "curve": [{"w":1,"eta":0.0,"ci_hi":0.0122},{"w":4,"eta":0.0,"ci_hi":0.0122},
                       {"w":16,"eta":0.0,"ci_hi":0.0122},{"w":32,"eta":0.0,"ci_hi":0.0122},
                       {"w":64,"eta":0.0,"ci_hi":0.0122},{"w":128,"eta":0.01667,"ci_hi":0.0385},
                       {"w":256,"eta":0.03667,"ci_hi":0.0647},{"w":512,"eta":0.01,"ci_hi":0.0289}]}

def test_reproduces_frozen_a2_rcert():
    rc, simple = C.r_cert(A2_STORED["curve"])
    assert rc == A2_STORED["r_cert"] == 512
    assert simple == A2_STORED["r_cert_simple"] == 512

def test_curve_key_normalization():
    live = [{"w": w, "eta_hat": c["eta"], "ci_hi": c["ci_hi"]} for w, c in
            zip([1,4,16,32,64,128,256,512], A2_STORED["curve"])]
    assert C.r_cert(live) == C.r_cert(A2_STORED["curve"])

def test_strict_less_than_simple_nonmonotone():
    curve = [{"w":1,"eta_hat":0.0},{"w":4,"eta_hat":0.0},{"w":16,"eta_hat":0.6},
             {"w":32,"eta_hat":0.3},{"w":64,"eta_hat":0.2}]
    rc, simple = C.r_cert(curve)
    assert rc == 4 and simple == 64
    assert not C.curve_is_monotone(curve)

def test_strict_equals_simple_monotone():
    curve = [{"w":1,"eta_hat":0.0},{"w":4,"eta_hat":0.1},{"w":16,"eta_hat":0.3},
             {"w":32,"eta_hat":0.6},{"w":64,"eta_hat":0.8}]
    rc, simple = C.r_cert(curve)
    assert rc == simple == 16
    assert C.curve_is_monotone(curve)

def test_r_cert_ci_aware_conservative():
    curve = [{"w":1,"eta_hat":0.1,"ci_hi":0.3},{"w":4,"eta_hat":0.2,"ci_hi":0.45},
             {"w":16,"eta_hat":0.3,"ci_hi":0.55},{"w":32,"eta_hat":0.4,"ci_hi":0.7}]
    rc_point, _ = C.r_cert(curve)
    rc_ci = C.r_cert_ci_aware(curve)
    assert rc_point == 32 and rc_ci == 4
    assert rc_ci <= rc_point
    assert C.r_cert_ci_aware([{"w":1,"eta_hat":0.1}]) is None

def test_certified_accuracy_non_increasing():
    rcerts = [512, 256, 64, 16, 4, 0]
    grid = [1, 4, 16, 32, 64, 128, 256, 512]
    ca = C.certified_accuracy_curve(rcerts, grid)
    vals = [p["certified_accuracy"] for p in ca]
    assert all(vals[i] >= vals[i+1] - 1e-9 for i in range(len(vals)-1))
    assert ca[0]["certified_accuracy"] == 5/6
    assert ca[-1]["certified_accuracy"] == 1/6

def test_radius_at_coverage():
    rcerts = [128]*96 + [16]*4
    grid = [1, 16, 64, 128, 256]
    assert C.radius_at_coverage(rcerts, grid, 0.95) == 128
    assert C.radius_at_coverage(rcerts, grid, 0.97) == 16

def test_summarize_flags_nonmonotone():
    strict = [4, 16, 512]; simple = [64, 16, 512]
    s = C.summarize_rcerts(strict, simple)
    assert s["n_strict_lt_simple"] == 1
    assert abs(s["frac_rcert_ge_50"] - 1/3) < 1e-9

def test_50bit_95pct_check_shape():
    rcerts = [64]*95 + [8]*5
    grid = [1, 8, 32, 64, 128, 256, 512]
    r95 = C.radius_at_coverage(rcerts, grid, 0.95)
    assert r95 == 64 and r95 >= 50

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")