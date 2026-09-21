from __future__ import annotations
import sys, json, tempfile
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e13_verify as V

def _gen_rec(name, per_w, recover):
    return {"generator": name, "k": 800, "weights": [w["w"] for w in per_w],
            "recover_score": recover, "per_w": per_w}

def _pw(w, eta, adv_a, eps_v, eps_max, floor, route, holds=None):
    rhs = eta + eps_max
    return {"w": w, "eta_w": eta, "adv_a_worst_target": adv_a, "adv_b_oracle": 1.0,
            "eps_v": eps_v, "eps_v_max": eps_max, "eps_v_mc_floor": floor,
            "eps_v_above_floor": max(0.0, eps_v - floor), "theorem2_rhs": rhs,
            "theorem2_holds": (adv_a <= rhs + 1e-9) if holds is None else holds,
            "theorem2_slack": rhs - adv_a, "route_label": route}

def _write(recs):
    d = Path(tempfile.mkdtemp())
    for r in recs:
        json.dump(r, open(d / f"{r['generator']}.json", "w"))
    return str(d)

def test_gridwide_hold_and_anchors():
    strong = _gen_rec("strong", [_pw(32, 0.04, 0.045, 0.044, 0.021, 0.044, "neither")], 0.0)
    ident = _gen_rec("identity", [_pw(32, 0.04, 1.0, 0.999, 0.96, 0.044, "both")], 1.0)
    local = _gen_rec("local_w2", [_pw(32, 0.04, 0.51, 0.997, 0.48, 0.044, "nonuniformity")], 0.0)
    d = _write([strong, ident, local])
    recs = V.load(d); rows = V.verify(recs); s = V.summarize(rows, recs)
    assert s["thm2_holds_gridwide"] is True
    assert s["strong_robust"] is True
    assert s["identity_oracle_collapse"] is True

def test_route_classification():
    strong = _gen_rec("strong", [_pw(32, 0.04, 0.045, 0.044, 0.021, 0.044, "neither")], 0.0)
    biased = _gen_rec("biased_p0.90", [_pw(32, 0.04, 0.05, 0.33, 0.03, 0.044, "neither")], 0.0)
    local = _gen_rec("local_w8", [_pw(32, 0.04, 0.14, 0.99, 0.11, 0.044, "nonuniformity")], 0.0)
    trunc = _gen_rec("trunc_2^2", [_pw(32, 0.04, 0.27, 0.995, 0.25, 0.044, "both")], 0.9)
    d = _write([strong, biased, local, trunc])
    recs = V.load(d); s = V.summarize(V.verify(recs), recs)
    assert "strong" in s["keystream_robust_generators"]
    assert "biased_p0.90" in s["keystream_robust_generators"]
    assert "local_w8" in s["permutation_weak_generators"]
    assert "trunc_2^2" in s["permutation_weak_generators"]

def test_floor_subtraction_puts_uniform_near_zero():
    strong = _gen_rec("strong", [_pw(1, 0.00125, 0.005, 0.25, 0.02, 0.25, "neither")], 0.0)
    d = _write([strong]); recs = V.load(d); s = V.summarize(V.verify(recs), recs)
    r = s["per_gen_worst"]["strong"]
    assert r["eps_v_above_floor"] < 0.05
    assert r["eps_v"] > 0.2

def test_violation_flagged():
    bad = _gen_rec("broken", [_pw(32, 0.04, 0.9, 0.5, 0.3, 0.044, "nonuniformity", holds=False)], 0.0)
    d = _write([bad]); recs = V.load(d); s = V.summarize(V.verify(recs), recs)
    assert s["thm2_holds_gridwide"] is False

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")