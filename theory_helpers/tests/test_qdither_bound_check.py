from __future__ import annotations
import sys, json, tempfile
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import qdither_bound_check as QB

def _fake_consolidated(sigmas=(0.0, 0.05, 0.25, 1.0), adj=(1.32, 1.30, 1.25, 1.12), sig=(4.42, 4.18, 3.61, 2.72), att=(1.007, 1.024, 1.011, 1.000), ceil=(0.88, 0.88, 0.87, 0.80)):
    pooled = [{"sigma_lsb": s, "attacker_m1_ratio_mean": att[i],
               "adjacency_mi_total_mean": adj[i], "signal_mi_total_mean": sig[i]}
              for i, s in enumerate(sigmas)]
    channels = {"C": {"adv_t5_curve": [{"sigma_lsb": s, "adv_t5_proxy": 0.0} for s in sigmas]}}
    return {"pooled": pooled, "channels": channels}

def _write(tmp, obj):
    p = Path(tmp) / "_qdither_consolidated.json"
    json.dump(obj, open(p, "w")); return str(p)

def test_bound_holds_all_sigma():
    p = _write(tempfile.mkdtemp(), _fake_consolidated())
    rows = QB.check(p, adv_t4=0.0)
    assert all(r["holds"] for r in rows)
    assert len(rows) == 4

def test_bound_monotone_decreasing():
    p = _write(tempfile.mkdtemp(), _fake_consolidated())
    rows = QB.check(p, adv_t4=0.0)
    b = [r["bound_adj"] for r in rows]
    assert all(b[i] >= b[i + 1] - 1e-9 for i in range(len(b) - 1))

def test_wide_margin_at_sigma0():
    p = _write(tempfile.mkdtemp(), _fake_consolidated())
    rows = QB.check(p, adv_t4=0.0)
    r0 = rows[0]
    assert r0["adv_t5_max"] < 0.05
    assert r0["bound_min_proxy"] > 0.7
    assert r0["margin"] > 0.6

def test_proxy_is_lower_bound_flagged_by_min_selection():
    p = _write(tempfile.mkdtemp(), _fake_consolidated())
    rows = QB.check(p, adv_t4=0.0)
    for r in rows:
        assert abs(r["bound_min_proxy"] - min(r["bound_adj"], r["bound_sig"])) < 1e-9

def test_adv_t4_substitution_shifts_bound():
    p = _write(tempfile.mkdtemp(), _fake_consolidated())
    r0 = QB.check(p, adv_t4=0.0)
    r1 = QB.check(p, adv_t4=0.1)
    for a, b in zip(r0, r1):
        assert abs((b["bound_adj"] - a["bound_adj"]) - 0.1) < 1e-9

def test_violation_detected_if_adv_exceeds_bound():
    bad = _fake_consolidated(att=(3.0, 3.0, 3.0, 3.0))
    p = _write(tempfile.mkdtemp(), bad)
    rows = QB.check(p, adv_t4=0.0)
    assert any(not r["holds"] for r in rows)

def test_summary_fields():
    p = _write(tempfile.mkdtemp(), _fake_consolidated())
    rows = QB.check(p, adv_t4=0.0)
    s = QB.summarize(rows, 0.0)
    assert s["n_hold"] == s["n"] == 4
    assert s["bound_monotone_decreasing_in_sigma"] is True
    assert s["margin_at_sigma0"] > 0.6 and s["min_margin"] > 0.4

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")