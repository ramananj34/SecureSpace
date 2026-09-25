from __future__ import annotations
import sys, json, tempfile
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "fl_core")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e16_verify as V

def _rec(regime, bm, mse, f1):
    return {"regime": regime, "bm_frac": bm, "final_mse_ratio": mse, "final_f1": f1, "collapsed": mse > 100}

def _write(recs):
    d = Path(tempfile.mkdtemp()) / "runs_e16"; d.mkdir(parents=True)
    for r in recs:
        json.dump(r, open(d / f"{r['regime']}_bm{int(r['bm_frac']*100)}.json", "w"))
    return str(d)

D16 = [
    _rec("fedavg", 0.0, 0.957, 1.0), _rec("fedavg", 0.10, 4.38e3, 0.0),
    _rec("fedavg", 0.30, 7.22e4, 0.0), _rec("fedavg", 0.50, 3.64e5, 0.0),
    _rec("median", 0.0, 0.929, 1.0), _rec("median", 0.10, 0.935, 1.0),
    _rec("median", 0.30, 0.968, 1.0), _rec("median", 0.50, 5.73e5, 0.0),
    _rec("fedavg_perm", 0.0, 0.957, 1.0), _rec("fedavg_perm", 0.10, 0.95, 1.0),
    _rec("fedavg_perm", 0.30, 0.967, 1.0), _rec("fedavg_perm", 0.50, 0.927, 1.0),
    _rec("median_perm", 0.0, 0.929, 1.0), _rec("median_perm", 0.10, 0.935, 1.0),
    _rec("median_perm", 0.30, 0.964, 1.0), _rec("median_perm", 0.50, 1.0, 1.0),
]

def test_sec85_regime3b_within_5pct():
    recs = V.load(_write(D16))
    s = V.sec85_check(recs)
    mp = s["per_regime"]["median_perm"]
    assert mp["within_tol"] is True
    assert s["regime3b_within_5pct_at_30"] is True

def test_sec85_regime1_collapses():
    recs = V.load(_write(D16))
    s = V.sec85_check(recs)
    assert s["regime1_collapse_bm"] == 0.10

def test_3b_vs_2_rescue_at_50():
    recs = V.load(_write(D16))
    c = V.compare_3b_vs_2(recs)
    at50 = [r for r in c["rows"] if r["bm"] == 0.50][0]
    assert at50["r2_robust"] is False and at50["r3b_robust"] is True
    assert at50["keyed_perm_rescues_median"] is True
    assert c["keyed_perm_rescues_median_at_some_bm"] is True
    at30 = [r for r in c["rows"] if r["bm"] == 0.30][0]
    assert at30["r2_robust"] and at30["r3b_robust"] and not at30["keyed_perm_rescues_median"]

def test_regime_table_shape():
    recs = V.load(_write(D16))
    tbl, regimes, bms = V.regime_table(recs)
    assert set(regimes) == {"fedavg", "median", "fedavg_perm", "median_perm"}
    assert 0.5 in bms
    assert tbl["median"][0.50]["mse_ratio"] > 100
    assert tbl["median_perm"][0.50]["mse_ratio"] < 2

def test_theorem5prime_block_runs():
    b = V.theorem5prime_block()
    assert b["derivation"]["median_robust_bounded_by_honest_spread"] is True
    assert b["derivation"]["trap_d_aligned_defeats_dadv_reduction"] is True

def test_main_smoke():
    runs = _write(D16)
    out = Path(tempfile.mkdtemp()) / "v.json"
    sys.argv = ["e16_verify", "--runs", runs, "--out", str(out)]
    V.main()
    res = json.load(open(out))
    assert res["sec85"]["regime3b_within_5pct_at_30"] is True
    assert res["compare_3b_vs_2"]["keyed_perm_rescues_median_at_some_bm"] is True

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")