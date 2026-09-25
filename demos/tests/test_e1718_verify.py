from __future__ import annotations
import sys, json, tempfile
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e17e18_verify as V

def _write(d, records):
    p = Path(tempfile.mkdtemp())
    for name, rec in records.items():
        json.dump(rec, open(p / f"{name}.json", "w"))
    return str(p)

REAL = {
    "e18_bercurves": {"baseline": {"rate": 0.4444, "rows": [{"ebn0_db": e, "bler": 0.1} for e in [1,2,3]]},
                      "amrcc": {"rate": 0.4444, "rows": [{"ebn0_db": e, "bler": 0.1} for e in [1,2,3]]},
                      "compare": {"all_overlap": True, "rows": [{"ebn0_db": e, "baseline_bler": 0.1,
                                  "amrcc_bler": 0.1, "ci_overlap": True} for e in [1,2,3]]}},
    "e18_epsdec": {"rows": [{"ebn0_db": 3.0, "eps_dec": 0.0, "ci_lo": 0.0, "ci_hi": 0.0092},
                            {"ebn0_db": 3.5, "eps_dec": 0.0, "ci_lo": 0.0, "ci_hi": 0.0092}]},
    "e17_uplink": {"targeted_damage": {"clean": 0.0, "undefended": 0.01976, "amrcc": 0.001339},
                   "l2": {"clean": 0.03, "undefended": 6.198, "amrcc": 2.691}, "w": 300, "d": 5000},
    "endtoend_D-16": {"chan": "D-16", "ebn0_db": 3.5, "n_frames": 3, "frame_failures": 0,
                      "bit_error_rate": 0.0, "f1_true": 1.0, "f1_recovered": 1.0, "detection_preserved": True},
}

def test_link_budget_0db():
    v = V.verify(V.load(_write("r", REAL)))
    assert v["link_budget_0db"]["all_ci_overlap"] is True

def test_eps_dec_validated():
    v = V.verify(V.load(_write("r", REAL)))
    assert v["eps_dec"]["validates_eps_dec_zero"] is True
    assert v["eps_dec"]["max_eps_dec"] < 0.01

def test_e17_targeting_defense():
    v = V.verify(V.load(_write("r", REAL)))
    u = v["e17_uplink"]
    assert u["defense_holds"] is True
    assert u["targeting_dilution"] > 10
    assert u["l2_dilution"] > 1

def test_endtoend_detection_preserved():
    v = V.verify(V.load(_write("r", REAL)))
    assert v["endtoend"]["detection_preserved"] is True
    assert v["endtoend"]["ber"] == 0.0

def test_main_smoke():
    out = Path(tempfile.mkdtemp()) / "v.json"
    sys.argv = ["e17e18_verify", "--runs", _write("r", REAL), "--out", str(out)]
    V.main()
    res = json.load(open(out))
    assert res["link_budget_0db"]["all_ci_overlap"] is True
    assert res["endtoend"]["detection_preserved"] is True

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")