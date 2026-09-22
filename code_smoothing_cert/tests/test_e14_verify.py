from __future__ import annotations
import sys, json, tempfile
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e14_verify as V

def _full_rec(chan, labels):
    recs = []
    for l in labels:
        label, rc, in_scope, missed_clean, monotone, fp = l
        if in_scope and rc is not None:
            rec = {"r_cert": rc, "r_cert_simple": rc, "r_cert_ci_aware": max(0, rc - 16),
                   "max_eta": 0.03 if rc >= 50 else 0.6}
        else:
            rec = {"r_cert": None, "r_cert_simple": None, "r_cert_ci_aware": None, "max_eta": None}
        rec.update({"label": list(label), "in_scope": in_scope, "missed_clean": missed_clean,
                    "monotone": monotone, "footprint_bits": fp})
        recs.append(rec)
    return {"chan": chan, "labels": recs}

def _cohen_rec(chan, labels):
    return {"chan": chan, "labels": [
        {"label": list(l[0]), "best_radius": l[1], "d_coords": l[2], "in_scope": l[3]} for l in labels]}

def _write(d, recs):
    p = Path(tempfile.mkdtemp()) / d; p.mkdir(parents=True)
    for r in recs:
        json.dump(r, open(p / f"{r['chan']}.json", "w"))
    return str(p)

def test_gate_frozen_repro_exact():
    full = {"A-2": _full_rec("A-2", [((4450,4560), 512, True, False, True, 4488)]),
            "D-1": _full_rec("D-1", [((5250,8508), 16, True, False, False, 26064)])}
    frozen_dir = _write("runs_e14", [
        {"chan":"A-2","labels":[{"label":[4450,4560],"r_cert":512}]},
        {"chan":"D-1","labels":[{"label":[5250,8508],"r_cert":16}]}])
    g = V.gate_frozen_repro(full, frozen_dir)
    assert g["exact_reproduction"] is True and g["n_checked"] == 2 and g["n_mismatch"] == 0

def test_gate_frozen_repro_detects_mismatch():
    full = {"A-2": _full_rec("A-2", [((4450,4560), 256, True, False, True, 4488)])}
    frozen_dir = _write("runs_e14", [{"chan":"A-2","labels":[{"label":[4450,4560],"r_cert":512}]}])
    g = V.gate_frozen_repro(full, frozen_dir)
    assert g["exact_reproduction"] is False and g["n_mismatch"] == 1

def test_certified_accuracy_and_coverage():
    full = {c: _full_rec(c, [((0,10), rc, True, False, True, 4000)])
            for c, rc in [("A", 512), ("B", 256), ("C", 128), ("D", 64), ("E", 16)]}
    strict, simple, ci, points, nonmono, oos = V.collect_rcerts(full)
    assert sorted(strict) == [16, 64, 128, 256, 512]
    import certificate as CERT
    grid = [1,16,32,64,128,256,512]
    r95 = CERT.radius_at_coverage(strict, grid, 0.95)
    assert CERT.radius_at_coverage(strict, grid, 1.0) == 16
    assert r95 == 16

def test_out_of_scope_gating():
    full = {"F": _full_rec("F", [((0,10), None, False, True, True, 4000)])}
    strict, simple, ci, points, nonmono, oos = V.collect_rcerts(full)
    assert len(strict) == 0 and oos == 1

def test_cross_method_agreement():
    points = [{"chan":"A-2","label":[4450,4560],"r_cert":512,"footprint_bits":4488,"r_cert_ci_aware":496,"max_eta":0.03},
              {"chan":"D-1","label":[5250,8508],"r_cert":16,"footprint_bits":26064,"r_cert_ci_aware":0,"max_eta":0.6}]
    full = {}
    cohen = {"A-2": _cohen_rec("A-2", [((4450,4560), 0.5, 100, True)]),
             "D-1": _cohen_rec("D-1", [((5250,8508), 0.0, 100, True)])}
    res = V.cohen_table_and_agreement(full, cohen, points)
    assert res["both_robust"] == 1 
    assert res["both_fragile"] == 1
    assert res["agreement_rate"] == 1.0
    for p in res["table"]["per_point"]:
        assert "incomparable_note" in p and not any("winner" in k.lower() for k in p)

def test_full_main_smoke():
    full = _write("runs_e14_full", [
        _full_rec("A-2", [((4450,4560), 512, True, False, True, 4488)]),
        _full_rec("D-1", [((5250,8508), 16, True, False, False, 26064)])])
    frozen = _write("runs_e14", [
        {"chan":"A-2","labels":[{"label":[4450,4560],"r_cert":512}]},
        {"chan":"D-1","labels":[{"label":[5250,8508],"r_cert":16}]}])
    cohen = _write("runs_cohen", [
        _cohen_rec("A-2", [((4450,4560), 0.5, 100, True)]),
        _cohen_rec("D-1", [((5250,8508), 0.0, 100, True)])])
    out = Path(tempfile.mkdtemp()) / "v.json"
    sys.argv = ["e14_verify", "--full", full, "--frozen", frozen, "--cohen", cohen, "--out", str(out)]
    V.main()
    res = json.load(open(out))
    assert res["gate_frozen_repro"]["exact_reproduction"] is True
    assert res["n_in_scope_points"] == 2
    assert res["cross_method_agreement"]["agreement_rate"] == 1.0

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")