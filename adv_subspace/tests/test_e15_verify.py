from __future__ import annotations
import sys, json, tempfile
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e15_verify as V

def _rec(chan, labels):
    L = []
    for lab, de, di, g, f, wl, cr, rd in labels:
        L.append({"label": list(lab), "d_adv_effective_trace": de, "d_adv_int": di,
                  "d_adv_geometric_dim": g, "d_adv_frobenius_sq": f,
                  "reduction_factor_at_dadv_eff": rd,
                  "theorem4": {str(w): {"within_20pct": ok, "rel_error": 0.05} for w, ok in wl},
                  "cross_term_distribution": {"rel_error": cr, "mean": 1.0, "trace_prediction": 1.0}})
    return {"chan": chan, "labels": L}

def _write(d, recs):
    p = Path(tempfile.mkdtemp()) / d; p.mkdir(parents=True)
    for r in recs: json.dump(r, open(p / f"{r['chan']}.json", "w"))
    return str(p)

def test_scalar_case_dadv_constant_1():
    recs = {f"C{i}": _rec(f"C{i}", [((0,10), 1.0, 1, 1, 1.0, [(1,True),(8,True)], 0.03, 600.0)])
            for i in range(5)}
    C = V.collect(recs)
    dd = np.array(C["dadv"])
    assert dd.min() == dd.max() == 1.0
    d66 = V.d66_coincide(C["dadv"], C["geom"], C["frob"])
    assert d66["three_quantities_coincide"] is True

def test_theorem4_within_fraction():
    recs = {"A": _rec("A", [((0,10), 1.0, 1, 1, 1.0, [(1,True),(8,True),(32,False)], 0.03, 600.0)])}
    C = V.collect(recs)
    assert sum(C["within"]) == 2 and len(C["within"]) == 3

def test_implied_dadv_frame_in_range():
    recs = {f"C{i}": _rec(f"C{i}", [((0,10), 1.0, 1, 1, 1.0, [(1,True)], 0.03, 600.0)]) for i in range(3)}
    C = V.collect(recs)
    frame = 40 * float(np.median(C["dadv"]))
    assert 25 <= frame <= 100

def test_reduction_in_range():
    recs = {"A": _rec("A", [((0,10), 1.0, 1, 1, 1.0, [(1,True)], 0.03, 607.0)])}
    C = V.collect(recs)
    assert 243 <= np.median(C["red"]) <= 972

def test_multi_output_diverging_d66():
    recs = {"A": _rec("A", [((0,10), 2.3, 3, 3, 5.1, [(1,True)], 0.04, 260.0)])}
    C = V.collect(recs)
    d66 = V.d66_coincide(C["dadv"], C["geom"], C["frob"])
    assert d66["three_quantities_coincide"] is False

def test_main_smoke():
    scalar = _write("runs_e15", [_rec("A", [((0,10), 1.0, 1, 1, 1.0, [(1,True),(8,True)], 0.03, 600.0)]),
                                 _rec("B", [((0,10), 1.0, 1, 1, 1.0, [(1,True),(8,True)], 0.02, 605.0)])])
    out = Path(tempfile.mkdtemp()) / "v.json"
    sys.argv = ["e15_verify", "--runs", scalar, "--runs-multi", "/nonexistent", "--out", str(out)]
    V.main()
    res = json.load(open(out))
    assert res["theorem4_within_20pct"]["fraction"] == 1.0
    assert res["d_adv_effective_trace"]["constant"] is True
    assert res["d_adv_frame_in_proposal_range_25_100"] is True

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")