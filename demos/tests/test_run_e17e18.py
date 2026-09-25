from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "ldpc"), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "amrcc"), str(_ROOT / "fl_core")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import run_e17e18 as R
import bp_decoder as BP

def test_e18_bercurves_wiring():
    code = BP.load_short_code()
    out = Path(tempfile.mkdtemp())
    rec = R.run_e18_bercurves(code, [2.0], n_trials=10, seed=0, out_dir=out, force=True)
    assert "compare" in rec and "all_overlap" in rec["compare"]
    assert (out / "e18_bercurves.json").exists()

def test_e17_uplink_wiring():
    out = Path(tempfile.mkdtemp())
    rec = R.run_e17_uplink(seed=0, out_dir=out, force=True)
    td = rec["targeted_damage"]
    assert td["undefended"] > td["amrcc"]
    assert (out / "e17_uplink.json").exists()

def test_epsdec_wiring():
    code = BP.load_short_code()
    out = Path(tempfile.mkdtemp())
    rec = R.run_e18_epsdec(code, [3.5], n_trials=20, seed=0, out_dir=out, force=True)
    assert rec["rows"][0]["eps_dec"] >= 0.0

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")