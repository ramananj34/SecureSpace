from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import nullspace_attack_utils.gf2 as gf2
from theorem_1.e8_helpers import windows_of, aggregate_windows, collateral_msb_fraction
from theorem_1.test_combined_defense import _build, SLOT, NSLOTS, HEADER

def test_windows_of_skips_empty_and_strides():
    di = np.zeros((10, 2), dtype=np.uint8)
    di[3, 0] = 1
    di[6, 1] = 1
    got = list(windows_of(di, window=2, b=2))
    starts = [w0 for w0, _ in got]
    assert starts == [2, 6]
    for w0, wbits in got:
        assert wbits.shape == (4,) and int(wbits.sum()) == 1

def test_aggregate_windows_reduction():
    pw = [{"targetable": True, "combined_passes_both": True, "footprint_exact": True, "naive_flagged_by_combined": True, "window_weight": 10, "naive_Hprime_synd_weight": 1300, "info_weight_total": 200, "collateral_info_weight": 190, "n_slots_touched": 39}, {"targetable": True, "combined_passes_both": True, "footprint_exact": True, "naive_flagged_by_combined": False, "window_weight": 20, "naive_Hprime_synd_weight": 0, "info_weight_total": 260, "collateral_info_weight": 240, "n_slots_touched": 40}]
    ag = aggregate_windows(pw)
    assert ag["n_windows_touched"] == 2
    assert ag["frac_combined_passes_both"] == 1.0
    assert ag["frac_footprint_exact"] == 1.0
    assert abs(ag["frac_naive_flagged_by_combined"] - 0.5) < 1e-9
    assert ag["collateral_info_weight_med"] == 215
    assert ag["n_slots_touched_max"] == 40
    assert aggregate_windows([]) == {"n_windows_touched": 0}

def test_collateral_msb_fraction():
    v = np.zeros(16, dtype=np.uint8)
    v[7] = 1
    v[8] = 1
    v[15] = 1
    frac, total = collateral_msb_fraction(v, header_bits=0, slot_bits=8, n_slots=2, target_slot=0, b=8)
    assert total == 2 and abs(frac - 0.5) < 1e-9
    v0 = np.zeros(16, dtype=np.uint8); v0[7] = 1
    assert collateral_msb_fraction(v0, 0, 8, 2, 0, 8) == (0.0, 0)

def test_end_to_end_synthetic_combined():
    code, secret, cd = _build()
    rng = np.random.default_rng(5)
    n_win = 6
    di = rng.integers(0, 2, size=(n_win * SLOT, 1), dtype=np.uint8)
    per_window = []
    for w0, wbits in windows_of(di, window=SLOT, b=1):
        a = cd.arms(wbits, slot=0)
        a["window_start"] = int(w0)
        per_window.append(a)
    assert per_window, "expected some non-empty windows"
    ag = aggregate_windows(per_window)
    assert ag["frac_targetable"] == 1.0
    assert ag["frac_combined_passes_both"] == 1.0
    assert ag["frac_footprint_exact"] == 1.0
    assert ag["frac_naive_flagged_by_combined"] >= 0.5

_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

def main():
    print("e8_helpers + synthetic E8 tests\n" + "-" * 48)
    fails = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            fails += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("-" * 48)
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())