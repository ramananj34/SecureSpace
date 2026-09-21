from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import compare_norms as CN

def test_reproduces_proposal_42_vs_023():
    b = CN.heuristic_bridges_from_anchor(CN.COHEN_IMAGENET_ANCHOR, 32400)
    assert 40 < b["bridge_fraction_of_dimension_bits"] < 45
    assert 0.2 < b["bridge_fraction_of_sqrt_dimension_bits"] < 0.26
    assert 170 < b["bridges_differ_by_factor"] < 185

def test_hamming_fraction():
    assert CN.our_hamming_fraction(50, 4488) == 50 / 4488
    assert np.isnan(CN.our_hamming_fraction(50, 0))

def test_cohen_l2_fraction():
    assert abs(CN.cohen_l2_fraction(1.0, 100, 1.0) - 0.1) < 1e-9

def test_l2_to_equiv_msb_flips():
    assert CN.l2_to_equivalent_msb_flips(0.5, 1.0) == 0.5
    assert CN.l2_to_equivalent_msb_flips(2.0, 1.0) == 2.0

def test_compare_point_has_no_winner_field():
    cp = CN.compare_point(r_cert_bits=64, k_region_bits=4488, cohen_R=0.3, d_coords=100)
    assert cp["ours_hamming_r_cert_bits"] == 64
    assert cp["cohen_l2_radius_signal_units"] == 0.3
    assert "ours_hamming_fraction" in cp and "cohen_l2_fraction" in cp
    forbidden = [k for k in cp if any(t in k.lower() for t in ("winner", "better", "beats", "vs_ratio"))]
    assert forbidden == [], f"emitted a comparison-scalar field: {forbidden}"
    assert "incomparable_note" in cp

def test_comparison_table_aggregates_separately():
    rows = [
        {"point_id": "A-2", "r_cert_bits": 512, "k_region_bits": 4488, "cohen_R": 0.4, "d_coords": 100},
        {"point_id": "D-1", "r_cert_bits": 16, "k_region_bits": 26064, "cohen_R": 0.1, "d_coords": 100},
        {"point_id": "M-3", "r_cert_bits": 128, "k_region_bits": 2000, "cohen_R": 0.25, "d_coords": 100},
    ]
    t = CN.comparison_table(rows)
    assert t["aggregate"]["n_points"] == 3
    assert t["aggregate"]["median_ours_r_cert_bits"] == 128
    assert abs(t["aggregate"]["median_cohen_l2_radius"] - 0.25) < 1e-9
    forbidden = [k for k in t["aggregate"] if "ratio" in k.lower() and "cohen" not in k.lower() and "ours" not in k.lower()]
    assert forbidden == []
    assert "note" in t["aggregate"]

def test_bridges_flagged_heuristic():
    b = CN.heuristic_bridges_from_anchor(CN.COHEN_IMAGENET_ANCHOR, 32400)
    assert "note" in b and "HEURISTIC" in b["note"].upper()

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")