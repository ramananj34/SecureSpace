import sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_ROOT), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "ldpc"),
           str(Path(__file__).resolve().parents[1])]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from frame_ops import (get_long_code, embed_window_in_frame, frame_arms, frame_analysis_over_stream, SLOT_BITS, HEADER_BITS)

def test_nullspace_arm_syndrome_zero():
    code = get_long_code()
    rng = np.random.default_rng(0)
    for w in (1, 2, 8, 50, 200, 400):
        wbits = np.zeros(SLOT_BITS, dtype=np.uint8)
        idx = rng.choice(SLOT_BITS, size=w, replace=False)
        wbits[idx] = 1
        rec = frame_arms(wbits, code=code, slot=0)
        assert rec["syndrome_nullspace"] == 0, (w, rec["syndrome_nullspace"])
    print("[ok] null-space arm: syndrome(encode(δ_info)) == 0 for all tested weights")

def test_naive_arm_flagged():
    code = get_long_code()
    rng = np.random.default_rng(1)
    flagged, total = 0, 0
    weights = []
    for w in (1, 2, 8, 50, 200):
        wbits = np.zeros(SLOT_BITS, dtype=np.uint8)
        wbits[rng.choice(SLOT_BITS, size=w, replace=False)] = 1
        rec = frame_arms(wbits, code=code, slot=0)
        total += 1
        flagged += int(rec["naive_flagged"])
        weights.append(rec["syndrome_naive_weight"])
    assert flagged == total, f"naive arm not always flagged: {flagged}/{total}"
    print(f"[ok] naive arm: flagged {flagged}/{total}; "
          f"syndrome weights {weights} (all > 0 ⇒ caught pre-LSTM)")

def test_parity_domination():
    code = get_long_code()
    rng = np.random.default_rng(2)
    half_parity = (code.n - code.k) / 2.0
    for w in (1, 5, 20, 100):
        wbits = np.zeros(SLOT_BITS, dtype=np.uint8)
        wbits[rng.choice(SLOT_BITS, size=w, replace=False)] = 1
        rec = frame_arms(wbits, code=code, slot=0)
        assert rec["delta_info_weight"] == w
        assert rec["delta_total_weight"] > 10 * w, (w, rec)
        frac = rec["delta_parity_weight"] / half_parity
        assert 0.5 < frac < 1.5, (w, rec["delta_parity_weight"], half_parity)
        print(f"     wt_info={w:4d}  wt_parity={rec['delta_parity_weight']:6d}  "
              f"wt_total={rec['delta_total_weight']:6d}  "
              f"(parity/((n−k)/2)={frac:.2f})")
    print("[ok] parity domination: wt(δ) ≈ wt_info + (n−k)/2 ≈ 16,300, ≫ wt_info")

def test_localization():
    code = get_long_code()
    rng = np.random.default_rng(3)
    wbits = np.zeros(SLOT_BITS, dtype=np.uint8)
    wbits[rng.choice(SLOT_BITS, size=10, replace=False)] = 1
    for slot in (0, 1, 17, 39):
        msg = embed_window_in_frame(wbits, slot=slot, k=code.k)
        lo = HEADER_BITS + slot * SLOT_BITS
        assert msg[lo:lo + SLOT_BITS].sum() == 10
        assert msg.sum() == 10
    print("[ok] localization: δ_info confined to its frame slot")

def test_stream_tiling():
    code = get_long_code()
    rng = np.random.default_rng(4)
    T, b = 650, 8
    bits = np.zeros((T, b), dtype=np.uint8)
    bits[120:140, :3] = (rng.random((20, 3)) < 0.4).astype(np.uint8)  # window @100
    bits[330:360, 2:5] = (rng.random((30, 3)) < 0.4).astype(np.uint8)  # window @300
    res = frame_analysis_over_stream(bits, code=code, window=100, slot=0, b=b)
    assert res["n_frames_touched"] == 2, res["n_frames_touched"]
    assert res["frac_frames_syndrome0"] == 1.0
    assert res["frac_frames_naive_flagged"] == 1.0
    print(f"[ok] stream tiling: {res['n_frames_touched']} frames touched; "
          f"syndrome0={res['frac_frames_syndrome0']:.0%}, "
          f"naive_flagged={res['frac_frames_naive_flagged']:.0%}, "
          f"median wt(δ)={res['delta_total_weight_med']}")

if __name__ == "__main__":
    test_nullspace_arm_syndrome_zero()
    test_naive_arm_flagged()
    test_parity_domination()
    test_localization()
    test_stream_tiling()
    print("\nall frame_ops tests passed.")