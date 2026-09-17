from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from smap_msl_dataset_api import Quantizer
from nullspace_attack_utils.frame_packing import FramePlan, pack_frame
from dither import (add_dither, dithered_bits, dithered_levels, dithered_pack_frame, sigma_in_lsb, sigma_from_lsb)

Q = Quantizer()

def test_add_dither_sigma0_exact_copy():
    x = np.linspace(-0.9, 0.9, 500)
    y = add_dither(x, 0.0, np.random.default_rng(0))
    assert np.array_equal(x, y) and y is not x

def test_dithered_bits_sigma0_equals_clean():
    x = np.linspace(-0.9, 0.9, 300)
    b = dithered_bits(x, 0.0, np.random.default_rng(1), Q)
    ref = np.asarray(Q.levels_to_bits(Q.quantize(x)), np.uint8)
    assert np.array_equal(b, ref)

def test_sigma0_reproduces_pack_frame():
    plan = FramePlan(channels=list(range(2)), window=4, b=8)
    rng = np.random.default_rng(2)
    windows = rng.uniform(-0.9, 0.9, size=(2, 4))
    got = dithered_pack_frame(plan, windows, Q, 0.0, np.random.default_rng(9))
    ref = pack_frame(plan, windows, Q)
    assert np.array_equal(got, ref) and got.size == plan.k

def test_level_change_monotone_in_sigma():
    x = np.random.default_rng(3).uniform(-0.9, 0.9, size=4000)
    l0 = Q.quantize(x).astype(np.int64)
    rates = []
    for s in [0.05, 0.1, 0.25, 0.5, 1.0]:
        lv = dithered_levels(x, s * Q.delta_lsb, np.random.default_rng(7), Q).astype(np.int64)
        rates.append(np.mean(lv != l0))
    assert all(rates[i] < rates[i + 1] + 1e-6 for i in range(len(rates) - 1))

def test_sigma_lsb_roundtrip():
    assert abs(sigma_from_lsb(sigma_in_lsb(0.00195, Q), Q) - 0.00195) < 1e-15
    assert abs(sigma_in_lsb(0.25 * Q.delta_lsb, Q) - 0.25) < 1e-12

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")