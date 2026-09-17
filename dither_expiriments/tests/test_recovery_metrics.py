from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "smap_msl_data"), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from smap_msl_dataset_api import Quantizer
from recovery_metrics import (m1_random_baseline, m1_bit_accuracy, score_candidate, observe, m4_msb_localization, spearman)

Q = Quantizer()

def test_baseline_half_at_balanced():
    assert abs(m1_random_baseline(10000, 5000) - 0.5) < 1e-3

def test_identity_perfect():
    rng = np.random.default_rng(0)
    k = 464; m_in = (rng.random(k) < 0.5).astype(np.uint8)
    perm = rng.permutation(k)
    rec = score_candidate(m_in, perm, perm, b=8, header_bits=400, slot_bits=32)
    assert rec["m1_bit_acc"] == 1.0
    assert rec["m2_pos_acc"] == 1.0

def test_random_candidate_near_baseline():
    rng = np.random.default_rng(1)
    k = 8000; m_in = (rng.random(k) < 0.5).astype(np.uint8)
    perm = rng.permutation(k)
    accs = []
    for s in range(30):
        pj = np.random.default_rng([2, s]).permutation(k)
        accs.append(m1_bit_accuracy(m_in, observe(m_in, perm), pj))
    m = np.mean(accs)
    assert abs(m - m1_random_baseline(k, int(m_in.sum()))) < 0.01

def test_m2_baseline_near_zero_random():
    rng = np.random.default_rng(3)
    k = 5000; m_in = (rng.random(k) < 0.5).astype(np.uint8)
    perm = rng.permutation(k)
    pj = np.random.default_rng(4).permutation(k)
    rec = score_candidate(m_in, perm, pj)
    assert rec["m2_pos_acc"] < 0.005

def test_degeneracy_flagged():
    rng = np.random.default_rng(5)
    k = 2000; m_in = np.zeros(k, np.uint8); m_in[:3] = 1
    perm = rng.permutation(k)
    pj = np.random.default_rng(6).permutation(k)
    rec = score_candidate(m_in, perm, pj)
    assert rec["near_degenerate"] is True
    assert rec["m1_baseline"] > 0.99

def test_m4_truth_is_one():
    rng = np.random.default_rng(7)
    b = 8; header = 16; slot = 24; k = header + slot
    m_in = (rng.random(k) < 0.5).astype(np.uint8)
    perm = rng.permutation(k)
    assert m4_msb_localization(perm, perm, k, b, header, slot) == 1.0

def test_m4_random_near_baseline():
    rng = np.random.default_rng(70)
    b = 8; header = 0; n_coords = 50; slot = n_coords * b; k = slot
    m_in = (rng.random(k) < 0.5).astype(np.uint8)
    perm = rng.permutation(k)
    vals = [m4_msb_localization(perm, np.random.default_rng([71, s]).permutation(k), k, b, header, slot)
            for s in range(40)]
    assert abs(np.mean(vals) - 1.0 / n_coords) < 0.01

def test_m3_and_telem_order_wiring():
    rng = np.random.default_rng(8)
    b = 8; header = 16; slot = 24; k = header + slot
    x = np.clip(np.cumsum(rng.normal(0, 0.05, slot // b)), -0.9, 0.9)
    bits = np.asarray(Q.levels_to_bits(Q.quantize(x)), np.uint8).reshape(-1)
    m_in = np.concatenate([rng.integers(0, 2, header).astype(np.uint8), bits])
    perm = rng.permutation(k)
    rec_true = score_candidate(m_in, perm, perm, quantizer=Q, b=b, header_bits=header, slot_bits=slot, include_m3=True, m3_random=16)
    assert rec_true["m4_msb_loc"] == 1.0
    assert rec_true["m3"]["ratio"] < 1e-9
    assert rec_true["telem_order"]["abs_spearman"] > 0.999
    pj = np.random.default_rng(9).permutation(k)
    rec_rand = score_candidate(m_in, perm, pj, quantizer=Q, b=b, header_bits=header, slot_bits=slot, include_m3=True, m3_random=64)
    assert rec_rand["m3"]["ratio"] > 0.5

def test_spearman_reflection_symmetric():
    x = np.linspace(0, 1, 200)
    assert spearman(x, x) > 0.999
    assert spearman(x, x[::-1]) < -0.999

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")