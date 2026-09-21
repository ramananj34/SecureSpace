from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import prng_distinguish as PD
import weak_prng as W

K, WGT = 256, 8

def test_eps_v_uniform_near_floor():
    floor = PD.eps_v_mc_floor(K, WGT, n_perm=2000, reps=5)
    e = PD.eps_v(W.strong, K, WGT, n_perm=2000)
    assert e < floor["mc_floor_mean"] + 5 * max(floor["mc_floor_std"], 0.005), (e, floor)
    assert e < 0.10

def test_eps_v_identity_near_one():
    e = PD.eps_v(W.identity, K, WGT, n_perm=1000)
    assert e > 0.9

def test_eps_v_monotone_local_shuffle():
    e128 = PD.eps_v(W.local_shuffle(128), K, WGT, n_perm=1500)
    e8 = PD.eps_v(W.local_shuffle(8), K, WGT, n_perm=1500)
    e2 = PD.eps_v(W.local_shuffle(2), K, WGT, n_perm=1500)
    assert e2 > e8 > e128
    assert e2 > 0.7

def test_eps_v_monotone_bias():
    e55 = PD.eps_v(W.biased_bit_fy(0.55), K, WGT, n_perm=2000)
    e90 = PD.eps_v(W.biased_bit_fy(0.90), K, WGT, n_perm=2000)
    floor = PD.eps_v_mc_floor(K, WGT, n_perm=2000, reps=5)["mc_floor_mean"]
    assert e90 > e55
    assert e90 > floor

def test_eps_v_reduced_round_chacha_stays_low():
    floor = PD.eps_v_mc_floor(K, WGT, n_perm=2000, reps=5)
    e4 = PD.eps_v(W.reduced_round_chacha(4), K, WGT, n_perm=2000)
    e2 = PD.eps_v(W.reduced_round_chacha(2), K, WGT, n_perm=2000)
    assert e4 < floor["mc_floor_mean"] + 0.10
    assert e2 < floor["mc_floor_mean"] + 0.10

def test_lemma1_holds_for_strong_any_support():
    e_fixed = PD.eps_v(W.strong, K, WGT, n_perm=2000, fixed_support=True)
    e_vary = PD.eps_v(W.strong, K, WGT, n_perm=2000, fixed_support=False)
    assert abs(e_fixed - e_vary) < 0.05 and max(e_fixed, e_vary) < 0.10

def test_perm_nonuniformity_report_flags_correctly():
    r_strong = PD.perm_nonuniformity_report(W.strong, K, M=150)
    r_id = PD.perm_nonuniformity_report(W.identity, K, M=150)
    r_local = PD.perm_nonuniformity_report(W.local_shuffle(2), K, M=150)
    assert r_strong["detectably_nonuniform"] is False
    assert r_id["detectably_nonuniform"] is True
    assert r_local["detectably_nonuniform"] is True

def test_keystream_bias_detects_source_weakness():
    rng = np.random.default_rng(0)
    unbiased = (rng.random(200000) < 0.5).astype(np.uint8)
    biased = (rng.random(200000) < 0.9).astype(np.uint8)
    ku = PD.keystream_bias(unbiased); kb = PD.keystream_bias(biased)
    assert ku["bias_abs"] < 0.01
    assert kb["bias_abs"] > 0.35
    assert abs(ku["lag1_autocorr"]) < 0.02

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")