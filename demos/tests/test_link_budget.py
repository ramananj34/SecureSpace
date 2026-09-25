from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "ldpc"), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import link_budget as LB
import bp_decoder as BP

def test_ebn0_conversion_uses_real_rate():
    code = BP.load_short_code()
    rate = code.k / code.n
    assert abs(rate - 0.4444) < 0.001
    s_lo = LB.ebn0_to_sigma(1.0, rate); s_hi = LB.ebn0_to_sigma(4.0, rate)
    assert s_hi < s_lo
    assert abs(s_lo - np.sqrt(1/(2 * 10**0.1 * rate))) < 1e-9

def test_permuted_message_gives_valid_codeword():
    code = BP.load_short_code()
    rng = np.random.default_rng(0)
    info = (rng.random(code.k) < 0.5).astype(np.uint8)
    msg = LB._permuted_message(info, code, 0, rng)
    c = np.asarray(code.encode(msg)).astype(np.uint8)
    assert code.is_codeword(c)
    assert set(np.unique(msg)).issubset({0, 1}) and msg.sum() == info.sum()

def test_bler_decreases_with_snr():
    code = BP.load_short_code()
    curve = LB.run_curve(code, [1.0, 3.0], n_trials=20, use_perm=False, max_iter=30, seed=1)
    b = [r["bler"] for r in curve["rows"]]
    assert b[1] <= b[0]

def test_baseline_amrcc_coincide():
    code = BP.load_short_code()
    base = LB.run_curve(code, [2.0], n_trials=40, use_perm=False, max_iter=40, seed=2)
    amrcc = LB.run_curve(code, [2.0], n_trials=40, use_perm=True, max_iter=40, seed=2)
    cmp = LB.compare_curves(base, amrcc)
    assert cmp["all_overlap"]

def test_eps_dec_small_at_good_snr():
    code = BP.load_short_code()
    r = LB.eps_dec_at(code, ebn0_db_operating=4.0, n_trials=40, max_iter=50, seed=3)
    assert r["eps_dec"] <= 0.3
    assert 0.0 <= r["ci_lo"] <= r["ci_hi"] <= 1.0

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")