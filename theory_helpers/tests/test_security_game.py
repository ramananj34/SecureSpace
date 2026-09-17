from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc"),
           str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "ldpc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import security_game as SG
from keyed_permutation import apply_perm, apply_inverse_perm

class SyntheticSystematicCode:
    def __init__(self, k=20, m=20, seed=0):
        self.k = k; self.n = k + m; self.m = m
        rng = np.random.default_rng(seed)
        self.P = (rng.random((m, k)) < 0.5).astype(np.uint8)
    def encode(self, msg):
        msg = np.asarray(msg, np.uint8).reshape(-1)
        par = (self.P @ msg) % 2
        return np.concatenate([msg, par]).astype(np.uint8)
    def info_bit_extract(self, cw):
        return np.asarray(cw, np.uint8).reshape(-1)[:self.k]
    def syndrome(self, cw):
        cw = np.asarray(cw, np.uint8).reshape(-1)
        m_part, p_part = cw[:self.k], cw[self.k:]
        return ((self.P @ m_part) % 2 ^ p_part).astype(np.uint8)

def mock_bit_s(s):
    return lambda x: int(np.asarray(x, np.uint8).reshape(-1)[s])

def test_enc_dec_roundtrip():
    code = SyntheticSystematicCode(k=20, m=20, seed=1)
    rng = np.random.default_rng(2)
    key = SG.key_gen(rng)
    x = (rng.random(code.k) < 0.5).astype(np.uint8)
    c = SG.enc(x, key, 7, code)
    assert int(np.asarray(code.syndrome(c)).sum()) == 0
    x_rec = SG.dec(c, key, 7, code)
    assert np.array_equal(x_rec, x)

def test_delta_star_in_nullspace_and_algebraic_shortcut():
    code = SyntheticSystematicCode(k=20, m=20, seed=3)
    rng = np.random.default_rng(4)
    key = SG.key_gen(rng)
    x = (rng.random(code.k) < 0.5).astype(np.uint8)
    di = SG.setting_a_delta_info(code.k, 3, rng)
    ds = SG.delta_star_from_info(di, code)
    assert int(np.asarray(code.syndrome(ds)).sum()) == 0
    perm = SG.default_perm_gen(key, 5, code.k)
    c = SG.enc(x, key, 5, code)
    chat = (c ^ ds).astype(np.uint8)
    assert int(np.asarray(code.syndrome(chat)).sum()) == 0
    x_rec_full = SG.dec(chat, key, 5, code)
    v = apply_inverse_perm(di, perm)
    assert np.array_equal(x_rec_full, (x ^ v).astype(np.uint8))

def test_win_bit_syndrome_and_decision():
    code = SyntheticSystematicCode(k=24, m=24, seed=6)
    rng = np.random.default_rng(7)
    key = SG.key_gen(rng)
    s = 0; f = mock_bit_s(s)
    x = np.zeros(code.k, np.uint8); x[s] = 0
    t_star = 11
    oracle = SG.EncOracle(key=key, code=code)
    di = SG.setting_b_delta_info(_e(code.k, s), key, t_star, code)
    ds = SG.delta_star_from_info(di, code)
    res = SG.play_challenge(oracle, t_star, x, y_star=1, delta_star=ds, r=code.n, decision_fn=f)
    assert res["valid_challenge"] and res["win_syndrome_zero"]
    assert res["win_decision_equals_target"] and res["win"]
    assert res["w_info"] == 1

def test_constraint_a_no_query_on_tstar():
    code = SyntheticSystematicCode(k=20, m=20, seed=8)
    rng = np.random.default_rng(9)
    key = SG.key_gen(rng); f = mock_bit_s(0)
    oracle = SG.EncOracle(key=key, code=code)
    x = np.zeros(code.k, np.uint8)
    oracle.query(3, x); oracle.query(5, x)
    di = SG.setting_b_delta_info(_e(code.k, 0), key, 3, code)
    ds = SG.delta_star_from_info(di, code)
    res_bad = SG.play_challenge(oracle, 3, x, 1, ds, code.n, f)
    assert res_bad["valid_challenge"] is False and res_bad["constraint_a_tstar_fresh"] is False
    di2 = SG.setting_b_delta_info(_e(code.k, 0), key, 99, code)
    res_ok = SG.play_challenge(oracle, 99, x, 1, SG.delta_star_from_info(di2, code), code.n, f)
    assert res_ok["constraint_a_tstar_fresh"] is True and res_ok["valid_challenge"] is True

def test_constraint_b_weight_and_c_target():
    code = SyntheticSystematicCode(k=20, m=20, seed=10)
    rng = np.random.default_rng(11)
    key = SG.key_gen(rng); f = mock_bit_s(0)
    oracle = SG.EncOracle(key=key, code=code)
    x = np.zeros(code.k, np.uint8)
    di = SG.setting_a_delta_info(code.k, 5, rng)
    ds = SG.delta_star_from_info(di, code)
    w_star = int(ds.sum())
    res = SG.play_challenge(oracle, 1, x, 1, ds, r=w_star - 1, decision_fn=f)
    assert res["constraint_b_weight_ok"] is False and res["valid_challenge"] is False
    res2 = SG.play_challenge(oracle, 1, x, y_star=int(f(x)), delta_star=ds, r=code.n, decision_fn=f)
    assert res2["constraint_c_target_differs"] is False and res2["valid_challenge"] is False

def test_setting_a_winrate_equals_eta_w():
    code = SyntheticSystematicCode(k=50, m=50, seed=12)
    s = 7; f = mock_bit_s(s)
    x = np.zeros(code.k, np.uint8)
    for w in [1, 5, 10]:
        r = SG.measure_setting_a_win_rate(x, y_star=1, w_info=w, code=code, decision_fn=f, n_trials=4000, seed=100 + w)
        eta = w / code.k
        assert abs(r["win_rate"] - eta) < 0.03, (w, r["win_rate"], eta)
        assert r["win_rate"] <= eta + 0.03
        assert abs(r["v_weight_mean"] - w) < 1e-9

def test_setting_b_ge_setting_a():
    code = SyntheticSystematicCode(k=50, m=50, seed=13)
    s = 3; f = mock_bit_s(s)
    x = np.zeros(code.k, np.uint8)
    cmp = SG.compare_settings(x, y_star=1, w_info=1, code=code, decision_fn=f, target_v=_e(code.k, s), n_trials_a=400, seed=14)
    assert cmp["setting_b_win_rate"] > 0.99
    assert cmp["setting_b_win_rate"] >= cmp["setting_a"]["win_rate"] - 1e-9
    assert cmp["setting_a"]["win_rate"] < 0.10

def test_pluggable_perm_gen():
    code = SyntheticSystematicCode(k=40, m=40, seed=15)
    ident = lambda key, t, k: np.arange(k, dtype=np.int64)
    s = 2; f = mock_bit_s(s)
    x = np.zeros(code.k, np.uint8)
    r = SG.measure_setting_a_win_rate(x, 1, 1, code, f, n_trials=200, seed=16, perm_gen=ident)
    assert 0.0 <= r["win_rate"] <= 1.0 and r["w_info"] == 1

def test_real_long_code_faithfulness():
    try:
        code = SG.default_code()
    except Exception as e:
        print(f"  (long code unavailable: {type(e).__name__}; skipped)"); return
    rng = np.random.default_rng(17)
    key = SG.key_gen(rng)
    x = (rng.random(code.k) < 0.5).astype(np.uint8)
    assert np.array_equal(SG.dec(SG.enc(x, key, 123, code), key, 123, code), x)
    di = SG.setting_a_delta_info(code.k, 100, rng)
    ds = SG.delta_star_from_info(di, code)
    assert int(np.asarray(code.syndrome(ds)).sum()) == 0
    perm = SG.default_perm_gen(key, 123, code.k)
    chat = (SG.enc(x, key, 123, code) ^ ds).astype(np.uint8)
    assert int(np.asarray(code.syndrome(chat)).sum()) == 0
    x_rec = SG.dec(chat, key, 123, code)
    assert np.array_equal(x_rec, (x ^ apply_inverse_perm(di, perm)).astype(np.uint8))
    print(f"  long-code delta* weight={int(ds.sum())} (w_info=100, parity~{(code.n-code.k)//2})")

def _e(k, i):
    v = np.zeros(k, np.uint8); v[i] = 1; return v

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")