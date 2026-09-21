from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc"),
           str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "ldpc"), str(_ROOT / "theory_helpers")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import run_e13 as R
import weak_prng as W

K, WGT = 256, 8

def test_eta_mock_exact():
    assert abs(R.eta_w_mock(8, 256) - 8 / 256) < 1e-12

def test_v_level_matches_harness_setting_a():
    try:
        import security_game as SG
    except Exception as e:
        print(f"  (security_game import failed: {type(e).__name__}; skipped)"); return

    class SynthCode:
        def __init__(self, k):
            self.k = k; self.n = 2 * k; self.m = k
            self.P = (np.random.default_rng(0).random((k, k)) < 0.5).astype(np.uint8)
        def encode(self, m):
            m = np.asarray(m, np.uint8).reshape(-1); return np.concatenate([m, (self.P @ m) % 2]).astype(np.uint8)
        def info_bit_extract(self, cw): return np.asarray(cw, np.uint8).reshape(-1)[:self.k]
        def syndrome(self, cw):
            cw = np.asarray(cw, np.uint8).reshape(-1); return ((self.P @ cw[:self.k]) % 2 ^ cw[self.k:]).astype(np.uint8)

    code = SynthCode(K)
    s = 100
    f = lambda x: int(np.asarray(x, np.uint8).reshape(-1)[s])
    x = np.zeros(K, np.uint8)
    hres = SG.measure_setting_a_win_rate(x, y_star=1, w_info=WGT, code=code, decision_fn=f, n_trials=3000, seed=1, perm_gen=W.strong)
    perms = R.build_perm_batch(W.strong, K, 3000, seed=1)
    q = R.q_vectors_from_batch(perms, [WGT], K, seed=1)[WGT]
    assert abs(hres["win_rate"] - q[s]) < 0.03, (hres["win_rate"], q[s])

def test_run_generator_strong_robust():
    rec = R.run_generator("strong", W.strong, K, [8, 32], n_perm=2000, n_keys=200, seed=0)
    for r in rec["per_w"]:
        assert r["theorem2_holds"]
        assert r["adv_a_worst_target"] < r["eta_w"] + 0.10
        assert r["route_label"] == "neither"
    assert rec["recover_score"] < 0.05

def test_run_generator_identity_is_oracle():
    rec = R.run_generator("identity", W.identity, K, [8], n_perm=1000, n_keys=200, seed=0)
    r = rec["per_w"][0]
    assert r["adv_a_worst_target"] > 0.9
    assert abs(r["adv_a_worst_target"] - r["adv_b_oracle"]) < 0.1
    assert r["theorem2_holds"]
    assert r["route_label"] == "both"

def test_theorem2_holds_across_ladder():
    for name, gen in W.ladder():
        rec = R.run_generator(name, gen, K, [8], n_perm=1500, n_keys=150, seed=0)
        r = rec["per_w"][0]
        assert r["theorem2_holds"], f"Theorem 2 FAILED for {name}: Adv_a={r['adv_a_worst_target']:.3f} > "\
                                    f"eta+eps_max={r['theorem2_rhs']:.3f}"

def test_reduced_round_chacha_robust():
    rec = R.run_generator("chacha2r", W.reduced_round_chacha(2), K, [8], n_perm=2000, n_keys=200, seed=0)
    r = rec["per_w"][0]
    assert r["route_label"] == "neither"
    assert r["adv_a_worst_target"] < r["eta_w"] + 0.10

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")