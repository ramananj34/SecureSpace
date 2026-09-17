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
import run_e10 as R
from recovery_metrics import observe, m1_random_baseline

Q = Quantizer()

def test_size_layout():
    lay = R.size_layout(32400)
    assert lay["k"] == 32400 and lay["header_bits"] == 400 and lay["n_coords"] == 4000
    assert lay["slot_bits"] == 32000
    lay2 = R.size_layout(800)
    assert lay2["k"] == 800 and lay2["header_bits"] == 0 and lay2["n_coords"] == 100

def test_block_to_message_shapes_and_sigma0():
    block = np.linspace(-0.5, 0.5, 100)
    m0, xs0 = R.block_to_message(block, 0.0, np.random.default_rng(0), Q, header_bits=0)
    ref_bits = np.asarray(Q.levels_to_bits(Q.quantize(block)), np.uint8).reshape(-1)
    assert m0.size == 800 and np.array_equal(m0, ref_bits)
    assert xs0.size == 100 and np.allclose(xs0, Q.dequantize(Q.quantize(block)))

def test_bitrun_delta_matches_full():
    rng = np.random.default_rng(1)
    k = 60
    m_obs = (rng.random(k) < 0.5).astype(np.int64)
    v = m_obs
    def full(sigma):
        s = v[sigma]; return float(np.sum(s[:-1] == s[1:]))
    def dfn(sigma, i, j):
        if i == j: return 0.0
        a = v[sigma[i]]; b = v[sigma[j]]
        if a == b: return 0.0
        pairs = set()
        for d in (i, j):
            if d > 0: pairs.add((d - 1, d))
            if d < k - 1: pairs.add((d, d + 1))
        old = new = 0
        for (p, q) in pairs:
            vp = v[sigma[p]]; vq = v[sigma[q]]
            old += (vp == vq)
            np_ = b if p == i else (a if p == j else vp)
            nq_ = b if q == i else (a if q == j else vq)
            new += (np_ == nq_)
        return float(new - old)
    sigma = rng.permutation(k)
    for _ in range(300):
        i, j = int(rng.integers(0, k)), int(rng.integers(0, k))
        if i == j: continue
        f0 = full(sigma); d = dfn(sigma, i, j)
        cand = sigma.copy(); cand[i], cand[j] = cand[j], cand[i]
        assert abs((full(cand) - f0) - d) < 1e-9

def test_arm_baseline_at_chance():
    rng = np.random.default_rng(2)
    lay = R.size_layout(800)
    block = np.clip(np.cumsum(rng.normal(0, 0.05, 100)), -0.9, 0.9)
    m_in, _ = R.block_to_message(block, 0.0, rng, Q, 0)
    perm = rng.permutation(lay["k"])
    b = R.arm_baseline(m_in, perm, Q, lay, n_draws=8, seed=3)
    assert abs(b["m1_ratio_mean"] - 1.0) < 0.03
    assert b["m2_pos_acc_mean"] < 0.01

def test_arm_ceiling_recovers_magnitude_rank_above_chance():
    n = 128
    x = np.clip(np.cumsum(np.random.default_rng(4).normal(0, 0.05, n)), -0.9, 0.9)
    xs0 = np.asarray(Q.dequantize(Q.quantize(x)), np.float64)
    r0 = R.arm_ceiling(xs0, iters=40000, restarts=3, seed=5, max_seg=16, seg_frac=0.3)
    assert r0["telem_order_abs_spearman"] > 0.25
    assert r0["telem_order_abs_spearman"] < 0.90
    noise = np.asarray(Q.dequantize(Q.quantize(np.random.default_rng(100).uniform(-0.5, 0.5, n))), np.float64)
    rn = R.arm_ceiling(noise, iters=40000, restarts=3, seed=111, max_seg=16, seg_frac=0.3)
    assert rn["telem_order_abs_spearman"] < 0.20

def test_arm_marginal_lifts_m1_not_m4():
    rng = np.random.default_rng(8)
    lay = R.size_layout(800); nc = lay["n_coords"]
    def blk(s):
        x = 0.55 + 0.08 * np.cumsum(np.random.default_rng([8, s]).normal(0, 1, nc)) / np.sqrt(nc)
        return np.clip(x, -0.9, 0.9)
    corpus, ncorp = R.collect_corpus(np.concatenate([blk(s) for s in range(20)]), nc, 0, 0.0, Q, n_frames=16, seed=9)
    p = R.arm_marginal.__globals__["DISC"].position_marginals(corpus)
    m_in, _ = R.block_to_message(blk(99), 0.0, rng, Q, 0)
    perm = rng.permutation(lay["k"])
    res = R.arm_marginal(m_in, perm, p, Q, lay, n_tiebreak=4, seed=10)
    base = m1_random_baseline(lay["k"], int(m_in.sum()))
    assert res["m1_bit_acc"] > base + 0.02
    assert res["m4_msb_loc"] < 6.0 / nc

def test_necessary_condition_marginal_auc():
    def blk(s):
        x = 0.55 + 0.08 * np.cumsum(np.random.default_rng([11, s]).normal(0, 1, 100)) / 10.0
        return np.clip(x, -0.9, 0.9)
    frames = np.stack([R.block_to_message(blk(s), 0.0, np.random.default_rng([12, s]), Q, 0)[0] for s in range(400)])
    nc = R.necessary_condition(frames, seed=0, do_joint=False, disc_kw=None)
    assert nc["auc_marginal"] > 0.85

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")