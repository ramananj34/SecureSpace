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
import discriminator as D
from recovery_metrics import (m1_random_baseline, m1_bit_accuracy, score_candidate, observe)

Q = Quantizer()

def _biased_smooth_slot(rng, n_coords=100):
    x = 0.55 + 0.08 * np.cumsum(rng.normal(0, 1, n_coords)) / np.sqrt(n_coords)
    x = np.clip(x, -0.9, 0.9)
    return np.asarray(Q.levels_to_bits(Q.quantize(x)), np.uint8).reshape(-1), x

def test_roc_auc_endpoints():
    s = np.array([0.1, 0.2, 0.3, 0.4]); y = np.array([0, 0, 1, 1])
    assert abs(D.roc_auc(s, y) - 1.0) < 1e-9
    assert abs(D.roc_auc(-s, y) - 0.0) < 1e-9
    rng = np.random.default_rng(0)
    sc = rng.random(4000); lab = (np.arange(4000) % 2)
    assert abs(D.roc_auc(sc, lab) - 0.5) < 0.05

def test_marginal_auc_high_when_structured():
    rng = np.random.default_rng(1)
    slots = np.stack([_biased_smooth_slot(np.random.default_rng([1, i]))[0] for i in range(400)])
    res = D.marginal_discriminator_auc(slots[:300], slots[300:], seed=2)
    assert res["auc_marginal"] > 0.9

def test_marginal_auc_chance_when_iid_uniform_planes():
    rng = np.random.default_rng(3)
    def slot():
        x = rng.uniform(-0.98, 0.98, 100)
        return np.asarray(Q.levels_to_bits(Q.quantize(x)), np.uint8).reshape(-1)
    slots = np.stack([slot() for _ in range(400)])
    res = D.marginal_discriminator_auc(slots[:300], slots[300:], seed=4)
    assert abs(res["auc_marginal"] - 0.5) < 0.06

def test_marginal_recovery_lifts_m1_not_coord_order():
    b = 8; slot = 800; k = slot; header = 0
    train = np.stack([_biased_smooth_slot(np.random.default_rng([5, i]))[0] for i in range(400)])
    p = D.position_marginals(train)
    m1_lifts, spearmans, m4s = [], [], []
    for t in range(10):
        m_in, _ = _biased_smooth_slot(np.random.default_rng([50, t]))
        perm = np.random.default_rng([51, t]).permutation(k)
        m_obs = observe(m_in, perm)
        pi_hat = D.marginal_recovery_permutation(m_obs, p, rng=np.random.default_rng([52, t]))
        rec = score_candidate(m_in, perm, pi_hat, quantizer=Q, b=b, header_bits=header, slot_bits=slot)
        base = m1_random_baseline(k, int(m_in.sum()))
        m1_lifts.append(rec["m1_bit_acc"] - base)
        spearmans.append(rec["telem_order"]["abs_spearman"])
        m4s.append(rec["m4_msb_loc"])
    assert np.mean(m1_lifts) > 0.02
    assert np.mean(m4s) < 6.0 / 100
    assert np.mean(spearmans) < 0.5

def test_marginal_recovery_permutation_matches_reconstruction():
    rng = np.random.default_rng(6)
    k = 800
    m_in = (rng.random(k) < 0.4).astype(np.uint8)
    p = rng.random(k)
    perm = rng.permutation(k)
    m_obs = observe(m_in, perm)
    from recovery_metrics import m1_bit_accuracy as m1
    rec_bits = D.marginal_recovery_reconstruction(m_obs, p)
    pi_hat = D.marginal_recovery_permutation(m_obs, p)
    m_hat = np.asarray(m_obs)[np.argsort(pi_hat, kind="mergesort")]
    assert np.array_equal(m_hat, rec_bits)

def test_slots_from_stream_shape():
    x = np.random.default_rng(7).uniform(-0.9, 0.9, 1050)
    S = D.slots_from_stream(x, Q, window=100)
    assert S.shape == (10, 800) and S.dtype == np.uint8

def test_mlp_discriminator_if_torch():
    if not D._HAVE_TORCH:
        print("  (torch absent: MLP discriminator test skipped)"); return
    slots = np.stack([_biased_smooth_slot(np.random.default_rng([8, i]))[0] for i in range(500)])
    state, m = D.train_discriminator(slots, hidden=(128, 32), epochs=25, seed=0, verbose=False)
    mres = D.marginal_discriminator_auc(slots[:350], slots[350:], seed=1)
    assert m["auc_joint_test"] > 0.9
    assert m["auc_joint_test"] >= mres["auc_marginal"] - 0.05

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")