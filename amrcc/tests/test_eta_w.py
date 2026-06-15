from __future__ import annotations
import sys
import time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_AMRCC = _THIS.parent
_ROOT = _AMRCC.parent
for _p in [str(_ROOT), str(_AMRCC), str(_ROOT / "smap_msl_data"), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from smap_msl_dataset_api import Quantizer
from eta_w import eta_w_estimate, eta_curve, max_eta

T = 600
WINDOW = (250, 100)
LABEL = [300, 330]

def _tele():
    rng = np.random.default_rng(0)
    return (0.5 * np.sin(np.linspace(0, 30, T)) + 0.02 * rng.standard_normal(T)).astype(np.float32)

def _mock(quantizer, tele, region, thresh):
    ref = np.asarray(quantizer.dequantize(quantizer.quantize(tele)), np.float32)
    sl = slice(region[0], region[0] + region[1])
    def detect_fn(chan_id, model, tf, sig, cmds, cfg):
        return np.asarray(sig, np.float32)
    def missed_fn(E, label):
        return bool(np.abs(np.asarray(E)[sl] - ref[sl]).max() > thresh)
    return detect_fn, missed_fn

def test_zero_weight_no_flip():
    q = Quantizer()
    tele = _tele()
    dfn, mfn = _mock(q, tele, WINDOW, thresh=1e-6)
    r = eta_w_estimate("X", None, None, tele, None, [LABEL], LABEL, SimpleNamespace(), q, w=0, n_samples=50, window=WINDOW, detect_fn=dfn, missed_fn=mfn)
    assert r["w"] == 0 and r["misses"] == 0 and r["eta_hat"] == 0.0
    assert r["missed_clean"] is False

def test_region_size_and_weight_bounds():
    q = Quantizer()
    tele = _tele()
    dfn, mfn = _mock(q, tele, WINDOW, thresh=0.05)
    r = eta_w_estimate("X", None, None, tele, None, [LABEL], LABEL, SimpleNamespace(), q, w=10, n_samples=20, window=WINDOW, detect_fn=dfn, missed_fn=mfn)
    assert r["region_timesteps"] == 100 and r["n_bits_region"] == 800
    try:
        eta_w_estimate("X", None, None, tele, None, [LABEL], LABEL, SimpleNamespace(), q, w=801, n_samples=5, window=WINDOW, detect_fn=dfn, missed_fn=mfn)
        assert False, "expected w out-of-range error"
    except ValueError:
        pass

def test_eta_monotone_trend_and_ci():
    q = Quantizer()
    tele = _tele()
    dfn, mfn = _mock(q, tele, WINDOW, thresh=0.4)
    curve = eta_curve("X", None, None, tele, None, [LABEL], LABEL, SimpleNamespace(), q, weights=[0, 5, 20, 80, 300], n_samples=300, window=WINDOW, detect_fn=dfn, missed_fn=mfn, seed=7)
    etas = [c["eta_hat"] for c in curve]
    assert etas[0] == 0.0
    assert etas[-1] > etas[0]
    assert all(b >= a - 0.12 for a, b in zip(etas, etas[1:]))
    for c in curve:
        assert 0.0 <= c["ci_lo"] <= c["eta_hat"] <= c["ci_hi"] <= 1.0
    m = max_eta(curve)
    assert m["max_eta_hat"] == max(etas)

def test_determinism_same_seed():
    q = Quantizer()
    tele = _tele()
    dfn, mfn = _mock(q, tele, WINDOW, thresh=0.2)
    kw = dict(window=WINDOW, detect_fn=dfn, missed_fn=mfn, seed=123)
    a = eta_w_estimate("X", None, None, tele, None, [LABEL], LABEL, SimpleNamespace(), q, w=40, n_samples=200, **kw)
    b = eta_w_estimate("X", None, None, tele, None, [LABEL], LABEL, SimpleNamespace(), q, w=40, n_samples=200, **kw)
    assert a["misses"] == b["misses"] and a["eta_hat"] == b["eta_hat"]

def test_flips_land_in_region_only():
    q = Quantizer()
    tele = _tele()
    levels0 = q.quantize(tele)
    bits0 = np.asarray(q.levels_to_bits(levels0))
    from sampling import selection_sample
    rng = np.random.default_rng(0)
    ts = np.arange(WINDOW[0], WINDOW[0] + WINDOW[1])
    flat = selection_sample(800, 1, rng)
    bits = bits0.copy()
    bits[ts[flat // 8], flat % 8] ^= 1
    changed = np.where(np.any(bits != bits0, axis=1))[0]
    assert changed.size == 1 and WINDOW[0] <= changed[0] < WINDOW[0] + WINDOW[1]

if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for fn in fns:
        t0 = time.time()
        fn()
        print(f"PASS  {fn.__name__:42s} {time.time() - t0:6.2f}s")
    print(f"\nall {len(fns)} eta_w tests passed")