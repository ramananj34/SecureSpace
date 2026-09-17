from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lstm_plausibility as LP

def _persistence_predict(model, tele, cmds, P, l_s, batch_windows=512):
    P = np.asarray(P, np.int64)
    return np.asarray(tele, np.float64)[P + l_s - 1]

def _target_range_matches():
    T = 600; l_s = 50; bs, bl = 200, 100
    P = LP._target_windows(bs, bl, l_s, T)
    assert P[0] == bs - l_s and P[-1] == bs + bl - 1 - l_s and P.size == bl

def test_target_windows():
    _target_range_matches()
    P = LP._target_windows(10, 100, 50, 600)
    assert P[0] == 0 and (P + 50).min() == 50

def test_reorder_error_smooth_low_shuffle_high():
    l_s = 50; T = 600; bs, bl = 200, 100
    tele = np.zeros(T)
    tele[:bs] = np.linspace(-0.5, 0.0, bs)
    tele[bs:bs + bl] = np.linspace(0.0, 0.9, bl)
    tele[bs + bl:] = 0.9
    cmds = np.zeros((T, 4))
    err_true, n = LP.window_reorder_error(None, tele, cmds, bs, bl, np.arange(bl), l_s, predict_fn=_persistence_predict)
    rng = np.random.default_rng(0)
    err_sh, _ = LP.window_reorder_error(None, tele, cmds, bs, bl, rng.permutation(bl), l_s, predict_fn=_persistence_predict)
    assert n == bl
    assert err_true < err_sh

def test_fitness_factory_sign():
    l_s = 50; T = 600; bs, bl = 200, 100
    tele = np.concatenate([np.linspace(-0.5, 0, bs), np.linspace(0, 0.9, bl), np.full(300, 0.9)])[:T]
    cmds = np.zeros((T, 3))
    fit = LP.lstm_plausibility_fitness_factory(None, tele, cmds, bs, bl, l_s, predict_fn=_persistence_predict)
    f_true = fit(np.arange(bl))
    f_sh = fit(np.random.default_rng(1).permutation(bl))
    assert f_true > f_sh

def test_shuffle_sensitivity_detects_true_order():
    l_s = 50; T = 600; bs, bl = 200, 100
    tele = np.concatenate([np.linspace(-0.5, 0, bs), np.linspace(0, 0.9, bl), np.full(300, 0.9)])[:T]
    cmds = np.zeros((T, 3))
    res = LP.shuffle_sensitivity(None, tele, cmds, bs, bl, l_s, n_shuffles=100, seed=2, predict_fn=_persistence_predict)
    assert res["frac_shuffles_worse_than_true"] > 0.8
    assert res["z_true_below_shuffles"] > 1.0

def test_shuffle_sensitivity_null_on_iid():
    l_s = 50; T = 600; bs, bl = 200, 100
    fracs = []
    for r in range(16):
        rng = np.random.default_rng([3, r])
        tele = rng.uniform(-0.5, 0.5, T)
        cmds = np.zeros((T, 3))
        res = LP.shuffle_sensitivity(None, tele, cmds, bs, bl, l_s, n_shuffles=100, seed=1000 + r, predict_fn=_persistence_predict)
        fracs.append(res["frac_shuffles_worse_than_true"])
    assert 0.3 < np.mean(fracs) < 0.7

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")