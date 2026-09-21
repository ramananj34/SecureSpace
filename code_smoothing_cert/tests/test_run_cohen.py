from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc"), str(_ROOT / "baseline_fgsm_pgd")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def test_frozen_config_constants():
    import run_e14_full as R
    assert R.FROZEN_GRID == [1, 4, 16, 32, 64, 128, 256, 512]
    assert R.FROZEN_N == 300 and R.FROZEN_SEED == 0

def test_r_cert_reuse_is_frozen_function():
    import run_e14_full as R
    from run_e14 import r_cert as frozen_rc
    assert R._r_cert_frozen is frozen_rc

def test_cohen_noised_detector_adds_noise_on_footprint_only():
    import run_cohen as RC
    calls = {}
    label = (100, 150)
    T = 400
    F_idx = np.arange(90, 161)
    tele_q = np.zeros(T)
    def mock_detect(chan, model, tf, tele, cmds, cfg):
        calls["last_footprint_absmean"] = float(np.mean(np.abs(tele[F_idx])))
        return [(100, 150)] if np.mean(np.abs(tele[F_idx])) < 0.3 else []
    def mock_missed(E, l): return not any(not (e[1] < l[0] or l[1] < e[0]) for e in E)
    import run_cohen
    run_cohen.detect = mock_detect
    run_cohen.missed = mock_missed
    fn = RC.make_noised_detector("X", None, None, tele_q, None, None, F_idx, label)
    rng = np.random.default_rng(0)
    assert fn(None, 0.0, rng) is True
    losses = sum(0 if fn(None, 2.0, np.random.default_rng([1, i])) else 1 for i in range(20))
    assert losses > 10
    def spy_detect(chan, model, tf, tele, cmds, cfg):
        assert np.allclose(tele[:90], 0.0) and np.allclose(tele[161:], 0.0)
        return [(100, 150)]
    run_cohen.detect = spy_detect; run_cohen.missed = mock_missed
    fn2 = RC.make_noised_detector("X", None, None, tele_q, None, None, F_idx, label)
    fn2(None, 1.0, np.random.default_rng(2))

def test_population_excludes_m6(monkeypatch=None):
    import run_e14_full as R
    from smap_msl_dataset_api import EXCLUDED_CHANNELS
    assert "M-6" in EXCLUDED_CHANNELS
    import inspect
    src = inspect.getsource(R.population)
    assert "EXCLUDED_CHANNELS" in src

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")