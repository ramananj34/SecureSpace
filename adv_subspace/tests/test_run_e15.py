from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc"), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import run_e15 as R

def test_anomaly_window_indices():
    w = R.anomaly_window_indices((300, 400), T=1000, l_s=250)
    assert w == (50, 300)
    w2 = R.anomaly_window_indices((100, 200), T=1000, l_s=250)
    assert w2 == (0, 250)
    assert R.anomaly_window_indices((10, 20), T=200, l_s=250) is None

def test_frame_scale_constants():
    assert R.L_S == 250 and R.FRAME_COORDS == 100 and R.SLOT_BITS == 800 and R.B == 8

def test_run_point_with_mock_model():
    try:
        import torch
    except Exception:
        print("  (torch absent: run_point test skipped)"); return
    from smap_msl_dataset_api import Quantizer
    from VENDOR_telemanom import VendoredConfig
    L, F = 250, 25
    a = torch.tensor(np.random.default_rng(0).normal(size=(L * F,)), dtype=torch.float32)
    class Lin(torch.nn.Module):
        def forward(self, x): return (x.reshape(x.shape[0], -1) @ a)[:, None]
    model = Lin()
    T = 600
    tele = np.sin(np.linspace(0, 20, T)) * 0.5
    cmds = np.zeros((T, F - 1))
    q = Quantizer(); cfg = VendoredConfig()
    rec = R.run_point("MOCK", model, None, tele, cmds, cfg, q, label=(300, 350), T=T, n_mc=4000, seed=1)
    assert rec["d_adv_int"] == 1
    assert 0.0 < rec["d_adv_effective_trace"] <= 1.0 + 1e-9
    assert rec["d_adv_geometric_dim"] <= 1
    assert rec["theorem4"] is not None
    for w, v in rec["theorem4"].items():
        if int(w) >= 8:
            assert v["within_20pct"], (w, v["rel_error"])
    assert rec["reduction_factor_at_dadv_eff"] > 100

def test_population_excludes_m6():
    import inspect
    from smap_msl_dataset_api import EXCLUDED_CHANNELS
    assert "M-6" in EXCLUDED_CHANNELS
    assert "EXCLUDED_CHANNELS" in inspect.getsource(R.population)

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")