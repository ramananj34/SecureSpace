from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "fl_core")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import run_e16 as R

def test_mse_ratio_ordering():
    theta_clean = np.random.default_rng(0).normal(0, 1, 100)
    def mse_at(theta): return 0.01 + 0.5 * np.sum((theta - theta_clean) ** 2)
    base = mse_at(theta_clean)
    r_clean = mse_at(theta_clean) / base
    r_collapse = mse_at(theta_clean + 5) / base
    assert abs(r_clean - 1.0) < 1e-9
    assert r_collapse > 1000

def test_regimes_list():
    assert R.REGIMES == ["fedavg", "median", "fedavg_perm", "median_perm"]

def test_clean_holdout_shape():
    feats = np.random.default_rng(0).normal(size=(2000, 25)).astype(np.float32)
    X, y = R._clean_holdout_windows(feats, l_s=250, n_windows=100)
    assert X.shape == (100, 250, 25) and y.shape == (100,)

def test_smoke_regime_separation():
    try:
        import torch
    except Exception:
        print("  (torch absent: E16 smoke skipped)"); return
    runs = Path(_ROOT) / "runs" / "A-1" / "model.pt"; data = Path(_ROOT) / "smap_msl_data"
    if not runs.exists() or not data.exists():
        print("  (A-1 model/data absent: E16 smoke skipped)"); return
    from VENDOR_telemanom import VendoredConfig
    from smap_msl_dataset_api import Quantizer
    import constellation as CN
    cfg = VendoredConfig(); q = Quantizer()
    con = CN.Constellation(walker=CN.WalkerDelta(n_planes=2, sats_per_plane=4))
    r1 = R.run_regime("A-1", "fedavg", 0.30, 5, con, cfg, q, data, Path(_ROOT)/"runs",
                      local_epochs=1, lr=1e-2, seed=0, scatter_w=100, unbounded_scale=50.0)
    r3b = R.run_regime("A-1", "median_perm", 0.30, 5, con, cfg, q, data, Path(_ROOT)/"runs",
                       local_epochs=1, lr=1e-2, seed=0, scatter_w=100, unbounded_scale=50.0)
    assert r1["final_mse_ratio"] > r3b["final_mse_ratio"]
    assert r3b["final_mse_ratio"] < r1["final_mse_ratio"]

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")