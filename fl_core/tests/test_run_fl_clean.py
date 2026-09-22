from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import federated as FED

def test_clean_round_all_aggregators_reduce_loss():
    rng = np.random.default_rng(0)
    d, M, lr = 20, 40, 0.5
    c = rng.normal(0, 1, d) + rng.normal(0, 0.1, (M, d))
    theta = rng.normal(0, 1, d)
    deltas = lr * (c - theta)
    def avg_loss(th): return float(np.mean([0.5 * np.sum((th - c[i])**2) for i in range(M)]))
    L0 = avg_loss(theta)
    for agg in ["fedavg", "coord_median", "krum"]:
        gd = FED.aggregate(agg, deltas, f=0)
        assert avg_loss(theta + gd) < L0, agg

def test_clean_round_no_byzantine_krum_f0():
    rng = np.random.default_rng(1)
    deltas = rng.normal(0, 0.1, (40, 10))
    gd = FED.aggregate("krum", deltas, f=0)
    assert gd.shape == (10,) and np.isfinite(gd).all()

def test_full_run_smoke():
    try:
        import torch
    except Exception:
        print("  (torch absent: full clean-round smoke skipped)"); return
    from pathlib import Path as P
    runs = P(_ROOT) / "runs" / "A-1" / "model.pt"
    data = P(_ROOT) / "smap_msl_data"
    if not runs.exists() or not data.exists():
        print("  (A-1 model / data absent: full clean-round smoke skipped)"); return
    import run_fl_clean as R
    from VENDOR_telemanom import VendoredConfig
    from smap_msl_dataset_api import Quantizer
    import constellation as CN
    cfg = VendoredConfig(); q = Quantizer()
    con = CN.Constellation(walker=CN.WalkerDelta(n_planes=2, sats_per_plane=3))
    rec = R.run_clean_round("A-1", data, P(_ROOT) / "runs", cfg, q, con,
                            aggregators=["fedavg", "coord_median"], local_epochs=1, lr=1e-3, seed=0,
                            do_theory=True)
    for agg in ["fedavg", "coord_median"]:
        assert rec["aggregators"][agg]["update_finite"]
        assert 0.0 <= rec["aggregators"][agg]["recall"] <= 1.0
    assert rec["d_adv_prime_single_anchor"]["d_adv_prime_int"] == 1

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")