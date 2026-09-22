from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import constellation as CN

def test_sat_count_and_positions():
    wd = CN.WalkerDelta(n_planes=5, sats_per_plane=8)
    assert wd.n_sats == 40
    pts = wd.positions(0.0)
    assert pts.shape == (40, 3)
    r = np.linalg.norm(pts, axis=1)
    assert np.allclose(r, wd.orbit_radius_km, atol=1e-6)

def test_contact_graph_symmetric_no_selfloops():
    wd = CN.WalkerDelta()
    A = CN.contact_graph(wd, 0.0)
    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 0)
    assert set(np.unique(A)).issubset({0, 1})

def test_connected_but_not_complete_at_default_range():
    wd = CN.WalkerDelta(max_isl_range_km=5000.0)
    A = CN.contact_graph(wd, 0.0)
    s = CN.graph_stats(A)
    assert s["connected"] is True
    assert s["density"] < 0.5
    assert s["min_degree"] >= 1

def test_shorter_range_disconnects():
    A_far = CN.contact_graph(CN.WalkerDelta(max_isl_range_km=5000.0), 0.0)
    A_near = CN.contact_graph(CN.WalkerDelta(max_isl_range_km=2000.0), 0.0)
    assert A_near.sum() < A_far.sum()
    assert CN.graph_stats(A_near)["connected"] is False

def test_los_occlusion():
    R = 6371.0 + 550.0
    a = np.array([R, 0, 0]); b = np.array([-R, 0, 0])
    assert CN._los_clear(a, b, 6371.0) is False
    c = np.array([R, 0, 0]); d = np.array([R * 0.99, R * 0.1, 0])
    assert CN._los_clear(c, d, 6371.0) is True

def test_coordinator_reachability():
    con = CN.Constellation(walker=CN.WalkerDelta(max_isl_range_km=5000.0), coordinator=0)
    mask = con.reachable_from_coordinator(0.0)
    assert mask.shape == (40,) and mask[0]
    assert mask.all()
    con2 = CN.Constellation(walker=CN.WalkerDelta(max_isl_range_km=2000.0), coordinator=0)
    assert not con2.reachable_from_coordinator(0.0).all()

def test_graph_stats_shape():
    A = CN.contact_graph(CN.WalkerDelta(), 0.0)
    s = CN.graph_stats(A)
    for k in ("n_sats", "n_edges", "density", "mean_degree", "min_degree", "max_degree", "connected"):
        assert k in s

def test_determinism():
    wd = CN.WalkerDelta()
    assert np.array_equal(CN.contact_graph(wd, 0.0), CN.contact_graph(wd, 0.0))
    assert np.allclose(wd.positions(0.3), wd.positions(0.3))

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")