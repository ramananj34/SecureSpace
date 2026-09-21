from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import attribution as A
import weak_prng as W

K, WGT = 256, 8

def test_recover_score_endpoints():
    assert A.recover_score(W.strong, K, n_keys=300)["recover_score"] < 0.05
    assert A.recover_score(W.identity, K, n_keys=300)["recover_score"] > 0.95
    assert A.recover_score(W.truncated_keyspace(1), K, n_keys=300)["recover_score"] > 0.95

def test_recover_score_truncated_keyspace_graded():
    # fewer effective keys -> higher collision rate.
    r4 = A.recover_score(W.truncated_keyspace(4), K, n_keys=400)["recover_score"]
    r64 = A.recover_score(W.truncated_keyspace(64), K, n_keys=400)["recover_score"]
    r1024 = A.recover_score(W.truncated_keyspace(1 << 10), K, n_keys=400)["recover_score"]
    assert r4 > r64 > r1024 - 1e-9
    assert r4 > 0.8

def test_biased_local_are_recovery_low():
    # non-uniform but UNPREDICTABLE: fresh perm per key -> recover ~0 (pure route-ii).
    assert A.recover_score(W.biased_bit_fy(0.9), K, n_keys=300)["recover_score"] < 0.05
    assert A.recover_score(W.local_shuffle(8), K, n_keys=300)["recover_score"] < 0.05

def test_nonuniformity_score_endpoints():
    floor = A.nonuniformity_score(W.strong, K, WGT, n_perm=2000)
    assert floor["eps_v_above_floor"] < 0.05                    # strong ~ at floor
    assert A.nonuniformity_score(W.local_shuffle(2), K, WGT, n_perm=2000)["eps_v"] > 0.7
    assert A.nonuniformity_score(W.identity, K, WGT, n_perm=1000)["eps_v"] > 0.9

def test_dissociation_labels():
    assert A.attribute(W.strong, K, WGT, n_keys=300, n_perm=2000)["route_label"] == "neither"
    assert A.attribute(W.local_shuffle(8), K, WGT, n_keys=300, n_perm=2000)["route_label"] == "nonuniformity"
    assert A.attribute(W.biased_bit_fy(0.9), K, WGT, n_keys=300, n_perm=2000)["route_label"] in ("nonuniformity", "neither")
    assert A.attribute(W.identity, K, WGT, n_keys=300, n_perm=1000)["route_label"] == "both"
    lbl_trunc4 = A.attribute(W.truncated_keyspace(4), K, WGT, n_keys=400, n_perm=2000)["route_label"]
    assert lbl_trunc4 in ("recovery", "both")
    lbl_trunc1024 = A.attribute(W.truncated_keyspace(1 << 10), K, WGT, n_keys=400, n_perm=2000)["route_label"]
    assert lbl_trunc1024 in ("recovery", "neither")

def test_reduced_round_chacha_neither():
    lbl = A.attribute(W.reduced_round_chacha(2), K, WGT, n_keys=300, n_perm=2000)["route_label"]
    assert lbl == "neither"

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")