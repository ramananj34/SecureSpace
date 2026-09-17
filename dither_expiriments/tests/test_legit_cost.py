from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from smap_msl_dataset_api import Quantizer
from legit_cost import reconstruction_cost, detection_cost

Q = Quantizer()

def test_sigma0_zero_cost():
    x = np.random.default_rng(0).uniform(-0.9, 0.9, size=2000)
    r = reconstruction_cost(x, Q, [0.0], n_reps=2)[0]
    assert r["added_mse_vs_nodither"] == 0.0
    assert r["level_change_rate"] == 0.0
    assert abs(r["noise_floor_db"]) < 1e-9
    assert all(v == 0.0 for v in r["plane_flip_rate"])

def test_level_change_matches_prediction():
    x = np.random.default_rng(1).uniform(-0.9, 0.9, size=200_000)
    r = reconstruction_cost(x, Q, [0.10 * Q.delta_lsb], n_reps=4)[0]
    pred = r["level_change_rate_pred"]
    assert abs(r["level_change_rate"] - pred) / pred < 0.15
    assert abs(r["lsb_flip_rate"] - r["level_change_rate"]) / r["level_change_rate"] < 0.20
    assert r["msb_flip_rate"] < 0.05 * r["lsb_flip_rate"]

def test_plane_flip_decreasing():
    x = np.random.default_rng(2).uniform(-0.9, 0.9, size=100_000)
    pf = reconstruction_cost(x, Q, [0.25 * Q.delta_lsb], n_reps=4)[0]["plane_flip_rate"]
    assert pf[0] > pf[1] > pf[2] and pf[-1] < pf[0]

def test_noise_floor_db_positive_and_grows():
    x = np.random.default_rng(3).uniform(-0.9, 0.9, size=50_000)
    rs = reconstruction_cost(x, Q, [0.1 * Q.delta_lsb, 0.5 * Q.delta_lsb], n_reps=4)
    assert rs[0]["noise_floor_db"] > 0 and rs[1]["noise_floor_db"] > rs[0]["noise_floor_db"]

def test_detection_cost_mock_wiring():
    labels = [(100, 150)]
    fixed_E = [(100, 150)]
    def mock_detect(chan, model, tf, tele, cmds, cfg): return list(fixed_E)
    def mock_missed(E, l): return not any(not (e[1] < l[0] or l[1] < e[0]) for e in E)
    def mock_eval(pred, lab):
        tp = sum(1 for p in pred if any(not (p[1] < l[0] or l[1] < p[0]) for l in lab))
        return {"tp": tp, "fp": len(pred) - tp, "fn": len(lab) - tp,
                "f0_5": 1.0 if tp == len(lab) and len(pred) == tp else 0.0}
    x = np.random.default_rng(4).uniform(-0.5, 0.5, size=600)
    res = detection_cost("X", None, None, x, None, labels, None, Q,
                         [0.0, 0.25 * Q.delta_lsb], n_reps=3,
                         detect_fn=mock_detect, missed_fn=mock_missed, evaluate_fn=mock_eval)
    assert res["clean"]["f0_5"] == 1.0
    for row in res["by_sigma"]:
        assert row["f0_5_mean"] == 1.0
        assert row["per_label_miss_rate"] == [0.0]
        assert row["any_label_ever_missed"] is False

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")