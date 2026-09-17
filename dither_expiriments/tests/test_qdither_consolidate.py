from __future__ import annotations
import sys
import json
import tempfile
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import qdither_consolidate as QC

def _fake_channel(std, sigmas=(0.0, 0.25, 1.0), n_coords=100, with_det=True):
    def size_rec(nc):
        ps = []
        for i, s in enumerate(sigmas):
            base = 1.0 / nc
            ps.append({
                "sigma_lsb": s, "m1_baseline": 0.5, "m4_baseline": base,
                "marginal": {"m1_ratio": 1.05, "m1_lift_over_random": 0.03,
                             "m4_msb_loc": base * (2.0 if i == 0 else 1.0)},
                "analytic_sa": {"m1_ratio": 1.0, "m2_pos_acc": 1.0 / nc, "m4_msb_loc": base,
                                "telem_order_abs_spearman": 0.05},
                "ceiling": {"telem_order_abs_spearman": max(0.0, 0.6 - 0.5 * s), "n_coords": nc},
            })
        return {"layout": {"n_coords": nc}, "per_sigma": ps}
    rc = [{"added_rms_lsb": 0.08 * s, "noise_floor_db": 2.4 * s, "lsb_flip_rate": 0.08 * s,
           "msb_flip_rate": 0.0, "level_change_rate": 0.08 * s} for s in sigmas]
    im = {"by_sigma": [{"adjacency_mi_total": max(0.0, 0.5 - 0.5 * s),
                        "signal_mi_total": max(0.0, 2.0 - 1.5 * s),
                        "level_kl_mean": 0.01 + 0.5 * s,
                        "bitplane_kl_total": 0.02 + 0.4 * s} for s in sigmas]}
    cc = {"reconstruction_cost": rc, "info_measures": im}
    if with_det:
        cc["detection_cost"] = {"clean": {"f0_5": 1.0},
                                "by_sigma": [{"f0_5_mean": 1.0, "f0_5_min": 1.0,
                                              "any_label_ever_missed": False} for _ in sigmas]}
    return {"chan": "X", "spacecraft": "SMAP", "train_std": std, "sigmas_lsb": list(sigmas),
            "channel_curves": cc, "sizes": {"800": size_rec(n_coords)}}

def test_consolidate_shapes_and_monotone():
    rec = _fake_channel(0.5)
    c = QC.consolidate_channel("X", rec, cost_db_target=0.1)
    ps = c["per_sigma"]
    assert len(ps) == 3
    assert all(abs(r["attacker_m1_ratio_mean"] - 1.0) < 1e-9 for r in ps)
    adj = [r["adjacency_mi_total"] for r in ps]
    sig = [r["signal_mi_total"] for r in ps]
    kl = [r["level_kl_mean"] for r in ps]
    assert adj[0] > adj[-1] and sig[0] > sig[-1] and kl[0] < kl[-1]

def test_adv_t5_proxy_decreases():
    rec = _fake_channel(0.5)
    c = QC.consolidate_channel("X", rec)
    adv = c["adv_t5_curve"]
    assert adv[0]["adv_t5_proxy"] > adv[-1]["adv_t5_proxy"]
    assert adv[0]["coord_excess_over_chance"] > 0.5
    assert adv[-1]["coord_excess_over_chance"] < 0.5

def test_operating_point_cost_bound():
    rec = _fake_channel(0.5, sigmas=(0.0, 0.25, 1.0))
    c = QC.consolidate_channel("X", rec, cost_db_target=0.1)
    op = c["operating_point"]
    assert op["largest_sigma_meeting_cost_and_detection"] == 0.0
    c2 = QC.consolidate_channel("X", rec, cost_db_target=1.0)
    assert c2["operating_point"]["largest_sigma_meeting_cost_and_detection"] == 0.25

def test_operating_point_respects_detection_miss():
    rec = _fake_channel(0.5, sigmas=(0.0, 0.25, 1.0))
    rec["channel_curves"]["detection_cost"]["by_sigma"][-1]["any_label_ever_missed"] = True
    c = QC.consolidate_channel("X", rec, cost_db_target=100.0)
    assert c["operating_point"]["largest_sigma_meeting_cost_and_detection"] == 0.25

def test_pool_excludes_flat():
    recs = {"flat": _fake_channel(0.0), "real": _fake_channel(0.5)}
    cons = {k: QC.consolidate_channel(k, r) for k, r in recs.items()}
    pooled = QC.pool(cons)
    assert np.isfinite(pooled[0]["m4_over_chance_mean_nondegenerate"])
    assert abs(pooled[0]["m4_over_chance_mean_nondegenerate"] - 2.0) < 1e-6

def test_end_to_end_files(tmp_path=None):
    d = Path(tempfile.mkdtemp())
    for name, std in [("A", 0.5), ("B", 0.0)]:
        json.dump(_fake_channel(std), open(d / f"{name}.json", "w"))
    cons = {c: QC.consolidate_channel(c, json.load(open(d / f"{c}.json"))) for c in ["A", "B"]}
    pooled = QC.pool(cons)
    assert len(pooled) == 3 and "sigma_lsb" in pooled[0]

def test_pool_detection_excludes_degenerate():
    flat = _fake_channel(0.0)
    for row in flat["channel_curves"]["detection_cost"]["by_sigma"]:
        row["f0_5_mean"] = 0.0; row["any_label_ever_missed"] = True
    for r in flat["channel_curves"]["reconstruction_cost"]:
        r["noise_floor_db"] = 0.0
    real = _fake_channel(0.5)
    cons = {"flat": QC.consolidate_channel("flat", flat),
            "real": QC.consolidate_channel("real", real)}
    pooled = QC.pool(cons)
    assert pooled[0]["any_inscope_label_missed"] is False
    assert abs(pooled[0]["detection_f0_5_mean_nondegenerate"] - 1.0) < 1e-9
    assert pooled[-1]["noise_floor_db_mean_nondegenerate"] > pooled[0]["noise_floor_db_mean_nondegenerate"]

def test_clean_fail_channel_out_of_scope():
    good = _fake_channel(0.5)
    bad = _fake_channel(0.6)
    for row in bad["channel_curves"]["detection_cost"]["by_sigma"]:
        row["any_label_ever_missed"] = True
    cons = {"good": QC.consolidate_channel("good", good),
            "bad": QC.consolidate_channel("bad", bad)}
    assert cons["bad"]["operating_point"]["in_scope_for_detection"] is False
    assert cons["bad"]["operating_point"]["largest_sigma_meeting_cost_and_detection"] is None
    pooled = QC.pool(cons)
    assert pooled[0]["any_inscope_label_missed"] is False
    assert pooled[0]["n_inscope_detection_channels"] == 1

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")