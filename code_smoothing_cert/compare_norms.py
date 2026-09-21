from __future__ import annotations
import numpy as np

COHEN_IMAGENET_ANCHOR = 1.3e-3

def our_hamming_fraction(r_cert_bits, k_region_bits):
    if k_region_bits <= 0:
        return float("nan")
    return float(r_cert_bits) / float(k_region_bits)

def cohen_l2_fraction(R, d_coords, delta_msb=1.0):
    scale = delta_msb * np.sqrt(max(d_coords, 1))
    return float(R) / float(scale) if scale > 0 else float("nan")

def l2_to_equivalent_msb_flips(R, delta_msb=1.0):
    return float(R) / float(delta_msb)


def heuristic_bridges_from_anchor(anchor, k, note=True):
    out = {
        "anchor_sigma_normalized_radius": float(anchor),
        "bridge_fraction_of_dimension_bits": float(anchor * k),
        "bridge_fraction_of_sqrt_dimension_bits": float(anchor * np.sqrt(k)),
        "bridges_differ_by_factor": float(np.sqrt(k)),
    }
    if note:
        out["note"] = ("HEURISTIC calibrations of Cohen's anchor into Hamming bits; they differ by "
                       "sqrt(k)~%.0fx -- the calibration is not derived (proposal E14 row / L11)."
                       % np.sqrt(k))
    return out

def compare_point(r_cert_bits, k_region_bits, cohen_R, d_coords, delta_msb=1.0):
    return {
        "ours_hamming_r_cert_bits": float(r_cert_bits),
        "ours_hamming_fraction": our_hamming_fraction(r_cert_bits, k_region_bits),
        "cohen_l2_radius_signal_units": float(cohen_R),
        "cohen_l2_fraction": cohen_l2_fraction(cohen_R, d_coords, delta_msb),
        "heuristic_l2_to_equiv_msb_flips": l2_to_equivalent_msb_flips(cohen_R, delta_msb),
        "incomparable_note": ("Hamming r_cert (bits) and Cohen L2 radius (signal units) are in different "
                              "spaces; fractions/bridges are heuristic context, not a common metric. "
                              "No single comparison scalar is reported (proposal L11)."),
    }

def comparison_table(rows):
    per_point = []
    for r in rows:
        cp = compare_point(r["r_cert_bits"], r["k_region_bits"], r["cohen_R"],
                           r["d_coords"], r.get("delta_msb", 1.0))
        cp["point_id"] = r.get("point_id")
        per_point.append(cp)
    def med(key):
        vals = [p[key] for p in per_point if np.isfinite(p[key])]
        return float(np.median(vals)) if vals else float("nan")
    agg = {
        "n_points": len(per_point),
        "median_ours_r_cert_bits": med("ours_hamming_r_cert_bits"),
        "median_ours_hamming_fraction": med("ours_hamming_fraction"),
        "median_cohen_l2_radius": med("cohen_l2_radius_signal_units"),
        "median_cohen_l2_fraction": med("cohen_l2_fraction"),
        "median_cohen_equiv_msb_flips": med("heuristic_l2_to_equiv_msb_flips"),
        "note": ("Native quantities aggregated SEPARATELY. Hamming and L2 medians are NOT combined into a "
                 "single ratio; the two certificates measure robustness in different norms (proposal L11)."),
    }
    return {"per_point": per_point, "aggregate": agg}