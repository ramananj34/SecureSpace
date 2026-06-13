from __future__ import annotations
import numpy as np

__all__ = ["windows_of", "aggregate_windows", "collateral_msb_fraction"]

def windows_of(delta_info_bits, window: int = 100, b: int = 8):
    di = np.asarray(delta_info_bits, dtype=np.uint8)
    T = di.shape[0]
    for w0 in range(0, T - window + 1, window):
        wbits = di[w0:w0 + window].reshape(window * b)
        if int(wbits.sum()) == 0:
            continue
        yield w0, wbits

def aggregate_windows(per_window: list) -> dict:
    if not per_window:
        return {"n_windows_touched": 0}
    has = lambda key: key in per_window[0]
    col = lambda key: np.array([w[key] for w in per_window], dtype=float)
    agg = {"n_windows_touched": len(per_window), "frac_targetable": float(np.mean([w["targetable"] for w in per_window])), "frac_combined_passes_both": float(np.mean([w.get("combined_passes_both", False) for w in per_window])), "frac_footprint_exact": float(np.mean([w.get("footprint_exact", False) for w in per_window])), "frac_naive_flagged_by_combined": float(np.mean([w["naive_flagged_by_combined"] for w in per_window])), "window_weight_med": int(np.median(col("window_weight"))), "naive_Hprime_synd_weight_med": int(np.median(col("naive_Hprime_synd_weight")))}
    if has("info_weight_total"):
        agg["info_weight_total_med"] = int(np.median(col("info_weight_total")))
        agg["collateral_info_weight_med"] = int(np.median(col("collateral_info_weight")))
        agg["collateral_info_weight_min"] = int(np.min(col("collateral_info_weight")))
        agg["collateral_info_weight_max"] = int(np.max(col("collateral_info_weight")))
        agg["n_slots_touched_med"] = int(np.median(col("n_slots_touched")))
        agg["n_slots_touched_max"] = int(np.max(col("n_slots_touched")))
    return agg

def collateral_msb_fraction(v, header_bits: int, slot_bits: int, n_slots: int, target_slot: int, b: int = 8):
    v = np.asarray(v, dtype=np.uint8)
    lo = header_bits + target_slot * slot_bits
    coll = v.copy()
    coll[lo:lo + slot_bits] = 0
    total = int(coll.sum())
    if total == 0:
        return 0.0, 0
    nsamp = slot_bits // b
    msb = 0
    for s in range(n_slots):
        if s == target_slot:
            continue
        sl = coll[header_bits + s * slot_bits: header_bits + (s + 1) * slot_bits].reshape(nsamp, b)
        msb += int(sl[:, b - 1].sum())
    return msb / total, total