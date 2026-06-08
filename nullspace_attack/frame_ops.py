from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "ldpc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from ldpc_ops import LDPCCode  # noqa: E402

HEADER_BITS = 400
SLOT_BITS = 800
N_SLOTS = 40

def get_long_code():
    return LDPCCode.dvbs2_long_rate12()

def embed_window_in_frame(window_bits, slot=0, k=32400, header_bits=HEADER_BITS, slot_bits=SLOT_BITS):
    window_bits = np.asarray(window_bits, dtype=np.uint8).reshape(-1)
    if window_bits.size != slot_bits:
        raise ValueError(f"expected {slot_bits} window bits, got {window_bits.size}")
    msg = np.zeros(k, dtype=np.uint8)
    lo = header_bits + slot * slot_bits
    msg[lo:lo + slot_bits] = window_bits
    return msg

def frame_arms(window_bits, code=None, slot=0):
    code = code or get_long_code()
    k, n = code.k, code.n
    msg = embed_window_in_frame(window_bits, slot=slot, k=k)
    delta_cw = np.asarray(code.encode(msg)).astype(np.uint8).reshape(-1)
    synd_ns = np.asarray(code.syndrome(delta_cw)).astype(np.int64)
    synd_ns_w = int(synd_ns.sum())
    if synd_ns_w != 0:
        raise AssertionError(
            f"null-space arm broken: syndrome(encode(δ_info)) weight={synd_ns_w} "
            f"(expected 0). δ=encode(δ_info) must lie in N(H).")
    w_info = int(msg.sum())
    w_total = int(delta_cw.sum())
    w_parity = w_total - w_info
    e_naive = np.zeros(n, dtype=np.uint8)
    e_naive[:k] = msg
    synd_naive = np.asarray(code.syndrome(e_naive)).astype(np.int64)
    w_synd_naive = int(synd_naive.sum())
    return {"syndrome_nullspace": synd_ns_w, "syndrome_naive_weight": w_synd_naive, "delta_info_weight": w_info, "delta_parity_weight": w_parity, "delta_total_weight": w_total, "naive_flagged": bool(w_synd_naive > 0)}

def frame_analysis_over_stream(delta_info_bits, code=None, window=100, slot=0, b=8):
    code = code or get_long_code()
    delta_info_bits = np.asarray(delta_info_bits, dtype=np.uint8)
    T, bb = delta_info_bits.shape
    if bb != b:
        raise ValueError(f"expected last axis b={b}, got {bb}")
    per_frame = []
    for w0 in range(0, T - window + 1, window):
        wbits = delta_info_bits[w0:w0 + window].reshape(window * b)
        if int(wbits.sum()) == 0:
            continue
        rec = frame_arms(wbits, code=code, slot=slot)
        rec["window_start"] = int(w0)
        per_frame.append(rec)
    n_touched = len(per_frame)
    if n_touched == 0:
        agg = {"n_frames_touched": 0, "frac_frames_syndrome0": 1.0, "frac_frames_naive_flagged": 0.0, "delta_total_weight_med": 0, "delta_total_weight_min": 0, "delta_total_weight_max": 0, "delta_info_weight_med": 0, "syndrome_naive_weight_med": 0}
        return {"per_frame": per_frame, **agg}
    tot = np.array([r["delta_total_weight"] for r in per_frame])
    inf = np.array([r["delta_info_weight"] for r in per_frame])
    nsw = np.array([r["syndrome_naive_weight"] for r in per_frame])
    agg = {"n_frames_touched": n_touched, "frac_frames_syndrome0": float(np.mean([r["syndrome_nullspace"] == 0 for r in per_frame])), "frac_frames_naive_flagged": float(np.mean([r["naive_flagged"] for r in per_frame])), "delta_total_weight_med": int(np.median(tot)), "delta_total_weight_min": int(tot.min()), "delta_total_weight_max": int(tot.max()), "delta_info_weight_med": int(np.median(inf)), "syndrome_naive_weight_med": int(np.median(nsw))}
    return {"per_frame": per_frame, **agg}