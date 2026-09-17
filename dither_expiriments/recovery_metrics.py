from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "smap_msl_data"), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import inverse_permutation, apply_inverse_perm

def m1_random_baseline(k, w):
    k = int(k); w = int(w)
    if k <= 1:
        return 1.0
    return (w * (w - 1) + (k - w) * (k - w - 1)) / (k * (k - 1))

def m2_random_baseline(k):
    return 1.0 / float(k)

def m4_random_baseline(n_coords):
    return 1.0 / float(n_coords)

def _as_perm(pi_hat, k):
    p = np.asarray(pi_hat, dtype=np.int64).ravel()
    if p.size != k:
        raise ValueError(f"pi_hat size {p.size} != k {k}")
    return p

def observe(m_in, perm_true):
    return np.asarray(m_in, np.uint8).ravel()[np.asarray(perm_true, np.int64)]

def _coord_plane_layout(k, b, header_bits, slot_bits):
    coord = np.full(k, -1, np.int64)
    plane = np.full(k, -1, np.int64)
    off = header_bits
    n_coords = 0
    while off + slot_bits <= k:
        c = slot_bits // b
        for j in range(c):
            base = off + j * b
            coord[base:base + b] = n_coords
            plane[base:base + b] = np.arange(b)
            n_coords += 1
        off += slot_bits
    return coord, plane, n_coords

def m1_bit_accuracy(m_in, m_obs, pi_hat):
    m_in = np.asarray(m_in, np.uint8).ravel()
    k = m_in.size
    m_hat = apply_inverse_perm(np.asarray(m_obs, np.uint8).ravel(), _as_perm(pi_hat, k))
    return float(np.mean(m_hat == m_in))

def m2_position_accuracy(perm_true, pi_hat):
    pt = np.asarray(perm_true, np.int64).ravel()
    return float(np.mean(_as_perm(pi_hat, pt.size) == pt))

def _recovered_coord_of_position(perm_true, pi_hat, coord, k):
    pt = np.asarray(perm_true, np.int64).ravel()
    inv_hat = inverse_permutation(_as_perm(pi_hat, k))
    true_source = pt[inv_hat]
    return coord[true_source]

def m4_msb_localization(perm_true, pi_hat, k, b, header_bits, slot_bits):
    coord, plane, n_coords = _coord_plane_layout(k, b, header_bits, slot_bits)
    rec_coord = _recovered_coord_of_position(perm_true, pi_hat, coord, k)
    msb_mask = (plane == (b - 1)) & (coord >= 0)
    denom = int(msb_mask.sum())
    if denom == 0:
        return float("nan")
    hit = (rec_coord[msb_mask] == coord[msb_mask])
    return float(hit.sum() / denom)

def m3_telemetry_mse_ratio(m_in, m_obs, pi_hat, quantizer, header_bits, slot_bits, n_random=32, seed=0):
    q = quantizer
    m_in = np.asarray(m_in, np.uint8).ravel(); k = m_in.size
    m_obs = np.asarray(m_obs, np.uint8).ravel()

    def _telem(bits_msg):
        vals = []
        off = header_bits
        while off + slot_bits <= k:
            seg = bits_msg[off:off + slot_bits]
            vals.append(np.asarray(q.dequantize(q.bits_to_levels(seg)), np.float64))
            off += slot_bits
        return np.concatenate(vals) if vals else np.zeros(0)

    truth = _telem(m_in)
    hat = _telem(apply_inverse_perm(m_obs, _as_perm(pi_hat, k)))
    mse_hat = float(np.mean((hat - truth) ** 2)) if truth.size else 0.0
    rng = np.random.default_rng(seed)
    rand_mses = []
    for _ in range(n_random):
        rp = rng.permutation(k)
        rand_mses.append(np.mean((_telem(apply_inverse_perm(m_obs, rp)) - truth) ** 2))
    mse_rand = float(np.mean(rand_mses)) if truth.size else 1.0
    return {"mse_hat": mse_hat, "mse_random_mean": mse_rand, "ratio": (mse_hat / mse_rand) if mse_rand > 0 else float("nan")}

def _rankdata(a):
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0 + 1.0
    return avg[inv]

def spearman(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0

def telemetry_order_recovery(perm_true, pi_hat, m_obs, quantizer, header_bits, slot_bits):
    q = quantizer
    m_obs = np.asarray(m_obs, np.uint8).ravel(); k = m_obs.size

    def _telem(bits_msg):
        vals = []
        off = header_bits
        while off + slot_bits <= k:
            vals.append(np.asarray(q.dequantize(q.bits_to_levels(bits_msg[off:off + slot_bits])), np.float64))
            off += slot_bits
        return np.concatenate(vals) if vals else np.zeros(0)

    truth = _telem(apply_inverse_perm(m_obs, np.asarray(perm_true, np.int64)))
    hat = _telem(apply_inverse_perm(m_obs, _as_perm(pi_hat, k)))
    if truth.size < 3:
        return {"spearman": float("nan"), "abs_spearman": float("nan")}
    s = spearman(truth, hat)
    return {"spearman": s, "abs_spearman": abs(s)}

def score_candidate(m_in, perm_true, pi_hat, quantizer=None, b=8, header_bits=0, slot_bits=None,
                    include_m3=False, m3_random=32, seed=0):
    m_in = np.asarray(m_in, np.uint8).ravel()
    k = m_in.size
    w = int(m_in.sum())
    m_obs = observe(m_in, perm_true)
    m1 = m1_bit_accuracy(m_in, m_obs, pi_hat)
    m1_base = m1_random_baseline(k, w)
    m2 = m2_position_accuracy(perm_true, pi_hat)
    near_degenerate = (w <= 1) or (w >= k - 1) or (m1_base >= 0.95)
    rec = {
        "k": k, "w": w, "one_frac": w / k,
        "m1_bit_acc": m1, "m1_baseline": m1_base,
        "m1_ratio": (m1 / m1_base) if m1_base > 0 else float("nan"),
        "m1_gate_pass": bool(m1_base > 0 and (m1 / m1_base) < 1.05),
        "m2_pos_acc": m2, "m2_baseline": m2_random_baseline(k),
        "near_degenerate": bool(near_degenerate),
    }
    if slot_bits is not None:
        coord, plane, n_coords = _coord_plane_layout(k, b, header_bits, slot_bits)
        rec["n_coords"] = n_coords
        rec["m4_msb_loc"] = m4_msb_localization(perm_true, pi_hat, k, b, header_bits, slot_bits)
        rec["m4_baseline"] = m4_random_baseline(n_coords) if n_coords > 0 else float("nan")
        if quantizer is not None:
            rec["telem_order"] = telemetry_order_recovery(perm_true, pi_hat, m_obs, quantizer, header_bits, slot_bits)
            if include_m3:
                rec["m3"] = m3_telemetry_mse_ratio(m_in, m_obs, pi_hat, quantizer, header_bits, slot_bits, n_random=m3_random, seed=seed)
    return rec