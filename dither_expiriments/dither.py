from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from nullspace_attack_utils.frame_packing import pack_frame

def sigma_in_lsb(sigma, quantizer):
    return float(sigma) / quantizer.delta_lsb

def sigma_from_lsb(sigma_lsb, quantizer):
    return float(sigma_lsb) * quantizer.delta_lsb

def add_dither(tele_scaled, sigma, rng):
    tele = np.asarray(tele_scaled, dtype=np.float64)
    if sigma is None or float(sigma) <= 0.0:
        return tele.copy()
    return tele + rng.normal(0.0, float(sigma), size=tele.shape)

def dithered_levels(tele_scaled, sigma, rng, quantizer):
    return quantizer.quantize(add_dither(tele_scaled, sigma, rng))

def dithered_bits(tele_scaled, sigma, rng, quantizer):
    lv = dithered_levels(tele_scaled, sigma, rng, quantizer)
    return np.asarray(quantizer.levels_to_bits(lv), dtype=np.uint8)

def dithered_reconstruction(tele_scaled, sigma, rng, quantizer):
    lv = dithered_levels(tele_scaled, sigma, rng, quantizer)
    return np.asarray(quantizer.dequantize(lv), dtype=np.float64)

def dithered_pack_frame(plan, windows, quantizer, sigma, rng, header_bits=None):
    w = add_dither(windows, sigma, rng)
    return pack_frame(plan, w, quantizer, header_bits=header_bits)