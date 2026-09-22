from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc"), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from smap_msl_dataset_api import Quantizer
from keyed_permutation import permutation_for_frame, inverse_permutation, apply_perm, apply_inverse_perm
from sampling import selection_sample

@dataclass
class GradientQuantizer:
    b: int = 8
    def __post_init__(self):
        self.q = Quantizer(x_min=-1.0, x_max=1.0, b=self.b)

    def encode(self, g):
        g = np.asarray(g, np.float64)
        gmin, gmax = float(g.min()), float(g.max())
        if gmax - gmin < 1e-12:
            scaled = np.zeros_like(g)
        else:
            scaled = 2.0 * (g - gmin) / (gmax - gmin) - 1.0
        levels = self.q.quantize(scaled)
        bits = np.asarray(self.q.levels_to_bits(levels), np.uint8).reshape(-1)
        return bits, (gmin, gmax)

    def decode(self, bits, rng_tuple):
        gmin, gmax = rng_tuple
        levels = self.q.bits_to_levels(np.asarray(bits, np.uint8))
        scaled = np.asarray(self.q.dequantize(levels), np.float64)
        if gmax - gmin < 1e-12:
            return np.zeros_like(scaled)
        return gmin + 0.5 * (scaled + 1.0) * (gmax - gmin)

    @property
    def k_g_per_coord(self):
        return self.b


def k_g(d_grad, b=8):
    return int(d_grad) * int(b)

def isl_key(base_seed, link_id):
    return np.random.default_rng([int(base_seed), 0x15, int(link_id)]).bytes(32)

def keyed_encode_bits(bits, key, t=0):
    bits = np.asarray(bits, np.uint8).reshape(-1)
    perm = permutation_for_frame(key, int(t), bits.size)
    return apply_perm(bits, perm)

def keyed_decode_bits(perm_bits, key, t=0):
    perm_bits = np.asarray(perm_bits, np.uint8).reshape(-1)
    perm = permutation_for_frame(key, int(t), perm_bits.size)
    return apply_inverse_perm(perm_bits, perm)

def poison_delta_info(k_g_bits, w, rng):
    di = np.zeros(int(k_g_bits), np.uint8)
    if w > 0:
        di[selection_sample(int(k_g_bits), int(w), rng)] = 1
    return di

def receiver_perturbation(delta_info, key, t=0):
    di = np.asarray(delta_info, np.uint8).reshape(-1)
    perm = permutation_for_frame(key, int(t), di.size)
    return apply_inverse_perm(di, perm)

def transmit_gradient(g, gq: GradientQuantizer, key, t=0, poison_w=0, seed=0):
    bits, rng_tuple = gq.encode(g)
    kg = bits.size
    tx = keyed_encode_bits(bits, key, t)
    rng = np.random.default_rng([seed, 7])
    if poison_w > 0:
        di = poison_delta_info(kg, poison_w, rng)
        v = receiver_perturbation(di, key, t)
        m_hat = bits ^ v
        slot_w = int(v.sum())
    else:
        m_hat = keyed_decode_bits(tx, key, t)
        slot_w = 0
    g_recv = gq.decode(m_hat, rng_tuple)
    return {"g_recv": g_recv, "poison_weight_received": slot_w,
            "poison_weight_injected": int(poison_w), "k_g": int(kg)}