from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from nullspace_attack_utils.frame_packing import FramePlan, pack_frame, unpack_frame
from keyed_permutation import permutation_for_frame, inverse_permutation

HEADER_BITS = 400
SLOT_BITS = 800
N_SLOTS = 40

def default_frame_plan(n_slots: int = N_SLOTS, window: int = 100, b: int = 8, header_bits: int = HEADER_BITS) -> FramePlan:
    return FramePlan(channels=list(range(n_slots)), window=window, b=b, header_bits=header_bits)

def encode_frame(plan: FramePlan, windows, quantizer, code, key: bytes, t: int, header_bits=None, mission_offset: int = 0) -> np.ndarray:
    if plan.k != code.k:
        raise ValueError(f"plan.k={plan.k} != code.k={code.k}")
    m_in = pack_frame(plan, windows, quantizer, header_bits=header_bits)
    perm = permutation_for_frame(key, t, plan.k, mission_offset=mission_offset)
    m = np.asarray(m_in)[perm].astype(np.int8)
    return np.asarray(code.encode(m)).astype(np.uint8)

def decode_frame(plan: FramePlan, codeword, quantizer, code, key: bytes, t: int, mission_offset: int = 0):
    if plan.k != code.k:
        raise ValueError(f"plan.k={plan.k} != code.k={code.k}")
    m_hat = np.asarray(code.info_bit_extract(codeword)).astype(np.int8)
    perm = permutation_for_frame(key, t, plan.k, mission_offset=mission_offset)
    m_in_rec = m_hat[inverse_permutation(perm)]
    return unpack_frame(plan, m_in_rec, quantizer)