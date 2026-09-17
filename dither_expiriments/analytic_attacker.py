from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sa_search import SAConfig, anneal_multi

def bit_run_fitness_factory(m_obs):
    m_obs = np.asarray(m_obs, np.uint8).ravel()
    k = m_obs.size

    def fitness_full(perm):
        inv = np.empty(k, np.int64); inv[perm] = np.arange(k, dtype=np.int64)
        rec = m_obs[inv]
        return float(np.sum(rec[:-1] == rec[1:]))

    return fitness_full, k

def _telem_from_msg(bits_msg, quantizer, header_bits, slot_bits, k):
    q = quantizer; vals = []
    off = header_bits
    while off + slot_bits <= k:
        vals.append(np.asarray(q.dequantize(q.bits_to_levels(bits_msg[off:off + slot_bits])), np.float64))
        off += slot_bits
    return np.concatenate(vals) if vals else np.zeros(0)

def coord_autocorr_fitness_factory(m_obs, quantizer, header_bits, slot_bits):
    m_obs = np.asarray(m_obs, np.uint8).ravel()
    k = m_obs.size

    def fitness_full(perm):
        inv = np.empty(k, np.int64); inv[perm] = np.arange(k, dtype=np.int64)
        rec = m_obs[inv]
        x = _telem_from_msg(rec, quantizer, header_bits, slot_bits, k)
        if x.size < 2:
            return 0.0
        d = np.diff(x)
        return float(-np.sum(d * d))

    return fitness_full, k

def analytic_attack(m_obs, k, fitness="bit_run", sa_config=None,
                    quantizer=None, header_bits=0, slot_bits=None, init_perm=None):
    cfg = sa_config or SAConfig()
    if fitness == "bit_run":
        ff, kk = bit_run_fitness_factory(m_obs)
        delta = None
    elif fitness == "coord_autocorr":
        if quantizer is None or slot_bits is None:
            raise ValueError("coord_autocorr needs quantizer + slot_bits")
        ff, kk = coord_autocorr_fitness_factory(m_obs, quantizer, header_bits, slot_bits)
        delta = None
    else:
        raise ValueError(f"unknown fitness {fitness!r}")
    assert kk == k
    res = anneal_multi(k, ff, cfg, delta_swap=delta, init_perm=init_perm)
    return res