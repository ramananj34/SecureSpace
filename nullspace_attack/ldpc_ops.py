from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.sparse import csr_matrix
from nullspace_attack.gf2 import unpack_bits

__all__ = ["LDPCCode", "p_sys_column"]

_U1 = np.uint64(1)

@dataclass
class LDPCCode:
    name: str
    H: csr_matrix
    n: int
    k: int
    encode_fn: Callable[[np.ndarray], np.ndarray]
    
    @property
    def m(self) -> int:
        return self.n - self.k

    @classmethod
    def dvbs2_long_rate12(cls) -> "LDPCCode":
        from ldpc.dvb_s2_ldpc import (build_dvbs2_h_long_rate12, RATE_1_2_LONG, encode_dvbs2_long_rate12)
        p = RATE_1_2_LONG
        return cls("dvbs2-long-r1/2", build_dvbs2_h_long_rate12(), p.n, p.k, encode_dvbs2_long_rate12)

    @classmethod
    def dvbs2_short_rate12(cls) -> "LDPCCode":
        from ldpc.dvb_s2_short import (build_dvbs2_h_short_rate12, encode_dvbs2_short_rate12, RATE_1_2_SHORT)
        p = RATE_1_2_SHORT
        return cls("dvbs2-short-r1/2", build_dvbs2_h_short_rate12(), p.n, p.k, encode_dvbs2_short_rate12)

    def encode(self, info: np.ndarray) -> np.ndarray:
        return self.encode_fn(np.asarray(info))

    def info_bit_extract(self, codeword: np.ndarray) -> np.ndarray:
        return np.asarray(codeword)[:self.k]

    def parity_of(self, info: np.ndarray) -> np.ndarray:
        return self.encode(info)[self.k:]

    def apply_B(self, m: np.ndarray) -> np.ndarray:
        return self.encode(m)

    def nullspace_column(self, i: int) -> np.ndarray:
        e = np.zeros(self.k, dtype=np.int8)
        e[i] = 1
        return self.encode(e)

    def syndrome(self, word: np.ndarray) -> np.ndarray:
        return ((self.H @ np.asarray(word).astype(np.int64)) % 2).astype(np.uint8)

    def is_codeword(self, word: np.ndarray) -> bool:
        return not self.syndrome(word).any()

    def materialize_P_sys(self, packed: bool = True) -> np.ndarray:
        m, k = self.m, self.k
        nwords = (k + 63) // 64
        coo = self.H[:, :k].tocoo()
        H_info_packed = np.zeros((m, nwords), dtype=np.uint64)
        rows = coo.row.astype(np.intp)
        word_idx = (coo.col >> 6).astype(np.intp)
        bitpos = (coo.col & 63).astype(np.uint64)
        np.bitwise_or.at(H_info_packed, (rows, word_idx), _U1 << bitpos)
        P_packed = np.bitwise_xor.accumulate(H_info_packed, axis=0)
        return P_packed if packed else unpack_bits(P_packed, k)

    def project_systematic(self, delta_prime: np.ndarray) -> np.ndarray:
        return self.encode(self.info_bit_extract(delta_prime))

def p_sys_column(P_packed: np.ndarray, i: int) -> np.ndarray:
    return ((P_packed[:, i >> 6] >> np.uint64(i & 63)) & _U1).astype(np.uint8)