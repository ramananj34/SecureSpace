from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from ldpc.dvb_s2_ldpc import DVBS2Params

RATE_1_2_SHORT = DVBS2Params(rate="1/2", frame="short", n=16200, k=7200)

C4_HIGH_DEGREE_ROWS = [
    [20,  712, 2386, 6354, 4061, 1062, 5045, 5158],
    [21, 2543, 5748, 4822, 2348, 3089, 6328, 5876],
    [22,  926, 5701,  269, 3693, 2438, 3190, 3507],
    [23, 2802, 4520, 3577, 5324, 1091, 4667, 4449],
    [24, 5140, 2003, 1263, 4742, 6497, 1185, 6202],
]

C4_LOW_DEGREE_ROWS = [
    [ 0, 4046, 6934],
    [ 1, 2855,   66],
    [ 2, 6694,  212],
    [ 3, 3439, 1158],
    [ 4, 3850, 4422],
    [ 5, 5924,  290],
    [ 6, 1467, 4049],
    [ 7, 7820, 2242],
    [ 8, 4606, 3080],
    [ 9, 4633, 7877],
    [10, 3884, 6868],
    [11, 8935, 4996],
    [12, 3028,  764],
    [13, 5988, 1057],
    [14, 7411, 3450],
]

def build_dvbs2_h_short_rate12() -> csr_matrix:
    p = RATE_1_2_SHORT
    n, k, M, q, n_minus_k = p.n, p.k, p.M, p.q, p.n_minus_k
    assert n_minus_k % M == 0
    all_rows = C4_HIGH_DEGREE_ROWS + C4_LOW_DEGREE_ROWS
    expected_rows = k // M
    assert len(all_rows) == expected_rows, f"Got {len(all_rows)} rows, expected {expected_rows}"
    H = lil_matrix((n_minus_k, n), dtype=np.int8)
    for m in range(k):
        row = m // M
        col = m % M
        for x in all_rows[row]:
            H[(x + col * q) % n_minus_k, m] = 1
    for j in range(n_minus_k):
        H[j, k + j] = 1
    for j in range(1, n_minus_k):
        H[j, k + j - 1] = 1
    return H.tocsr()

def encode_dvbs2_short_rate12(info_bits: np.ndarray) -> np.ndarray:
    p = RATE_1_2_SHORT
    n, k, M, q, n_minus_k = p.n, p.k, p.M, p.q, p.n_minus_k
    assert info_bits.shape == (k,)
    parity = np.zeros(n_minus_k, dtype=np.int8)
    all_rows = C4_HIGH_DEGREE_ROWS + C4_LOW_DEGREE_ROWS
    for m in range(k):
        if int(info_bits[m]) == 0:
            continue
        row = m // M
        col = m % M
        for x in all_rows[row]:
            parity[(x + col * q) % n_minus_k] ^= 1
    for j in range(1, n_minus_k):
        parity[j] ^= parity[j - 1]
    return np.concatenate([info_bits.astype(np.int8), parity])