from __future__ import annotations
import numpy as np
import nullspace_attack.gf2 as gf2
from nullspace_attack.ldpc_ops import LDPCCode
__all__ = ["b_transpose_apply", "information_set_matrix", "is_information_set", "project_information_set", "normal_form_matrix", "normal_form_projection"]

def b_transpose_apply(code: LDPCCode, delta_prime: np.ndarray, P_packed: np.ndarray) -> np.ndarray:
    k = code.k
    dp = np.asarray(delta_prime).astype(np.uint8).ravel()
    info = dp[:k].copy()
    sup = np.nonzero(dp[k:])[0]
    if sup.size:
        acc = np.bitwise_xor.reduce(P_packed[sup], axis=0)
        ptp = gf2.unpack_bits(acc, k)
    else:
        ptp = np.zeros(k, dtype=np.uint8)
    return (info ^ ptp).astype(np.uint8)

def information_set_matrix(code: LDPCCode, S, P_packed: np.ndarray) -> np.ndarray:
    k = code.k
    S = list(S)
    assert len(S) == k, f"information set needs k={k} positions, got {len(S)}"
    G = np.zeros((k, k), dtype=np.uint8)
    for r, s in enumerate(S):
        if s < k:
            G[r, s] = 1
        else:
            G[r, :] = gf2.unpack_bits(P_packed[s - k], k)
    return gf2.pack_bits(G)

def is_information_set(code: LDPCCode, S, P_packed: np.ndarray) -> bool:
    return gf2.rank(information_set_matrix(code, S, P_packed), code.k) == code.k

def project_information_set(code: LDPCCode, delta_prime: np.ndarray, S, P_packed: np.ndarray) -> np.ndarray | None:
    k = code.k
    G_S = information_set_matrix(code, S, P_packed)
    target = np.asarray(delta_prime).ravel()[list(S)].astype(np.uint8)
    info = gf2.solve(G_S, target, k)
    if info is None:
        return None
    return code.encode(info.astype(np.int8))

def normal_form_matrix(code: LDPCCode, P_packed: np.ndarray, max_k: int = 12000) -> np.ndarray:
    k = code.k
    if k > max_k:
        raise NotImplementedError(
            f"normal_form_matrix on k={k} is prohibitive (O(k^2 m)); fallback-1 is "
            f"deprioritized (§6.13). Use project_systematic / info-set / low-weight library.")
    P = gf2.unpack_bits(P_packed, k).astype(np.float32)
    M = (P.T @ P).astype(np.int64) % 2
    np.fill_diagonal(M, (np.diagonal(M) + 1) % 2)
    return gf2.pack_bits(M.astype(np.uint8))

def normal_form_projection(code: LDPCCode, delta_prime: np.ndarray, M_packed: np.ndarray, P_packed: np.ndarray) -> np.ndarray | None:
    rhs = b_transpose_apply(code, delta_prime, P_packed)
    c = gf2.solve(M_packed, rhs, code.k)
    if c is None:
        return None
    return code.encode(c.astype(np.int8))