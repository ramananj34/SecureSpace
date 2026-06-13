from __future__ import annotations
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

_STRIDE = 1 << 13  # 8192 blocks

def chacha20_keystream(key: bytes, t: int, n_bytes: int, mission_offset: int = 0) -> bytes:
    if len(key) != 32:
        raise ValueError(f"ChaCha20 key must be 32 bytes, got {len(key)}")
    iv_int = int(mission_offset) + int(t) * _STRIDE
    if iv_int < 0 or iv_int >= (1 << 128):
        raise ValueError("frame IV out of 128-bit range (t too large)")
    iv = iv_int.to_bytes(16, "little")
    enc = Cipher(algorithms.ChaCha20(key, iv), mode=None).encryptor()
    return enc.update(bytes(int(n_bytes)))

def fisher_yates(keystream: bytes, k: int) -> np.ndarray:
    r = np.frombuffer(keystream, dtype="<u8")
    if r.size < k - 1:
        raise ValueError(f"keystream too short: {r.size} u64 < {k - 1} needed")
    mods = np.arange(k, 1, -1, dtype=np.uint64)
    j_vals = (r[: k - 1] % mods).astype(np.int64)
    perm = np.arange(k, dtype=np.int64)
    for idx in range(k - 1):
        i = k - 1 - idx
        j = j_vals[idx]
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def permutation_for_frame(key: bytes, t: int, k: int, mission_offset: int = 0) -> np.ndarray:
    ks = chacha20_keystream(key, t, k * 8, mission_offset=mission_offset)
    return fisher_yates(ks, k)

def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size, dtype=perm.dtype)
    return inv

def apply_perm(vec: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return np.asarray(vec)[perm]

def apply_inverse_perm(vec: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return np.asarray(vec)[inverse_permutation(perm)]