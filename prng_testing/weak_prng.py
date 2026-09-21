from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import permutation_for_frame, fisher_yates  # frozen

np.seterr(over="ignore")
_MASK32 = np.uint32(0xFFFFFFFF)

def _fy_from_u64(r_u64, k):
    r = np.asarray(r_u64, dtype=np.uint64)
    if r.size < k - 1:
        raise ValueError(f"stream too short: {r.size} u64 < {k-1} needed")
    mods = np.arange(k, 1, -1, dtype=np.uint64)
    j_vals = (r[: k - 1] % mods).astype(np.int64)
    perm = np.arange(k, dtype=np.int64)
    for idx in range(k - 1):
        i = k - 1 - idx
        j = j_vals[idx]
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def _rotl32(x, n):
    return ((x << np.uint32(n)) | (x >> np.uint32(32 - n))) & _MASK32

def _qr(s, a, b, c, d):
    s[a] = (s[a] + s[b]) & _MASK32; s[d] = _rotl32(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & _MASK32; s[b] = _rotl32(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & _MASK32; s[d] = _rotl32(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & _MASK32; s[b] = _rotl32(s[b] ^ s[c], 7)

def _chacha_block(key32, counter, nonce12, rounds):
    consts = np.array([0x61707865, 0x3320646e, 0x79622d32, 0x6b206574], dtype=np.uint32)
    key = np.frombuffer(key32, dtype="<u4").astype(np.uint32)
    non = np.frombuffer(nonce12, dtype="<u4").astype(np.uint32)
    state = np.concatenate([consts, key, np.array([counter & 0xFFFFFFFF], np.uint32), non])
    w = state.copy()
    for _ in range(rounds // 2):
        _qr(w, 0, 4, 8, 12); _qr(w, 1, 5, 9, 13); _qr(w, 2, 6, 10, 14); _qr(w, 3, 7, 11, 15) 
        _qr(w, 0, 5, 10, 15); _qr(w, 1, 6, 11, 12); _qr(w, 2, 7, 8, 13); _qr(w, 3, 4, 9, 14) 
    return ((w + state) & _MASK32).tobytes()

def _chacha_stream_u64(key32, t, n_u64, rounds):
    nonce = (int(t) & ((1 << 96) - 1)).to_bytes(12, "little")
    need_blocks = (n_u64 * 8 + 63) // 64
    buf = b"".join(_chacha_block(key32, ctr, nonce, rounds) for ctr in range(need_blocks))
    return np.frombuffer(buf, dtype="<u8")[:n_u64].copy()

def strong(key, t, k):
    return permutation_for_frame(key, int(t), int(k))

def identity(key, t, k):
    return np.arange(int(k), dtype=np.int64)

def reduced_round_chacha(rounds):
    r = int(rounds)
    def gen(key, t, k):
        stream = _chacha_stream_u64(key, int(t), int(k), r)
        return _fy_from_u64(stream, int(k))
    gen.__name__ = f"chacha{r}r_fy"
    return gen

def lcg_fy():
    A = np.uint64(6364136223846793005); C = np.uint64(1442695040888963407)
    def gen(key, t, k):
        seed = int.from_bytes(key[:8], "little") ^ (int(t) * 0x9E3779B97F4A7C15)
        x = np.uint64(seed & ((1 << 64) - 1))
        out = np.empty(int(k), dtype=np.uint64)
        for i in range(int(k)):
            x = (A * x + C)
            out[i] = x
        return _fy_from_u64(out, int(k))
    gen.__name__ = "lcg_fy"
    return gen

def biased_bit_fy(p):
    pp = float(p)
    def gen(key, t, k):
        rng = np.random.default_rng([int.from_bytes(key[:8], "little") & ((1 << 63) - 1), int(t)])
        nbits = int(k) * 64
        bits = (rng.random(nbits) < pp).astype(np.uint64)
        words = bits.reshape(int(k), 64)
        weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))
        u = (words * weights).sum(axis=1, dtype=np.uint64)
        return _fy_from_u64(u, int(k))
    gen.__name__ = f"biased_bit_fy_p{pp:.3f}"
    return gen

def truncated_keyspace(n_keys):
    N = int(n_keys)
    def gen(key, t, k):
        eff = (int.from_bytes(key, "little") % N).to_bytes(32, "little")
        return permutation_for_frame(eff, int(t), int(k))
    gen.__name__ = f"trunc_keyspace_{N}"
    return gen

def local_shuffle(window):
    W = int(window)
    def gen(key, t, k):
        k = int(k)
        rng = np.random.default_rng([int.from_bytes(key[:8], "little") & ((1 << 63) - 1), int(t)])
        perm = np.arange(k, dtype=np.int64)
        w = max(1, min(W, k))
        for b0 in range(0, k, w):
            blk = perm[b0:b0 + w].copy(); rng.shuffle(blk); perm[b0:b0 + w] = blk
        return perm
    gen.__name__ = f"local_shuffle_w{W}"
    return gen

def ladder(k=None):
    """Return [(name, gen), ...]. Ordering is by intended weakness; the true x-axis is the MEASURED
    eps_perm (Day 3), not this list order."""
    L = [
        ("strong", strong),
        ("chacha20r", reduced_round_chacha(20)),
        ("chacha8r", reduced_round_chacha(8)),
        ("chacha4r", reduced_round_chacha(4)),
        ("chacha2r", reduced_round_chacha(2)),
        ("chacha1r", reduced_round_chacha(1)),
        ("lcg_fy", lcg_fy()),
        ("biased_p0.50", biased_bit_fy(0.50)),
        ("biased_p0.60", biased_bit_fy(0.60)),
        ("biased_p0.75", biased_bit_fy(0.75)),
        ("biased_p0.90", biased_bit_fy(0.90)),
        ("trunc_2^10", truncated_keyspace(1 << 10)),
        ("trunc_2^6", truncated_keyspace(1 << 6)),
        ("trunc_2^2", truncated_keyspace(1 << 2)),
        ("trunc_1", truncated_keyspace(1)),
        ("local_w64", local_shuffle(64)),
        ("local_w8", local_shuffle(8)),
        ("local_w2", local_shuffle(2)),
        ("identity", identity),
    ]
    return L

GENERATORS = {name: gen for name, gen in ladder()}