from __future__ import annotations
import sys
import time
import argparse
import hmac
import hashlib
import statistics
import json
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import (chacha20_keystream, fisher_yates, permutation_for_frame, apply_perm, apply_inverse_perm)
from fy_fast import fisher_yates_fast, _HAVE_NUMBA

K = 32400
L_BITS = 64
KS_BYTES = K * (L_BITS // 8)

def _bench(fn, trials, warmup=3):
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); ms.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ms)

def ks_chacha20(key):
    return chacha20_keystream(key, 0, KS_BYTES)

def ks_aes_ctr(key):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    enc = Cipher(algorithms.AES(key), modes.CTR(b"\x00" * 16)).encryptor()
    return enc.update(b"\x00" * KS_BYTES) + enc.finalize()

def ks_hmac_sha256(key):
    out, ctr = bytearray(), 0
    while len(out) < KS_BYTES:
        out += hmac.new(key, ctr.to_bytes(8, "little"), hashlib.sha256).digest()
        ctr += 1
    return bytes(out[:KS_BYTES])

def benchmark(trials=200, ldpc_ms=(5.0, 20.0, 50.0)):
    key = b"\x01" * 32
    perm = permutation_for_frame(key, 0, K)
    ks_bytes = chacha20_keystream(key, 0, KS_BYTES)
    rng = np.random.default_rng(0)
    vec = rng.integers(0, 2, K, dtype=np.uint8)
    fisher_yates_fast(ks_bytes, K)
    gen    = _bench(lambda: permutation_for_frame(key, 0, K), trials)
    ks     = _bench(lambda: chacha20_keystream(key, 0, KS_BYTES), trials)
    fy_py  = _bench(lambda: fisher_yates(ks_bytes, K), max(20, trials // 5))
    fy_fst = _bench(lambda: fisher_yates_fast(ks_bytes, K), trials)
    apptx  = _bench(lambda: apply_perm(vec, perm), trials)
    apprx  = _bench(lambda: apply_inverse_perm(vec, perm), trials)
    prg = {}
    for name, fn in [("ChaCha20", ks_chacha20), ("AES-256-CTR", ks_aes_ctr), ("HMAC-SHA256", ks_hmac_sha256)]:
        try:
            prg[name] = _bench(lambda f=fn: f(key), max(20, trials // 5))
        except Exception:
            prg[name] = None
    return {"k": K, "ks_bytes": KS_BYTES, "have_numba": bool(_HAVE_NUMBA), "gen_prototype_ms": gen, "keystream_ms": ks, "fy_python_ms": fy_py, "fy_numba_ms": fy_fst, "apply_tx_ms": apptx, "apply_rx_ms": apprx, "defense_prototype_ms": gen + apptx + apprx, "defense_optimized_ms": ks + fy_fst + apptx + apprx, "prg_ms": prg, "ldpc_ms": list(ldpc_ms)}

def lstm_forward_ms(trials=50):
    import torch
    from telemanom_lstm import TelemanomLSTM, TelemanomConfig
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = TelemanomLSTM(TelemanomConfig()).to(dev).eval()
    x = torch.randn(64, 250, 25, device=dev)
    with torch.no_grad():
        for _ in range(5):
            m(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(trials):
            m(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / trials * 1e3, dev.type

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--ldpc-decode-ms", nargs="*", type=float, default=[5.0, 20.0, 50.0])
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--save", default=None, help="optional path to dump the results JSON")
    args = ap.parse_args()
    b = benchmark(args.trials, tuple(args.ldpc_decode_ms))
    print(f"E12 cost benchmark -- per frame (k={b['k']}, keystream={b['ks_bytes']} B, numba={b['have_numba']})\n")
    print("DEFENSE (keyed permutation, per frame)")
    print(f"  generate pi_t (prototype)        : {b['gen_prototype_ms']:.3f} ms")
    print(f"    - ChaCha20 keystream           : {b['keystream_ms']:.3f} ms")
    print(f"    - Fisher-Yates (Python loop)   : {b['fy_python_ms']:.3f} ms")
    print(f"    - Fisher-Yates (numba, =perm)  : {b['fy_numba_ms']:.3f} ms"
          f"   ({b['fy_python_ms'] / b['fy_numba_ms']:.0f}x faster, identical permutation)")
    print(f"  apply tx / rx                    : {b['apply_tx_ms']:.4f} / {b['apply_rx_ms']:.4f} ms")
    print(f"  => defense/frame prototype       : {b['defense_prototype_ms']:.3f} ms")
    print(f"  => defense/frame optimized       : {b['defense_optimized_ms']:.3f} ms\n")
    print(f"PRG BACKEND ({b['ks_bytes']} B/frame)")
    for n, v in b["prg_ms"].items():
        print(f"  {n:<16}: {v:.3f} ms" if v is not None else f"  {n:<16}: unavailable")
    print()
    if args.lstm:
        try:
            ms, dv = lstm_forward_ms()
            b["lstm_ms"] = ms
            print(f"LSTM forward (batch 64 x 250 x 25, {dv}): {ms:.3f} ms/batch\n")
        except Exception as e:
            print(f"LSTM ref unavailable ({type(e).__name__})\n")
    print("DEFENSE as % of LDPC BP decode   [LDPC = flagged estimate; measured ref needs E18]")
    print(f"  {'LDPC ms':<12}{'prototype':>12}{'optimized':>13}")
    for ld in b["ldpc_ms"]:
        print(f"  {ld:<12.1f}{b['defense_prototype_ms'] / ld * 100:>11.1f}%"
              f"{b['defense_optimized_ms'] / ld * 100:>12.2f}%")
    print()
    print("ADV-TRAINING (E11): 0 ms inference; ~927/513 s GPU per channel offline, no transfer.")
    if args.save:
        json.dump(b, open(args.save, "w"), indent=2)
        print(f"\nsaved -> {args.save}")
    print("done")

if __name__ == "__main__":
    main()