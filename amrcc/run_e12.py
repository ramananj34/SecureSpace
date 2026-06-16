from __future__ import annotations
import sys
import time
import argparse
import hmac
import hashlib
import statistics
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import (chacha20_keystream, fisher_yates, permutation_for_frame, apply_perm, apply_inverse_perm)

K = 32400
L_BITS = 64
KS_BYTES = K * (L_BITS // 8)

def _bench(fn, trials, warmup=3):
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); ms.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(ms), "mean_ms": statistics.fmean(ms), "std_ms": statistics.pstdev(ms), "min_ms": min(ms)}

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--ldpc-decode-ms", nargs="*", type=float, default=[5.0, 20.0, 50.0], help="ESTIMATE(s) of DVB-S2 n=64800 rate-1/2 BP decode time/frame " "(no decoder in repo -> E18; ratio scales with this)")
    ap.add_argument("--lstm", action="store_true", help="also time a TelemanomLSTM forward (needs torch)")
    args = ap.parse_args()
    key = b"\x01" * 32
    perm = permutation_for_frame(key, 0, K)
    ks_bytes = chacha20_keystream(key, 0, KS_BYTES)
    rng = np.random.default_rng(0)
    vec = rng.integers(0, 2, K, dtype=np.uint8)
    print(f"E12 cost benchmark -- per frame (k={K}, L={L_BITS} bits/idx, keystream={KS_BYTES} B)")
    print(f"trials={args.trials}\n")
    gen = _bench(lambda: permutation_for_frame(key, 0, K), args.trials)
    ks = _bench(lambda: chacha20_keystream(key, 0, KS_BYTES), args.trials)
    fy = _bench(lambda: fisher_yates(ks_bytes, K), max(20, args.trials // 4))
    apptx = _bench(lambda: apply_perm(vec, perm), args.trials)
    apprx = _bench(lambda: apply_inverse_perm(vec, perm), args.trials)
    defense_proto = gen["median_ms"] + apptx["median_ms"] + apprx["median_ms"]
    npfy = _bench(lambda: rng.permutation(K), args.trials)
    defense_opt = ks["median_ms"] + npfy["median_ms"] + apptx["median_ms"] + apprx["median_ms"]
    print("DEFENSE (keyed permutation, per frame)")
    print(f"  generate pi_t (ChaCha20+FY, prototype) : {gen['median_ms']:.3f} ms")
    print(f"    - ChaCha20 keystream                 : {ks['median_ms']:.3f} ms")
    print(f"    - Fisher-Yates (Python swap loop)    : {fy['median_ms']:.3f} ms   <-- prototype bottleneck")
    print(f"    - Fisher-Yates (native-C ref, np)    : {npfy['median_ms']:.3f} ms   (achievable w/ C swap loop)")
    print(f"  apply pi  (tx gather)                  : {apptx['median_ms']:.4f} ms")
    print(f"  apply pi^-1 (rx scatter+invert)        : {apprx['median_ms']:.4f} ms")
    print(f"  => defense/frame, prototype            : {defense_proto:.3f} ms")
    print(f"  => defense/frame, optimized (C FY)     : {defense_opt:.3f} ms\n")
    print(f"PRG BACKEND (produce {KS_BYTES} B keystream/frame)")
    for name, fn in [("ChaCha20", ks_chacha20), ("AES-256-CTR", ks_aes_ctr), ("HMAC-SHA256-CTR", ks_hmac_sha256)]:
        try:
            r = _bench(lambda f=fn: f(key), max(20, args.trials // 4))
            print(f"  {name:<16}: {r['median_ms']:.3f} ms")
        except Exception as e:
            print(f"  {name:<16}: unavailable ({type(e).__name__}: {e})")
    print()
    if args.lstm:
        try:
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
                for _ in range(50):
                    m(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                lstm_ms = (time.perf_counter() - t0) / 50 * 1e3
            print(f"LSTM forward (batch 64 x 250 x 25, {dev.type}): {lstm_ms:.3f} ms/batch "
                  f"({lstm_ms / 64 * 1e3:.1f} us/seq)\n")
        except Exception as e:
            print(f"LSTM ref unavailable ({type(e).__name__})\n")
    print("DEFENSE as % of LDPC BP decode   [LDPC = ESTIMATE, no decoder in repo -> E18]")
    print(f"  {'LDPC ms/frame':<16}{'prototype':>12}{'optimized':>14}")
    for ld in args.ldpc_decode_ms:
        print(f"  {ld:<16.1f}{defense_proto / ld * 100:>11.1f}%{defense_opt / ld * 100:>13.2f}%")
    print()
    print("ADVERSARIAL TRAINING (E11) -- for comparison")
    print("  marginal inference cost : 0 ms/frame (identical model forward; no per-frame op)")
    print("  offline training cost   : ~927 s (D-1) / ~513 s (M-3) GPU, PER CHANNEL, no transfer")
    print("  rescue                  : uncertain (L_inf != Hamming; D-1 re-measure this Day 6)")
    print(f"  vs keyed interleaving   : ~{defense_opt:.2f} ms/frame optimized, UNIVERSAL "
          "(all channels, no retrain)")
    print("\ndone -- defense cost is real; LDPC row is a flagged estimate (parameterise via --ldpc-decode-ms)")

if __name__ == "__main__":
    main()