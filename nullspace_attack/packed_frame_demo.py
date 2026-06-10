from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "ldpc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from frame_ops import get_long_code, HEADER_BITS, SLOT_BITS, N_SLOTS

def pack_frame(per_slot_windows, k, header=HEADER_BITS, slot_bits=SLOT_BITS):
    msg = np.zeros(k, dtype=np.uint8)
    for slot, w in enumerate(per_slot_windows):
        w = np.asarray(w, dtype=np.uint8).reshape(-1)
        lo = header + slot * slot_bits
        msg[lo:lo + slot_bits] = w
    return msg

def main():
    code = get_long_code(); k, n = code.k, code.n
    rng = np.random.default_rng(0)
    windows, per_slot_info_wt = [], []
    for s in range(N_SLOTS):
        w = np.zeros(SLOT_BITS, dtype=np.uint8)
        wt = int(rng.integers(12, 31))
        w[rng.choice(SLOT_BITS, size=wt, replace=False)] = 1
        windows.append(w); per_slot_info_wt.append(wt)
    msg = pack_frame(windows, k)
    w_info = int(msg.sum())
    cw = np.asarray(code.encode(msg)).astype(np.uint8)
    synd_ns = int(np.asarray(code.syndrome(cw)).sum())
    w_total = int(cw.sum()); w_parity = w_total - w_info
    e = np.zeros(n, dtype=np.uint8); e[:k] = msg
    synd_naive = int(np.asarray(code.syndrome(e)).sum())
    locerr = 0
    for s, wt in enumerate(per_slot_info_wt):
        lo = HEADER_BITS + s * SLOT_BITS
        if int(msg[lo:lo + SLOT_BITS].sum()) != wt:
            locerr += 1
    print(f"frame: n={n} k={k}  slots={N_SLOTS}x{SLOT_BITS}b + header {HEADER_BITS}b")
    print(f"filled {N_SLOTS}/40 channel slots, info weight per slot {min(per_slot_info_wt)}-{max(per_slot_info_wt)} bits")
    print(f"total info weight packed: {w_info} bits  (localization errors: {locerr})")
    print(f"\nnull-space arm: syndrome weight = {synd_ns}  -> {'PASSES' if synd_ns==0 else 'FAIL'}")
    print(f"naive arm:      syndrome weight = {synd_naive}  -> {'FLAGGED' if synd_naive>0 else 'passes(!)'}")
    print(f"\ncodeword weight wt(δ) = {w_total:,}  = info {w_info} + parity {w_parity:,}")
    print(f"parity / ((n-k)/2={int((n-k)/2):,}) = {w_parity/((n-k)/2):.2f}  (parity-dominated, frame-wide)")
    assert synd_ns == 0 and synd_naive > 0 and locerr == 0
    print("\nframe-level β2/β3 holds: 40-channel packed frame passes the syndrome as one")
    print("codeword; the naive variant is flagged; bypass cost is parity-pinned ~16,300.")

if __name__ == "__main__":
    main()