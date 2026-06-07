import sys
import time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
from ldpc.dvb_s2_ldpc import (build_dvbs2_h_long_rate12, RATE_1_2_LONG, encode_dvbs2_long_rate12)

def main():
    print("Building H matrix...")
    H = build_dvbs2_h_long_rate12()
    print(f"  H shape = {H.shape}, nnz = {H.nnz}")
    rng = np.random.default_rng(42)

    print("\nEncoding 3 random codewords and verifying H·c = 0")
    for trial in range(3):
        info = rng.integers(0, 2, size=RATE_1_2_LONG.k, dtype=np.int8)
        t0 = time.time()
        cw = encode_dvbs2_long_rate12(info)
        t_enc = time.time() - t0
        assert cw.shape == (RATE_1_2_LONG.n,)
        assert np.array_equal(cw[: RATE_1_2_LONG.k], info)
        t0 = time.time()
        syndrome = (H @ cw.astype(np.int64)) % 2
        t_check = time.time() - t0
        n_nonzero = int(np.count_nonzero(syndrome))
        weight = int(cw.sum())
        info_weight = int(info.sum())
        parity_weight = weight - info_weight
        status = "PASS" if n_nonzero == 0 else f"FAIL ({n_nonzero} nonzero syndrome bits)"
        print(f"  Trial {trial}: info_weight={info_weight}, "
              f"parity_weight={parity_weight}, cw_weight={weight}, "
              f"syndrome_nonzero={n_nonzero}  {status}")
        print(f"encode: {t_enc:.2f}s, syndrome: {t_check:.3f}s")
        assert n_nonzero == 0, "Codeword does not satisfy H·c = 0"
    print("\nAll-zero codeword sanity check")
    info = np.zeros(RATE_1_2_LONG.k, dtype=np.int8)
    cw = encode_dvbs2_long_rate12(info)
    assert cw.sum() == 0
    syndrome = (H @ cw.astype(np.int64)) % 2
    assert syndrome.sum() == 0
    print("All-zero codeword has all-zero syndrome ")
    print("\nUnit-vector test (info bit 0 = 1, rest = 0)")
    info = np.zeros(RATE_1_2_LONG.k, dtype=np.int8)
    info[0] = 1
    cw = encode_dvbs2_long_rate12(info)
    syndrome = (H @ cw.astype(np.int64)) % 2
    print(f"  cw weight: {cw.sum()}")
    print(f"  syndrome weight: {syndrome.sum()}")
    assert syndrome.sum() == 0
    print("  Unit-vector encoding satisfies H·c = 0")
    print("\nAll verification PASSED")

if __name__ == "__main__":
    main()