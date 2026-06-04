import sys
import time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
from ldpc.dvb_s2_short import (build_dvbs2_h_short_rate12,encode_dvbs2_short_rate12, RATE_1_2_SHORT)

def main():
    print("Building short-frame H matrix...")
    H = build_dvbs2_h_short_rate12()
    print(f"  H shape = {H.shape}, nnz = {H.nnz}")
    assert H.shape == (9000, 16200)
    assert H.nnz == 48599
    rng = np.random.default_rng(7)
    print("\nEncoding 3 random codewords and verifying H·c = 0")
    for trial in range(3):
        info = rng.integers(0, 2, size=RATE_1_2_SHORT.k, dtype=np.int8)
        t0 = time.time()
        cw = encode_dvbs2_short_rate12(info)
        t_enc = time.time() - t0
        assert cw.shape == (RATE_1_2_SHORT.n,)
        assert np.array_equal(cw[: RATE_1_2_SHORT.k], info)
        t0 = time.time()
        syndrome = (H @ cw.astype(np.int64)) % 2
        t_check = time.time() - t0
        n_nonzero = int(np.count_nonzero(syndrome))
        weight = int(cw.sum())
        status = "PASS" if n_nonzero == 0 else f"✗ FAIL ({n_nonzero} nonzero)"
        print(f"Trial {trial}: cw_weight={weight}, syndrome_nonzero={n_nonzero}  {status}")
        print(f"encode: {t_enc:.3f}s, syndrome: {t_check:.3f}s")
        assert n_nonzero == 0
    print("\nAll-zero codeword sanity check")
    info = np.zeros(RATE_1_2_SHORT.k, dtype=np.int8)
    cw = encode_dvbs2_short_rate12(info)
    assert cw.sum() == 0
    syndrome = (H @ cw.astype(np.int64)) % 2
    assert syndrome.sum() == 0
    print("All-zero codeword has all-zero syndrome")
    print("\nShort-frame verification PASSED")

if __name__ == "__main__":
    main()