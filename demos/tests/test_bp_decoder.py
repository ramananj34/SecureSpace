from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "ldpc"), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bp_decoder as BP

def _code_and_codeword(seed=0):
    code = BP.load_short_code()
    info = (np.random.default_rng(seed).random(code.k) < 0.5).astype(np.uint8)
    c = np.asarray(code.encode(info)).astype(np.uint8)
    assert code.is_codeword(c)
    return code, info, c

def test_short_code_loads():
    code = BP.load_short_code()
    assert code.n == 16200
    assert 0 < code.k < code.n
    assert code.H.shape == (code.n - code.k, code.n)
    assert code.m == code.n - code.k
    assert 0.4 < code.k / code.n < 0.6

def test_noiseless_decodes_to_itself():
    code, info, c = _code_and_codeword(0)
    llr = np.where(c == 1, -10.0, 10.0)
    chat, iters, ok = BP.bp_decode(code, llr, max_iter=50)
    assert ok and np.array_equal(chat, c)
    assert iters <= 1

def test_low_noise_corrects():
    code, info, c = _code_and_codeword(1)
    rng = np.random.default_rng(2)
    llr = BP.awgn_llr_seeded(c, sigma=0.5, rng=rng)
    chat, iters, ok = BP.bp_decode(code, llr, max_iter=50)
    assert ok
    assert np.array_equal(chat, c)

def test_syndrome_zero_on_success():
    code, info, c = _code_and_codeword(3)
    llr = BP.awgn_llr_seeded(c, sigma=0.6, rng=np.random.default_rng(4))
    chat, iters, ok = BP.bp_decode(code, llr, max_iter=50)
    if ok:
        assert np.all(code.syndrome(chat) == 0)

def test_high_noise_may_fail_gracefully():
    code, info, c = _code_and_codeword(5)
    llr = BP.awgn_llr_seeded(c, sigma=2.0, rng=np.random.default_rng(6))
    chat, iters, ok = BP.bp_decode(code, llr, max_iter=20)
    assert chat.shape == (code.n,)
    assert isinstance(ok, bool)

def test_recovers_info_bits_on_success():
    code, info, c = _code_and_codeword(7)
    llr = BP.awgn_llr_seeded(c, sigma=0.5, rng=np.random.default_rng(8))
    chat, iters, ok = BP.bp_decode(code, llr, max_iter=50)
    if ok:
        info_hat = code.info_bit_extract(chat)
        assert np.array_equal(info_hat, info)

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")