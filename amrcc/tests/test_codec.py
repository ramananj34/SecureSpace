from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_AMRCC = _THIS.parent
_ROOT = _AMRCC.parent
for _p in [str(_ROOT), str(_AMRCC), str(_ROOT / "smap_msl_data"), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "ldpc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from nullspace_attack_utils.ldpc_ops import LDPCCode
from nullspace_attack_utils.frame_packing import pack_frame, unpack_frame
from smap_msl_dataset_api import Quantizer
from keyed_permutation import permutation_for_frame, inverse_permutation
from codec import default_frame_plan, encode_frame, decode_frame

KEY = bytes(range(32))
KEY2 = bytes(range(31, -1, -1))

_CODE = None
def code():
    global _CODE
    if _CODE is None:
        _CODE = LDPCCode.dvbs2_long_rate12()
    return _CODE

def test_plan_matches_code():
    assert default_frame_plan().k == code().k == 32400

def test_message_roundtrip_and_codespace():
    c = code()
    rng = np.random.default_rng(0)
    m_in = rng.integers(0, 2, c.k, dtype=np.int8)
    for t in [0, 1, 7, 2 ** 20]:
        perm = permutation_for_frame(KEY, t, c.k)
        cw = np.asarray(c.encode((m_in[perm]).astype(np.int8)))
        assert c.is_codeword(cw)
        m_hat = np.asarray(c.info_bit_extract(cw)).astype(np.int8)
        m_rec = m_hat[inverse_permutation(perm)]
        assert np.array_equal(m_rec, m_in)

def test_window_roundtrip():
    c = code()
    q = Quantizer()
    plan = default_frame_plan()
    rng = np.random.default_rng(1)
    windows = rng.uniform(-1.0, 1.0, size=(plan.n_channels, plan.window))
    _, windows_ref = unpack_frame(plan, pack_frame(plan, windows, q), q)
    for t in [0, 3, 2 ** 20]:
        cw = encode_frame(plan, windows, q, c, KEY, t)
        header, windows_dec = decode_frame(plan, cw, q, c, KEY, t)
        assert header.size == plan.header_bits
        assert np.array_equal(windows_dec, windows_ref)

def test_wrong_key_or_counter_corrupts():
    c = code()
    q = Quantizer()
    plan = default_frame_plan()
    rng = np.random.default_rng(2)
    windows = rng.uniform(-1.0, 1.0, size=(plan.n_channels, plan.window))
    _, ref = unpack_frame(plan, pack_frame(plan, windows, q), q)
    cw = encode_frame(plan, windows, q, c, KEY, 5)
    _, w_badkey = decode_frame(plan, cw, q, c, KEY2, 5)
    _, w_badctr = decode_frame(plan, cw, q, c, KEY, 6)
    assert not np.array_equal(w_badkey, ref)
    assert not np.array_equal(w_badctr, ref)

def test_weight_preservation():
    c = code()
    rng = np.random.default_rng(3)
    m_in = rng.integers(0, 2, c.k, dtype=np.int8)
    perm = permutation_for_frame(KEY, 11, c.k)
    assert int(m_in[perm].sum()) == int(m_in.sum())

def test_receiver_perturbation_support_is_permuted():
    c = code()
    delta_info = np.zeros(c.k, dtype=np.int8)
    supp = np.arange(400, 400 + 175)
    delta_info[supp] = 1
    perm = permutation_for_frame(KEY, 5, c.k)
    v = delta_info[inverse_permutation(perm)]
    assert int(v.sum()) == int(delta_info.sum())
    assert set(np.where(v)[0].tolist()) == set(perm[supp].tolist())

if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items())
           if n.startswith("test_") and callable(v)]
    for fn in fns:
        t0 = time.time()
        fn()
        print(f"PASS  {fn.__name__:46s} {time.time() - t0:6.2f}s")
    print(f"\nall {len(fns)} codec tests passed")