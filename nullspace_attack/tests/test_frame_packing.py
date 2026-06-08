import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "smap_msl_data"))
import numpy as np
from smap_msl_dataset_api import Quantizer
from nullspace_attack.frame_packing import FramePlan, pack_frame, unpack_frame, channel_slice

def _ok(name):
    print(f"  PASS  {name}")

def main():
    rng = np.random.default_rng(0)
    q = Quantizer(x_min=-1.0, x_max=1.0, b=8)
    plan = FramePlan(channels=[f"CH-{i}" for i in range(40)], window=100, b=8, header_bits=400)
    assert plan.n_channels == 40 and plan.slot_bits == 800 and plan.k == 32400, \
        f"layout wrong: n={plan.n_channels} slot={plan.slot_bits} k={plan.k}"
    _ok("layout: 40 x 800 + 400 header = k = 32,400 (matches long-frame LDPC)")
    windows = rng.uniform(-0.999, 0.999, size=(40, 100))
    header = rng.integers(0, 2, size=400, dtype=np.int8)
    m_in = pack_frame(plan, windows, q, header_bits=header)
    assert m_in.size == 32400 and set(np.unique(m_in)).issubset({0, 1})
    hdr_out, win_out = unpack_frame(plan, m_in, q)
    err = np.max(np.abs(win_out - windows))
    assert err <= 0.5 * q.delta_lsb + 1e-9, f"dequant error {err} > Δ_LSB/2 {0.5*q.delta_lsb}"
    _ok(f"pack->unpack roundtrip within Δ_LSB/2 (max err {err:.2e}, Δ_LSB/2 {0.5*q.delta_lsb:.2e})")
    assert np.array_equal(hdr_out, header), "header not preserved"
    _ok("header bits preserved exactly")
    c = 5
    s, e = channel_slice(plan, c)
    m2 = m_in.copy()
    m2[s:e] ^= 1
    _, win2 = unpack_frame(plan, m2, q)
    changed = np.where(np.any(np.abs(win2 - win_out) > 0, axis=1))[0]
    assert changed.tolist() == [c], f"slot flip changed channels {changed.tolist()}, expected [{c}]"
    _ok(f"channel_slice localization: perturbing slot {c} changes only channel {c}")
    s0, e0 = channel_slice(plan, 0)
    sN, eN = channel_slice(plan, plan.n_channels - 1)
    assert s0 == plan.header_bits and eN == plan.k, "slices do not tile [header, k)"
    assert all(channel_slice(plan, i)[1] == channel_slice(plan, i + 1)[0] for i in range(plan.n_channels - 1)), "slices not contiguous"
    _ok("channel slices tile [header_bits, k) contiguously")
    print("\nframe_packing tests PASSED.")

if __name__ == "__main__":
    main()