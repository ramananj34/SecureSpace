from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
for _p in [str(_ROOT), str(_THIS.parent), str(_ROOT / "amrcc"), str(_ROOT / "smap_msl_data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import grad_channel as GC

def test_gradient_quant_roundtrip_within_lsb():
    g = np.random.default_rng(0).normal(0, 2.5, 200)
    gq = GC.GradientQuantizer(b=8)
    bits, rng_tuple = gq.encode(g)
    g_rec = gq.decode(bits, rng_tuple)
    dlsb_native = (g.max() - g.min()) / 256
    assert np.abs(g - g_rec).max() <= dlsb_native / 2 + 1e-9

def test_gradient_quant_no_clip_on_large_values():
    g = np.array([-6.0, 5.0, 0.0, 3.3, -2.1] * 40)
    gq = GC.GradientQuantizer(b=8)
    g_rec = gq.decode(*gq.encode(g))
    assert abs(g_rec[0] - (-6.0)) < 0.05 and abs(g_rec[1] - 5.0) < 0.05

def test_kg_scale():
    assert GC.k_g(86890, 8) == 695120 and GC.k_g(200, 8) == 1600

def test_keyed_encode_decode_roundtrip():
    bits = (np.random.default_rng(1).random(1600) < 0.5).astype(np.uint8)
    key = GC.isl_key(0, link_id=3)
    tx = GC.keyed_encode_bits(bits, key, t=5)
    rec = GC.keyed_decode_bits(tx, key, t=5)
    assert np.array_equal(rec, bits)
    assert not np.array_equal(tx, bits)

def test_isl_keys_independent():
    assert GC.isl_key(0, 1) != GC.isl_key(0, 2)
    assert GC.isl_key(0, 5) == GC.isl_key(0, 5)

def test_poison_scatter_lemma1():
    kg = 1600; w = 50
    key = GC.isl_key(0, 0)
    rng = np.random.default_rng(2)
    hits = np.zeros(kg); weights = []
    for i in range(2000):
        di = GC.poison_delta_info(kg, w, np.random.default_rng([2, i]))
        v = GC.receiver_perturbation(di, key, t=i)
        hits += v; weights.append(int(v.sum()))
    hits /= 2000
    assert all(x == w for x in weights)
    assert abs(hits.mean() - w / kg) < 0.005

def test_transmit_clean_roundtrip():
    g = np.random.default_rng(3).normal(0, 1.5, 100)
    gq = GC.GradientQuantizer(b=8)
    key = GC.isl_key(0, 0)
    res = GC.transmit_gradient(g, gq, key, t=0, poison_w=0)
    dlsb_native = (g.max() - g.min()) / 256
    assert res["poison_weight_received"] == 0
    assert np.abs(res["g_recv"] - g).max() <= dlsb_native / 2 + 1e-9

def test_transmit_poisoned_lands_random_weight():
    g = np.random.default_rng(4).normal(0, 1.5, 100)
    gq = GC.GradientQuantizer(b=8)
    key = GC.isl_key(0, 0)
    res = GC.transmit_gradient(g, gq, key, t=0, poison_w=40, seed=1)
    assert res["poison_weight_received"] == 40
    clean = GC.transmit_gradient(g, gq, key, t=0, poison_w=0)["g_recv"]
    assert not np.allclose(res["g_recv"], clean)

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print(f"PASS {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(fns)} PASS")