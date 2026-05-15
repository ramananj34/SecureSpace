import sys
from pathlib import Path
import numpy as np
import pytest
from smap_msl_dataset_api import ChannelScaler, Quantizer

#Quantizer tests
class TestQuantizer:
    def test_b8_default_constants(self):
        #For b=8 on [-1, 1]: Δ_LSB = 2^-7, Δ_MSB = 1, c_quant ≈ 1.333
        q = Quantizer()
        assert q.n_levels == 256
        assert q.delta_lsb == pytest.approx(2 ** -7)
        assert q.delta_msb == pytest.approx(1.0)
        #c_quant = (4^8 - 1) / 3 * Δ_LSB^2 = 65535 / 3 * (2^-14) = 21845 / 16384 ≈ 1.3333
        assert q.c_quant() == pytest.approx(21845 / 16384, rel=1e-6)

    def test_roundtrip_in_range(self):
        #Q^-1(Q(x)) should be within Δ_LSB/2 of x for x in [-1, 1]
        q = Quantizer()
        np.random.seed(42)
        x = np.random.uniform(-1, 1, size=1000)
        levels = q.quantize(x)
        x_back = q.dequantize(levels)
        max_err = np.max(np.abs(x - x_back))
        #Mid-tread reconstruction: max error is Δ_LSB / 2
        assert max_err <= q.delta_lsb / 2 + 1e-12

    def test_clipping_above(self):
        #Values > x_max clip to the top level
        q = Quantizer()
        x = np.array([1.5, 2.0, 100.0])
        levels = q.quantize(x)
        assert np.all(levels == 255)

    def test_clipping_below(self):
        #Values < x_min clip to level 0
        q = Quantizer()
        x = np.array([-1.5, -2.0, -100.0])
        levels = q.quantize(x)
        assert np.all(levels == 0)

    def test_endpoints(self):
        #Boundary values should not crash and should be in range
        q = Quantizer()
        levels = q.quantize(np.array([q.x_min, q.x_max]))
        assert levels[0] == 0
        assert levels[1] == 255

    def test_levels_to_bits_lsb_first(self):
        #Bit ordering: level 5 = binary 00000101 -> LSB-first bits [1,0,1,0,0,0,0,0]
        q = Quantizer()
        # Single scalar level -> shape (b,)
        levels = np.array(5, dtype=np.uint8)
        bits = q.levels_to_bits(levels)
        assert bits.shape == (8,), f"got {bits.shape}"
        np.testing.assert_array_equal(bits, [1, 0, 1, 0, 0, 0, 0, 0])
        #Array of levels -> shape (n, b)
        levels = np.array([5], dtype=np.uint8)
        bits = q.levels_to_bits(levels)
        assert bits.shape == (1, 8), f"got {bits.shape}"
        np.testing.assert_array_equal(bits[0], [1, 0, 1, 0, 0, 0, 0, 0])

    def test_bits_to_levels_roundtrip(self):
        #evels_to_bits then bits_to_levels should be identity
        q = Quantizer()
        levels = np.array([[5, 100, 255], [0, 1, 254]], dtype=np.uint8)
        bits = q.levels_to_bits(levels)
        assert bits.shape == (2, 3, 8), f"got {bits.shape}"
        back = q.bits_to_levels(bits)
        np.testing.assert_array_equal(back, levels)
        #Also test flattened form (last axis = n_coords * b)
        bits_flat = bits.reshape(2, 24)
        back_flat = q.bits_to_levels(bits_flat)
        np.testing.assert_array_equal(back_flat, levels)

    def test_bits_to_levels_multidim(self):
        #Works on higher-dim level arrays
        q = Quantizer()
        np.random.seed(0)
        levels = np.random.randint(0, 256, size=(4, 10), dtype=np.uint8)
        bits = q.levels_to_bits(levels)
        assert bits.shape == (4, 10, 8), f"got {bits.shape}"
        back = q.bits_to_levels(bits)
        np.testing.assert_array_equal(back, levels)

    def test_msb_flip_subtracts_or_adds_msb_weight(self):
        #Critical for the signed receiver-side form in proposal Theorem 4: XOR-flipping a bit either adds OR subtracts the bit's weight, depending on the bit's current value. We verify this for the MSB
        q = Quantizer()
        #Pick a level with MSB = 0 and another with MSB = 1
        #Level 100 = 0b01100100, MSB (bit 7) = 0
        #Level 200 = 0b11001000, MSB (bit 7) = 1
        for level in [100, 200]:
            levels = np.array([level], dtype=np.uint8)
            bits = q.levels_to_bits(levels) #shape (1, 8)
            x = q.dequantize(levels)[0]
            #Flip MSB
            bits_flipped = bits.copy()
            bits_flipped[0, 7] ^= 1
            level_flipped = q.bits_to_levels(bits_flipped)[0]
            x_flipped = q.dequantize(np.array([level_flipped]))[0]
            delta = x_flipped - x
            #MSB weight is Δ_MSB = 2^{b-1} * Δ_LSB = 1 for b=8 on [-1,1]
            #If MSB was 0 -> flips to 1, value increases by Δ_MSB
            #If MSB was 1 -> flips to 0, value decreases by Δ_MSB
            if (level >> 7) & 1 == 0:
                assert delta == pytest.approx(q.delta_msb, abs=1e-9), \
                    f"level {level} MSB 0->1 should add Δ_MSB, got Δ={delta}"
            else:
                assert delta == pytest.approx(-q.delta_msb, abs=1e-9), \
                    f"level {level} MSB 1->0 should subtract Δ_MSB, got Δ={delta}"

#ChannelScaler tests
class TestChannelScaler:
    def test_basic_transform(self):
        #[train_min, train_max] -> [-1, 1]
        s = ChannelScaler(chan_id="X", train_min=0.0, train_max=10.0)
        assert s.transform(np.array([0.0]))[0] == pytest.approx(-1.0)
        assert s.transform(np.array([10.0]))[0] == pytest.approx(1.0)
        assert s.transform(np.array([5.0]))[0] == pytest.approx(0.0)

    def test_out_of_range_not_clipped(self):
        #Scaler doesn't clip, it just scales. Clipping happens at quantizer#
        s = ChannelScaler(chan_id="X", train_min=0.0, train_max=10.0)
        #value 15 is 50% beyond max -> scales to 2.0
        assert s.transform(np.array([15.0]))[0] == pytest.approx(2.0)
        #value -5 -> scales to -2.0
        assert s.transform(np.array([-5.0]))[0] == pytest.approx(-2.0)

    def test_inverse_transform_roundtrip(self):
        #Transform then inverse_transform is identity (within float precision)
        s = ChannelScaler(chan_id="X", train_min=-3.5, train_max=7.2)
        np.random.seed(1)
        x = np.random.uniform(-10, 20, size=100)
        y = s.transform(x)
        x_back = s.inverse_transform(y)
        np.testing.assert_allclose(x_back, x, atol=1e-10)

    def test_flat_channel_returns_zero(self):
        #Degenerate (flat_train) channels return zeros
        s = ChannelScaler(chan_id="X", train_min=0.5, train_max=0.5)
        out = s.transform(np.array([0.5, 1.0, 100.0]))
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""
(base) jovyan@jupyter-jramanan-colgate-edu---fe9dad03:~/work/SecureSpace$ PYTHONPATH=src pytest smap_msl_data/smap_msl_dataset_api_unit_tests.py -v
================================================================================= test session starts =================================================================================
platform linux -- Python 3.13.12, pytest-9.0.3, pluggy-1.6.0 -- /opt/conda/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/jovyan/work/SecureSpace
plugins: anyio-4.12.1
collected 13 items                                                                                                                                                                    

smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestQuantizer::test_b8_default_constants PASSED                                                                               [  7%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestQuantizer::test_roundtrip_in_range PASSED                                                                                 [ 15%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestQuantizer::test_clipping_above PASSED                                                                                     [ 23%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestQuantizer::test_clipping_below PASSED                                                                                     [ 30%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestQuantizer::test_endpoints PASSED                                                                                          [ 38%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestQuantizer::test_levels_to_bits_lsb_first PASSED                                                                           [ 46%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestQuantizer::test_bits_to_levels_roundtrip PASSED                                                                           [ 53%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestQuantizer::test_bits_to_levels_multidim PASSED                                                                            [ 61%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestQuantizer::test_msb_flip_subtracts_or_adds_msb_weight PASSED                                                              [ 69%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestChannelScaler::test_basic_transform PASSED                                                                                [ 76%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestChannelScaler::test_out_of_range_not_clipped PASSED                                                                       [ 84%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestChannelScaler::test_inverse_transform_roundtrip PASSED                                                                    [ 92%]
smap_msl_data/smap_msl_dataset_api_unit_tests.py::TestChannelScaler::test_flat_channel_returns_zero PASSED                                                                      [100%]

================================================================================= 13 passed in 0.91s ==================================================================================
(base) jovyan@jupyter-jramanan-colgate-edu---fe9dad03:~/work/SecureSpace$ 
"""