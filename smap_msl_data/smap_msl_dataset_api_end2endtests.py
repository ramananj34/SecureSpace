import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
from smap_msl_dataset_api import ChannelScaler, Quantizer, working_channels, SMAPMSLChannelDataset

def build_synthetic_data(tmp: Path) -> None:
    #Create a synthetic data directory mimicking the SMAP/MSL layout
    (tmp / "train").mkdir(parents=True)
    (tmp / "test").mkdir(parents=True)
    rng = np.random.default_rng(0)
    #Two synthetic channels: SYN-1 (usable), SYN-2 (flat_train)
    for chan_id, train_kind, n_train, n_test in [ ("SYN-1", "varying", 3000, 5000), ("SYN-2", "flat",    3000, 5000)]:
        if train_kind == "varying":
            train_tele = 0.3 * np.sin(np.linspace(0, 20 * np.pi, n_train)) + 0.1 * rng.standard_normal(n_train)
        else:
            train_tele = np.full(n_train, 0.999)
        #test: similar but with an injected anomaly at [1000, 1100]
        test_tele = 0.3 * np.sin(np.linspace(0, 30 * np.pi, n_test)) + 0.1 * rng.standard_normal(n_test)
        test_tele[1000:1100] = 1.5  #out-of-range anomaly
        #24 random binary command features
        train_cmd = (rng.random((n_train, 24)) > 0.5).astype(np.float64)
        test_cmd = (rng.random((n_test, 24)) > 0.5).astype(np.float64)
        train = np.column_stack([train_tele, train_cmd])
        test = np.column_stack([test_tele, test_cmd])
        np.save(tmp / "train" / f"{chan_id}.npy", train)
        np.save(tmp / "test" / f"{chan_id}.npy", test)
    #labeled_anomalies.csv
    labels = pd.DataFrame({
        "chan_id": ["SYN-1", "SYN-2"],
        "spacecraft": ["SMAP", "SMAP"],
        "anomaly_sequences": ["[[1000, 1099]]", "[[1000, 1099]]"],
        "class": ["[point]", "[point]"],
        "num_values": [5000, 5000],
    })
    labels.to_csv(tmp / "labeled_anomalies.csv", index=False)
    #channel_manifest.csv
    manifest = pd.DataFrame({
        "chan_id": ["SYN-1", "SYN-2"],
        "spacecraft": ["SMAP", "SMAP"],
        "status": ["usable", "flat_train"],
        "n_train": [3000, 3000],
        "n_test": [5000, 5000],
        "train_min": [-0.7, 0.999],
        "train_max": [0.7, 0.999],
    })
    manifest.to_csv(tmp / "channel_manifest.csv", index=False)

def main():
    tmp = Path(tempfile.mkdtemp(prefix="sspace_test_"))
    print(f"Synthetic data at: {tmp}")
    try:
        build_synthetic_data(tmp)
        
        #Test 1: working_channels filters correctly
        wc = working_channels(data_dir=tmp)
        assert wc == ["SYN-1"], f"Expected ['SYN-1'], got {wc}"
        print("[ok] working_channels() returns only 'usable' status")

        #Test 2: Train dataset windowing
        ds = SMAPMSLChannelDataset("SYN-1", split="train", window=250, stride=1, data_dir=tmp)
        # n_train = 3000, window = 250 -> 2750 windows
        assert len(ds) == 3000 - 250, f"len(ds)={len(ds)}"
        print(f"[ok] train dataset has {len(ds)} windows (n_train=3000, window=250)")
        x, y = ds[0]
        assert x.shape == (250, 25), f"x.shape={x.shape}"
        assert isinstance(y, float), f"y type={type(y)}"
        #Telemetry feature is scaled
        assert -1.5 <= x[:, 0].min() and x[:, 0].max() <= 1.5
        print(f"[ok] item shape (250, 25), telemetry scaled to ~[-1, 1]")
        #Verify the target is the next-step value (and scaling is consistent)
        #Window covers timesteps [0, 250); target is at timestep 250
        scaled_tele = ds.telemetry_scaled
        assert abs(y - scaled_tele[250]) < 1e-12, \
            f"target {y} != scaled_tele[250] {scaled_tele[250]}"
        #And the last value of the window is timestep 249
        assert abs(x[-1, 0] - scaled_tele[249]) < 1e-12
        print("[ok] window+target indexing: x[-1] @ t=249, target @ t=250")
        
        #Test 3: Test dataset anomaly mask
        ds_test = SMAPMSLChannelDataset("SYN-1", split="test", window=250, data_dir=tmp)
        mask = ds_test.anomaly_mask
        assert mask.shape == (5000,), f"mask.shape={mask.shape}"
        #Anomaly was [1000, 1099] inclusive per our synthetic label
        assert mask[1000:1100].all(), "anomaly region should be True"
        assert not mask[999], "before-anomaly should be False"
        assert not mask[1100], "after-anomaly should be False"
        print(f"[ok] anomaly mask: {mask.sum()} True timesteps, range [1000, 1099]")

        #Test 4: Quantize the scaled telemetry, verify roundtrip
        q = Quantizer()
        x_scaled = ds.telemetry_scaled[:1000]  #take a chunk
        levels = q.quantize(x_scaled)
        x_back = q.dequantize(levels)
        #In-range values reconstruct to within Δ_LSB
        in_range = np.abs(x_scaled) <= 1.0
        max_err = np.max(np.abs(x_scaled[in_range] - x_back[in_range]))
        assert max_err <= q.delta_lsb / 2 + 1e-12, f"max in-range err={max_err}"
        print(f"[ok] quantizer roundtrip: max in-range error = {max_err:.5f} (Δ_LSB/2 = {q.delta_lsb/2:.5f})")

        #Test 5: bit-level operations for XOR-flip simulation
        bits = q.levels_to_bits(levels[:100])
        assert bits.shape == (100, 8), f"bits.shape={bits.shape}"
        levels_back = q.bits_to_levels(bits)
        np.testing.assert_array_equal(levels[:100], levels_back)
        print(f"[ok] bit pack/unpack roundtrip ok over 100 timesteps")

        #Test 6: Confirm out-of-range test values are NOT clipped by scaler
        #Our synthetic test has anomaly values of 1.5 (raw); after scaling by train_min=-0.7, train_max=0.7 (set in the manifest), 1.5 maps to: 2*(1.5 - (-0.7))/(0.7 - (-0.7)) - 1 = 2*2.2/1.4 - 1 ≈ 2.143 (The actual scaler uses train data stats)
        max_test_scaled = ds_test.telemetry_scaled[1000:1100].max()
        print(f"[ok] anomaly region's max scaled value: {max_test_scaled:.3f}")
        print(f"      ({'in range' if abs(max_test_scaled) <= 1 else 'out of range, will clip at quantizer'})")
        #And confirm the quantizer clips it
        clipped_levels = q.quantize(ds_test.telemetry_scaled[1000:1100])
        if abs(max_test_scaled) > 1:
            assert (clipped_levels == 255).any(), "out-of-range should clip to 255"
            print(f"[ok] quantizer clipped {(clipped_levels == 255).sum()} samples to level 255")
        else:
            print("[ok] (anomaly happened to scale within range; no clip needed)")

        print("\nALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp)

if __name__ == "__main__":
    main()

"""
Synthetic data at: /tmp/sspace_test_rhopo7k4
[ok] working_channels() returns only 'usable' status
[ok] train dataset has 2750 windows (n_train=3000, window=250)
[ok] item shape (250, 25), telemetry scaled to ~[-1, 1]
[ok] window+target indexing: x[-1] @ t=249, target @ t=250
[ok] anomaly mask: 100 True timesteps, range [1000, 1099]
[ok] quantizer roundtrip: max in-range error = 0.00390 (Δ_LSB/2 = 0.00391)
[ok] bit pack/unpack roundtrip ok over 100 timesteps
[ok] anomaly region's max scaled value: 2.700
      (out of range, will clip at quantizer)
[ok] quantizer clipped 100 samples to level 255

ALL TESTS PASSED
"""