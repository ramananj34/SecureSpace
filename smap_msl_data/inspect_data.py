import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("smap_msl_data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
LABELS = DATA_DIR / "labeled_anomalies.csv"

def main():
    #1. Labeled anomalies
    df = pd.read_csv(LABELS)
    print(f"Labeled anomalies CSV: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Spacecraft breakdown:\n{df['spacecraft'].value_counts()}")
    print(f"Unique channels: {df['chan_id'].nunique()}")
    print()

    #2. Channel files
    train_files = sorted(TRAIN_DIR.glob("*.npy"))
    test_files = sorted(TEST_DIR.glob("*.npy"))
    print(f"Channels: train={len(train_files)}, test={len(test_files)}")

    #3. Inspect a sample channel
    sample_chan = "A-1"  #SMAP channel, well-known
    train = np.load(TRAIN_DIR / f"{sample_chan}.npy")
    test = np.load(TEST_DIR / f"{sample_chan}.npy")
    print(f"\nSample channel {sample_chan}:")
    print(f"train shape: {train.shape}  (timesteps, features)")
    print(f"test shape: {test.shape}")
    print(f"train range: [{train[:, 0].min():.3f}, {train[:, 0].max():.3f}]")
    print(f"test range: [{test[:, 0].min():.3f}, {test[:, 0].max():.3f}]")
    print(f"feature 0 is the telemetry value to predict;")
    print(f"features 1..{train.shape[1]-1} are encoded command information.")

    #4. Get this channel's anomaly intervals
    row = df[df["chan_id"] == sample_chan].iloc[0]
    print(f"\nAnomaly sequences for {sample_chan}: {row['anomaly_sequences']}")
    print(f" (these are [start, end] index ranges into the *test* array)")

    #5. Total timesteps across all channels
    total_train = sum(np.load(f).shape[0] for f in train_files)
    total_test = sum(np.load(f).shape[0] for f in test_files)
    print(f"\nTotal timesteps: train={total_train:,}  test={total_test:,}")
    print(f"Total: ~{(total_train+total_test)/1e3:.0f}K timesteps")

    #6. Plot the sample channel
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    ax[0].plot(train[:, 0], lw=0.5)
    ax[0].set_title(f"{sample_chan} — training data (feature 0 = telemetry)")
    ax[0].set_ylim(-1.1, 1.1)
    ax[1].plot(test[:, 0], lw=0.5)
    #overlay anomaly regions
    anomalies = eval(row["anomaly_sequences"])  # it's a string of a list
    for (s, e) in anomalies:
        ax[1].axvspan(s, e, color="red", alpha=0.3)
    ax[1].set_title(f"{sample_chan} — test data (anomalies in red)")
    ax[1].set_ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig("smap_msl_data/sample_channel.png", dpi=120)
    print("\nPlot saved to smap_msl_data/sample_channel.png")

if __name__ == "__main__":
    main()