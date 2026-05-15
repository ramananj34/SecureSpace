from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("smap_msl_data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
LABELS = DATA_DIR / "labeled_anomalies.csv"

#A channel is "usable" if its training telemetry has std >= this threshold. 0.01 in normalized units = 1% of the [-1, 1] range. Anything below this and the LSTM has nothing meaningful to learn about telemetry dynamics.
TRAIN_STD_THRESHOLD = 0.01

def classify_channel(chan_id: str) -> dict:
    train = np.load(TRAIN_DIR / f"{chan_id}.npy")
    test = np.load(TEST_DIR  / f"{chan_id}.npy")
    #Feature 0 is the telemetry value we predict
    tele_train = train[:, 0]
    tele_test  = test[:, 0]
    info = {
        "chan_id": chan_id,
        "n_train": train.shape[0],
        "n_test": test.shape[0],
        "n_features": train.shape[1],
        "train_min": float(tele_train.min()),
        "train_max": float(tele_train.max()),
        "train_mean": float(tele_train.mean()),
        "train_std": float(tele_train.std()),
        "test_min": float(tele_test.min()),
        "test_max": float(tele_test.max()),
        "test_std": float(tele_test.std()),
    }
    #Classify
    if info["train_std"] < TRAIN_STD_THRESHOLD:
        info["status"] = "flat_train"
    elif info["n_train"] < 1000:
        info["status"] = "too_short"
    else:
        info["status"] = "usable"
    return info


def main():
    df_anom = pd.read_csv(LABELS).drop_duplicates("chan_id")  #81 unique
    print(f"Processing {len(df_anom)} unique channels")
    rows = []
    for _, row in df_anom.iterrows():
        info = classify_channel(row["chan_id"])
        info["spacecraft"] = row["spacecraft"]
        rows.append(info)
    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values(["spacecraft", "chan_id"]).reset_index(drop=True)
    #Save
    out_path = DATA_DIR / "channel_manifest.csv"
    manifest.to_csv(out_path, index=False)
    print(f"\nSaved manifest: {out_path}")
    #Summary
    print("\n=== Summary by status ===")
    for status, grp in manifest.groupby("status"):
        print(f"  {status:12s} {len(grp):3d} channels  "
              f"(SMAP: {(grp['spacecraft']=='SMAP').sum()}, "
              f"MSL: {(grp['spacecraft']=='MSL').sum()})")
    print("\n=== First 10 flat_train channels (if any) ===")
    flat = manifest[manifest["status"] == "flat_train"]
    if len(flat) > 0:
        print(flat[["chan_id", "spacecraft", "train_std", "train_min", "train_max"]].head(10).to_string(index=False))
    else:
        print("  (none)")
    print("\n=== Usable channels: distribution of train_std ===")
    use = manifest[manifest["status"] == "usable"]
    print(f"  count: {len(use)}")
    print(f"  train_std percentiles: 5%={use['train_std'].quantile(0.05):.3f}  "
          f"50%={use['train_std'].quantile(0.50):.3f}  "
          f"95%={use['train_std'].quantile(0.95):.3f}")
    print(f"  n_train percentiles:   5%={use['n_train'].quantile(0.05):.0f}  "
          f"50%={use['n_train'].quantile(0.50):.0f}  "
          f"95%={use['n_train'].quantile(0.95):.0f}")

if __name__ == "__main__":
    main()


"""
RESULTS: 
Processing 81 unique channels

Saved manifest: smap_msl_data/channel_manifest.csv

=== Summary by status ===
  flat_train    24 channels  (SMAP: 17, MSL: 7)
  too_short      5 channels  (SMAP: 3, MSL: 2)
  usable        52 channels  (SMAP: 34, MSL: 18)

=== First 10 flat_train channels (if any) ===
chan_id spacecraft    train_std  train_min  train_max
    C-2        MSL 0.000000e+00  -1.000000  -1.000000
   D-14        MSL 0.000000e+00  -1.000000  -1.000000
    M-6        MSL 0.000000e+00  -1.000000  -1.000000
   P-10        MSL 1.241427e-03   0.985882   1.001129
   P-14        MSL 8.192435e-05   0.999000   1.000000
    S-2        MSL 0.000000e+00  -1.000000  -1.000000
    T-5        MSL 0.000000e+00  -1.000000  -1.000000
    A-1       SMAP 1.110223e-16   0.999000   0.999000
    A-5       SMAP 7.441729e-05  -1.000000  -0.999000
    B-1       SMAP 0.000000e+00  -1.000000  -1.000000

=== Usable channels: distribution of train_std ===
  count: 52
  train_std percentiles: 5%=0.040  50%=0.496  95%=0.810
  n_train percentiles:   5%=1526  50%=2835  95%=3088
"""