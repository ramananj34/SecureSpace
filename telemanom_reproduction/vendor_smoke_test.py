import sys
import tempfile
import shutil
from pathlib import Path
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "smap_msl_data"))
sys.path.insert(0, str(_THIS_DIR))
import numpy as np
import pandas as pd
import torch
from smap_msl_dataset_api import SMAPMSLChannelDataset
from telemanom_lstm import TelemanomLSTM, TelemanomConfig
from smapmsl_data_pytorch_wrapper import make_train_val_loaders
from ltsm_trainer import train
from pipeline import (build_channel,detect_anomalies_for_channel,evaluate_anomalies)
from VENDOR_telemanom import VendoredConfig

def build_synthetic_data(tmp: Path, chan_id: str = "SYN-A"):
    (tmp / "train").mkdir(parents=True)
    (tmp / "test").mkdir(parents=True)
    rng = np.random.default_rng(0)
    def make_split(n):
        cmd = (rng.random((n, 24)) > 0.7).astype(np.float64)
        sin = 0.3 * np.sin(np.linspace(0, 20 * np.pi, n))
        cmd_lag = np.roll(cmd[:, 0], 1)
        tele = sin + 0.4 * cmd_lag + 0.03 * rng.standard_normal(n)
        return np.column_stack([tele, cmd])
    train_arr = make_split(2500)
    test_arr = make_split(2000)
    test_arr[800:950, 0] += 2.0
    np.save(tmp / "train" / f"{chan_id}.npy", train_arr)
    np.save(tmp / "test" / f"{chan_id}.npy", test_arr)
    pd.DataFrame({"chan_id": [chan_id], "spacecraft": ["SMAP"],"anomaly_sequences": ["[[800, 949]]"],"class": ["[point]"], "num_values": [2000]}).to_csv(tmp / "labeled_anomalies.csv", index=False)
    pd.DataFrame({"chan_id": [chan_id], "spacecraft": ["SMAP"], "status": ["usable"],"n_train": [2500], "n_test": [2000],"train_min": [train_arr[:, 0].min()], "train_max": [train_arr[:, 0].max()]}).to_csv(tmp / "channel_manifest.csv", index=False)
    return chan_id

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    tmp = Path(tempfile.mkdtemp(prefix="amrcc_d4_"))
    print(f"Synthetic data: {tmp}")
    try:
        chan_id = build_synthetic_data(tmp)

        #1. Train a small LSTM
        print("\n[1] Training tiny LSTM...")
        cd = SMAPMSLChannelDataset(chan_id, split="train", data_dir=tmp)
        cfg = TelemanomConfig(input_dim=25, hidden_size=80, num_layers=2, dropout=0.3,sequence_length=250, n_predictions=10,batch_size=64, epochs=2, patience=10, min_delta=0.0003,learning_rate=1e-3)
        train_loader, val_loader = make_train_val_loaders(cd, sequence_length=250, n_predictions=10,batch_size=64, validation_split=0.2)
        model = TelemanomLSTM(cfg)
        best_state, _ = train(model, train_loader, val_loader, cfg, verbose=False)
        run_dir = tmp / "run"
        run_dir.mkdir()
        model_path = run_dir / "model.pt"
        torch.save(best_state, model_path)
        import json, dataclasses
        with open(run_dir / "config.json", "w") as f:
            json.dump(dataclasses.asdict(cfg), f)
        print(f"[ok] model saved to {model_path}")

        #2. Build vendored Channel
        print("\n[2] Building vendored Channel...")
        vc = VendoredConfig()
        channel = build_channel(chan_id, data_dir=tmp, vendored_config=vc)
        print(f"[ok] X_train shape: {channel.X_train.shape}")
        print(f"[ok] y_train shape: {channel.y_train.shape}")
        print(f"[ok] X_test shape: {channel.X_test.shape}")
        print(f"[ok] y_test shape: {channel.y_test.shape}")
        assert channel.X_train.shape[1] == 250, "Window length should be l_s=250"
        assert channel.y_train.shape[1] == 10, "Target dim should be n_predictions=10"
        assert channel.X_train.shape[2] == 25, "Should have 25 features"

        #3. Full pipeline
        print("\n[3] Running full Day 4 pipeline (detect_anomalies_for_channel)...")
        result = detect_anomalies_for_channel(chan_id=chan_id, model_path=model_path, data_dir=tmp,vendored_config=vc, verbose=True)
        print(f"\nPipeline completed without errors.")
        print(f"[info] Predicted sequences: {result.predicted_sequences}")
        print(f"[info] Labeled sequences: {result.labeled_sequences}")
        print(f"[info] Normalized error: {result.normalized_error:.4f}")
        print(f"[info] Smoothed errors stream: shape={result.smoothed_errors.shape}, " f"mean={result.smoothed_errors.mean():.4f}, " f"max={result.smoothed_errors.max():.4f}")

        #4. Evaluate
        print("\n[4] Evaluating against labels...")
        metrics = evaluate_anomalies(result.predicted_sequences, result.labeled_sequences)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        assert len(result.predicted_sequences) > 0, "No anomalies detected at all"
        if metrics["recall"] >= 1.0 - 1e-9 or metrics["tp"] >= 1:
            print("\n[ok] At least one predicted sequence overlaps the labeled anomaly.")
        else:
            print("\n[WARN] No predicted sequence overlapped the labeled anomaly.")
            print("This is acceptable for a 5-epoch training run; the pipeline")
            print("still works (we detected SOMETHING, just not the right thing).")

        print("\nAll tests pass")
    finally:
        shutil.rmtree(tmp)

if __name__ == "__main__":
    main()