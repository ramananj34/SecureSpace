import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "smap_msl_data"))
sys.path.insert(0, str(_THIS_DIR))

from smap_msl_dataset_api import SMAPMSLChannelDataset
from telemanom_lstm import TelemanomLSTM, TelemanomConfig, count_parameters
from smapmsl_data_pytorch_wrapper import TelemanomTrainingDataset, make_train_val_loaders
from ltsm_trainer import train

def build_synthetic_channel(tmp: Path, chan_id: str = "SYN-1", n_train: int = 3500, n_test: int = 2000) -> None:
    (tmp / "train").mkdir(parents=True)
    (tmp / "test").mkdir(parents=True)
    rng = np.random.default_rng(0)

    def make_split(n):
        cmd = (rng.random((n, 24)) > 0.7).astype(np.float64)
        sin_component = 0.4 * np.sin(np.linspace(0, 20 * np.pi, n))
        cmd_lag = np.roll(cmd[:, 0], 1)  # depend on first command, one step lag
        tele = sin_component + 0.3 * cmd_lag + 0.05 * rng.standard_normal(n)
        return np.column_stack([tele, cmd])

    np.save(tmp / "train" / f"{chan_id}.npy", make_split(n_train))
    np.save(tmp / "test" / f"{chan_id}.npy", make_split(n_test))
    pd.DataFrame({"chan_id": [chan_id], "spacecraft": ["SMAP"], "anomaly_sequences": ["[[1000, 1099]]"], "class": ["[point]"], "num_values": [n_test]}).to_csv(tmp / "labeled_anomalies.csv", index=False)
    pd.DataFrame({"chan_id": [chan_id], "spacecraft": ["SMAP"], "status": ["usable"], "n_train": [n_train], "n_test": [n_test], "train_min": [-1.0], "train_max": [1.0]}).to_csv(tmp / "channel_manifest.csv", index=False)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    tmp = Path(tempfile.mkdtemp(prefix="amrcc_d3_"))
    print(f"Synthetic data: {tmp}")
    try:
        build_synthetic_channel(tmp)
        cd = SMAPMSLChannelDataset("SYN-1", split="train", data_dir=tmp)
        td = TelemanomTrainingDataset(cd, sequence_length=250, n_predictions=10)
        print(f"[ok] training windows: {len(td)}")
        x, y = td[0]
        assert x.shape == (250, 25), f"x.shape={x.shape}"
        assert y.shape == (10,), f"y.shape={y.shape}"
        print(f"[ok] item shapes correct: x={tuple(x.shape)}, y={tuple(y.shape)}")

        train_loader, val_loader = make_train_val_loaders(cd, sequence_length=250, n_predictions=10, batch_size=64, validation_split=0.2)
        print(f"[ok] loaders: train_batches={len(train_loader)}, val_batches={len(val_loader)}")

        cfg = TelemanomConfig(input_dim=25, hidden_size=80, num_layers=2, dropout=0.3, sequence_length=250, n_predictions=10, batch_size=64, epochs=5, patience=10, min_delta=0.0003, learning_rate=1e-3)
        model = TelemanomLSTM(cfg)
        print(f"[ok] model built: {count_parameters(model):,} params")
        print("Training (5 epochs)...")
        best_state, history = train(model, train_loader, val_loader, cfg, verbose=True)
        first_val = history.val_loss[0]
        last_val = history.val_loss[-1]
        best_val = history.best_val_loss
        print(f"\n[ok] First val_loss: {first_val:.6f}")
        print(f"[ok] Best val_loss:  {best_val:.6f} at epoch {history.best_epoch+1}")
        print(f"[ok] Final val_loss: {last_val:.6f}")
        assert best_val < first_val, f"Val loss did not improve: {first_val} -> {best_val}"
        improvement = (first_val - best_val) / first_val
        print(f"[ok] Improvement: {improvement*100:.1f}%")
        model_check = TelemanomLSTM(cfg)
        model_check.load_state_dict(best_state)
        print(f"[ok] Best state dict loads cleanly into a fresh model")
        cd_test = SMAPMSLChannelDataset("SYN-1", split="test", data_dir=tmp)
        td_test = TelemanomTrainingDataset(cd_test, sequence_length=250, n_predictions=10)
        x_test, y_test = td_test[100]
        model_check.eval()
        with torch.no_grad():
            pred = model_check(x_test.unsqueeze(0))
        mse = ((pred[0] - y_test) ** 2).mean().item()
        print(f"[ok] Test-set MSE on one sample: {mse:.6f}")

        print("\nAll Tests Passed")
    finally:
        shutil.rmtree(tmp)
    
if __name__ == "__main__":
    main()

"""
Synthetic data: /tmp/amrcc_d3_fohp5n83
[ok] training windows: 3241
[ok] item shapes correct: x=(250, 25), y=(10,)
[ok] loaders: train_batches=41, val_batches=11
[ok] model built: 86,890 params
Training (5 epochs)...
  epoch   1/5: train=0.154446  val=0.074504 *
  epoch   2/5: train=0.067309  val=0.060944 *
  epoch   3/5: train=0.061657  val=0.057951 *
  epoch   4/5: train=0.058526  val=0.056235 *
  epoch   5/5: train=0.057563  val=0.055035 *
  done: best val_loss = 0.055035 at epoch 5

[ok] First val_loss: 0.074504
[ok] Best val_loss:  0.055035 at epoch 5
[ok] Final val_loss: 0.055035
[ok] Improvement: 26.1%
[ok] Best state dict loads cleanly into a fresh model
[ok] Test-set MSE on one sample: 0.032711

All Tests Passed
"""