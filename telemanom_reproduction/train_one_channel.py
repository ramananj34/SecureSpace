"""
Train a telemanom LSTM on a single SMAP/MSL channel.
Usage:
    python train_one_channel.py M-1
    python train_one_channel.py M-1 --epochs 35 --batch-size 64
    python train_one_channel.py M-1 --data-dir smap_msl_data --out-dir runs
Outputs (to out-dir/<chan_id>/):
    model.pt        — best model state_dict
    config.json     — TelemanomConfig used
    history.json    — TrainHistory (per-epoch losses, best epoch)
    training_log.txt — stdout capture
Reproducible runs: set --seed to override the default 42.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "smap_msl_data"))
sys.path.insert(0, str(_THIS_DIR))

from smap_msl_dataset_api import SMAPMSLChannelDataset, working_channels
from telemanom_lstm import TelemanomLSTM, TelemanomConfig, count_parameters
from smapmsl_data_pytorch_wrapper import make_train_val_loaders
from ltsm_trainer import train

def parse_args():
    p = argparse.ArgumentParser(description="Train telemanom LSTM on one channel.")
    p.add_argument("chan_id", help="Channel ID, e.g., M-1, A-2.")
    p.add_argument("--data-dir", default="smap_msl_data", help="Root data directory.")
    p.add_argument("--out-dir", default="runs", help="Output directory for trained models.")
    p.add_argument("--epochs", type=int, default=35, help="Max epochs (early stopping may halt earlier).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    return p.parse_args()

def main():
    args = parse_args()
    #Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) / args.chan_id
    out_dir.mkdir(parents=True, exist_ok=True)
    #Confirm the channel is in the working corpus
    working = working_channels(data_dir=data_dir)
    if args.chan_id not in working:
        print(f"WARNING: {args.chan_id} is not in the working corpus.", file=sys.stderr)
        print(f"  Working channels: {len(working)} total", file=sys.stderr)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[setup] Device: {device}")
    print(f"[setup] Channel: {args.chan_id}")
    print(f"[setup] Data dir: {data_dir.resolve()}")
    print(f"[setup] Output dir: {out_dir.resolve()}")
    print(f"[setup] Seed: {args.seed}")
    
    #Build data pipeline
    cd = SMAPMSLChannelDataset(args.chan_id, split="train", data_dir=data_dir)
    print(f"[data] {args.chan_id} train: n_timesteps={cd.n_timesteps}")
    print(f"[data] {args.chan_id} train_min={cd.scaler.train_min:.4f}, train_max={cd.scaler.train_max:.4f}")
    n_features = cd._raw.shape[1]
    print(f"[data] {args.chan_id} input features: {n_features}")
    config = TelemanomConfig(input_dim=n_features, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, learning_rate=args.learning_rate)
    train_loader, val_loader = make_train_val_loaders(cd, sequence_length=config.sequence_length, n_predictions=config.n_predictions, batch_size=config.batch_size, validation_split=config.validation_split)
    print(f"[data] Windows: train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    #Build model
    model = TelemanomLSTM(config)
    n_params = count_parameters(model)
    print(f"[model] Parameters: {n_params:,}")

    #Train
    t0 = time.time()
    best_state, history = train(model, train_loader, val_loader, config, device=device, verbose=True)
    elapsed = time.time() - t0
    print(f"[train] Total wall time: {elapsed:.1f}s")

    #Save
    torch.save(best_state, out_dir / "model.pt")
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(
            {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
                "best_epoch": history.best_epoch,
                "best_val_loss": history.best_val_loss,
                "stopped_early": history.stopped_early,
                "wall_time_seconds": elapsed,
                "n_parameters": n_params,
                "channel": args.chan_id,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    print(f"[save] Wrote {out_dir/'model.pt'}")
    print(f"[save] Wrote {out_dir/'config.json'}")
    print(f"[save] Wrote {out_dir/'history.json'}")

if __name__ == "__main__":
    main()