from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "smap_msl_data"))
sys.path.insert(0, str(_THIS_DIR))
import numpy as np
import pandas as pd
import torch
from smap_msl_dataset_api import SMAPMSLChannelDataset
from telemanom_lstm import TelemanomLSTM, TelemanomConfig, count_parameters
from smapmsl_data_pytorch_wrapper import make_train_val_loaders
from ltsm_trainer import train

EXCLUDED_CHANNELS = frozenset({"M-6"})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(_PROJECT_ROOT / "smap_msl_data"))
    p.add_argument("--out-dir",  default=str(_PROJECT_ROOT / "runs"))
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="'cuda' | 'cpu' | None=auto")
    p.add_argument("--resume-from", default=None, help="Skip channels in alphabetical order up to (and excluding) this one.")
    p.add_argument("--only-status", choices=["usable", "flat_train", "too_short"], default=None, help="Restrict to channels with this status field in channel_manifest.csv.")
    return p.parse_args()

def eligible_channels(manifest_path: Path, only_status: str | None = None) -> list[tuple[str, str, int]]:
    m = pd.read_csv(manifest_path)
    m = m[~m["chan_id"].isin(EXCLUDED_CHANNELS)]
    if only_status is not None:
        m = m[m["status"] == only_status]
    priority = {"usable": 0, "flat_train": 1, "too_short": 2}
    m = m.copy()
    m["priority"] = m["status"].map(priority).fillna(99).astype(int)
    m = m.sort_values(["priority", "chan_id"]).reset_index(drop=True)
    return list(zip(m["chan_id"], m["status"], m["n_train"].astype(int)))

def already_trained(chan_id: str, out_dir: Path) -> bool:
    return (out_dir / chan_id / "model.pt").exists()

def train_one(chan_id: str, args, device: torch.device, data_dir: Path, out_dir: Path) -> dict:
    chan_out = out_dir / chan_id
    chan_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cd = SMAPMSLChannelDataset(chan_id, split="train", data_dir=data_dir)
        n_features = cd._raw.shape[1]
        if cd.n_timesteps < 250 + 10 + 1:
            return {"chan_id": chan_id, "status": "skipped_too_short", "reason": f"n_timesteps={cd.n_timesteps} too small for window=250+lookahead=10", "wall_time_s": 0.0}
        config = TelemanomConfig(input_dim=n_features, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, learning_rate=args.learning_rate)
        train_loader, val_loader = make_train_val_loaders(cd, sequence_length=config.sequence_length, n_predictions=config.n_predictions, batch_size=config.batch_size, validation_split=config.validation_split)
        model = TelemanomLSTM(config)
        n_params = count_parameters(model)
        best_state, history = train(model, train_loader, val_loader,config, device=device, verbose=False)
        contains_nan = any(torch.isnan(v).any().item() for v in best_state.values()if torch.is_tensor(v) and v.is_floating_point())
        if contains_nan:
            return {"chan_id": chan_id, "status": "failed_nan", "wall_time_s": time.time() - t0, "n_features": n_features, "n_train_timesteps": cd.n_timesteps}
        torch.save(best_state, chan_out / "model.pt")
        with open(chan_out / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        with open(chan_out / "history.json", "w") as f:
            json.dump({"train_loss": history.train_loss, "val_loss": history.val_loss, "best_epoch": history.best_epoch, "best_val_loss": history.best_val_loss, "stopped_early": history.stopped_early, "wall_time_seconds": time.time() - t0, "n_parameters": n_params, "channel": chan_id, "seed": args.seed, "n_features": n_features }, f, indent=2)
        return {"chan_id": chan_id, "status": "ok", "wall_time_s": time.time() - t0, "best_epoch": history.best_epoch, "best_val_loss": history.best_val_loss, "n_train_windows": len(train_loader.dataset), "n_features": n_features}
    except Exception as e:
        return {"chan_id": chan_id, "status": "failed_exception", "exception_type": type(e).__name__, "exception_message": str(e), "traceback": traceback.format_exc(), "wall_time_s": time.time() - t0}

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "all_training_log.jsonl"
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest_path = data_dir / "channel_manifest.csv"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return 1
    channels = eligible_channels(manifest_path, only_status=args.only_status)
    if args.resume_from is not None:
        skip = True
        filtered = []
        for c, status, n_train in channels:
            if c == args.resume_from:
                skip = False
            if not skip:
                filtered.append((c, status, n_train))
        channels = filtered
    print(f"All training run")
    print(f"Data dir: {data_dir}")
    print(f"Out dir: {out_dir}")
    print(f"Device: {device}")
    print(f"Channels to consider: {len(channels)} "
          f"({sum(1 for _, s, _ in channels if s == 'usable')} usable, "
          f"{sum(1 for _, s, _ in channels if s == 'flat_train')} flat_train, "
          f"{sum(1 for _, s, _ in channels if s == 'too_short')} too_short)")
    print()
    counts = {"ok": 0, "skipped_existing": 0, "skipped_too_short": 0, "failed_nan": 0, "failed_exception": 0}
    t_total_start = time.time()
    for idx, (chan_id, status, n_train) in enumerate(channels, 1):
        print(f"[{idx:>3}/{len(channels)}] {chan_id:<6} (status={status}, n_train={n_train})  ",
              end="", flush=True)
        if already_trained(chan_id, out_dir):
            print("→ already trained, skipping")
            counts["skipped_existing"] += 1
            with open(log_path, "a") as f:
                f.write(json.dumps({"chan_id": chan_id, "status": "skipped_existing", "manifest_status": status, "n_train": n_train}) + "\n")
            continue
        result = train_one(chan_id, args, device, data_dir, out_dir)
        result["manifest_status"] = status
        result["n_train"] = n_train
        status_str = result["status"]
        counts[status_str] = counts.get(status_str, 0) + 1
        if status_str == "ok":
            print(f"-> ok  (best_val={result['best_val_loss']:.4f} @ epoch {result['best_epoch']+1}, "
                  f"wall={result['wall_time_s']:.1f}s)")
        elif status_str == "failed_exception":
            print(f"-> FAILED ({result['exception_type']}: {result['exception_message'][:80]})")
        elif status_str == "failed_nan":
            print(f"-> FAILED (model state contains NaN)")
        elif status_str == "skipped_too_short":
            print(f"-> skipped ({result['reason']})")
        else:
            print(f"-> {status_str}")
        with open(log_path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")
    elapsed = time.time() - t_total_start
    print()
    print(f"Training run complete in {elapsed/60:.1f} min")
    for k, v in sorted(counts.items()):
        print(f"  {k:<25} {v:>3}")
    print()
    print(f"Log: {log_path}")
    print(f"Trained models: {out_dir}/<chan_id>/model.pt")
    return 0

if __name__ == "__main__":
    sys.exit(main())