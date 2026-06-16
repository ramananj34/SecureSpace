from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_THIS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from smap_msl_dataset_api import SMAPMSLChannelDataset, working_channels
from smapmsl_data_pytorch_wrapper import make_train_val_loaders
from telemanom_lstm import TelemanomLSTM, TelemanomConfig, count_parameters 
from adv_train import train_adv

DELTA_LSB = 2 ** -7

def train_channel(chan, args, device):
    out_dir = Path(args.out_dir) / chan
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)
    cd = SMAPMSLChannelDataset(chan, split="train", data_dir=data_dir)
    n_features = cd._raw.shape[1]
    config = TelemanomConfig(input_dim=n_features, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, learning_rate=args.learning_rate)
    train_loader, val_loader = make_train_val_loaders(cd, sequence_length=config.sequence_length, n_predictions=config.n_predictions, batch_size=config.batch_size, validation_split=config.validation_split)
    model = TelemanomLSTM(config)
    eps_lsb = args.eps / DELTA_LSB
    alpha = (args.eps / 4.0) if args.alpha is None else args.alpha
    print(f"[{chan}] feat={n_features}  windows: train_batches={len(train_loader)} "
          f"val_batches={len(val_loader)}")
    print(f"[{chan}] eps={args.eps:.5f} ({eps_lsb:.1f} LSB)  alpha={alpha:.6f} (eps/4)  "
          f"T={args.steps}  monitor={args.monitor}  gen={args.gen_mode}  "
          f"params={count_parameters(model):,}")
    t0 = time.time()
    best_state, history = train_adv(model, train_loader, val_loader, config, device=device, eps=args.eps, alpha=args.alpha, steps=args.steps, monitor=args.monitor, clip_lo=args.clip_lo, clip_hi=args.clip_hi, random_start=not args.no_random_start, gen_mode=args.gen_mode, seed=args.seed, verbose=True)
    elapsed = time.time() - t0
    torch.save(best_state, out_dir / "model.pt")
    json.dump(asdict(config), open(out_dir / "config.json", "w"), indent=2)
    json.dump({"train_loss": history.train_loss, "val_loss": history.val_loss,
               "best_epoch": history.best_epoch, "best_val_loss": history.best_val_loss,
               "stopped_early": history.stopped_early, "wall_time_seconds": elapsed,
               "channel": chan, "seed": args.seed, "val_loss_kind": args.monitor},
              open(out_dir / "history.json", "w"), indent=2)
    json.dump({"method": "madry_pgd_linf_feat0", "eps": args.eps, "eps_lsb": eps_lsb, "alpha": alpha, "alpha_frac": (alpha / args.eps if args.eps else None), "steps": args.steps, "monitor": args.monitor, "random_start": not args.no_random_start, "gen_mode": args.gen_mode, "clip": [args.clip_lo, args.clip_hi], "delta_lsb": DELTA_LSB, "base_runs": "runs", "seed": args.seed}, open(out_dir / "advtrain_meta.json", "w"), indent=2)
    print(f"[{chan}] saved -> {out_dir}  ({elapsed:.0f}s, best epoch {history.best_epoch+1})\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=["D-1", "M-3"], help="channels to adv-train (default: the E9 fragile set)")
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--out-dir", default=str(_THIS / "runs_advtrain"))
    ap.add_argument("--eps", type=float, default=2 ** -3, help="L_inf budget on [-1,1] (default 2^-3 = 16 LSB; 2^-4 = 8 LSB)")
    ap.add_argument("--alpha", type=float, default=None, help="PGD step (default eps/4)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--monitor", choices=["adv", "clean"], default="adv", help="early-stopping metric: adversarial val loss (default) or clean")
    ap.add_argument("--gen-mode", choices=["eval", "train"], default="train")
    ap.add_argument("--no-random-start", action="store_true")
    ap.add_argument("--clip-lo", type=float, default=-1.0)
    ap.add_argument("--clip-hi", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    working = set(working_channels(data_dir=Path(args.data_dir)))
    print(f"device={device}  channels={args.channels}  out={args.out_dir}  "
          f"eps={args.eps} ({args.eps/DELTA_LSB:.1f} LSB)\n")
    for chan in args.channels:
        if chan not in working:
            print(f"[{chan}] WARNING: not in working corpus", file=sys.stderr)
        mp = Path(args.out_dir) / chan / "model.pt"
        if mp.exists() and not args.force:
            print(f"[{chan}] cached -> skip"); continue
        train_channel(chan, args, device)
    print("done -- adv-trained models in runs_advtrain/ (frozen runs/ untouched).")
    print("Next (Day 6): re-measure E9/E14 with --runs-dir pointing at runs_advtrain/.")

if __name__ == "__main__":
    main()