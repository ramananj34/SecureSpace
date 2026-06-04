from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "smap_msl_data"))
sys.path.insert(0, str(_THIS_DIR))
import pandas as pd
import torch
from pipeline import detect_anomalies_for_channel, evaluate_anomalies
from VENDOR_telemanom import VendoredConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(_PROJECT_ROOT / "smap_msl_data"))
    p.add_argument("--runs-dir", default=str(_PROJECT_ROOT / "runs"))
    p.add_argument("--device", default=None, help="'cuda' | 'cpu' | None=auto-detect")
    p.add_argument("--force", action="store_true", help="Re-evaluate even if per-channel cache exists.")
    return p.parse_args()

def find_trained_channels(runs_dir: Path) -> list[str]:
    return sorted(sub.name for sub in runs_dir.iterdir() if sub.is_dir() and (sub / "model.pt").exists())


def evaluate_one(chan_id: str, runs_dir: Path, data_dir: Path, vc: VendoredConfig, device: torch.device | None, force: bool) -> tuple[dict, bool]:
    chan_dir = runs_dir / chan_id
    cache_path = chan_dir / "day5_eval.json"
    if cache_path.exists() and not force:
        with open(cache_path) as f:
            return json.load(f), True
    t0 = time.time()
    try:
        result = detect_anomalies_for_channel(chan_id=chan_id, model_path=chan_dir / "model.pt", data_dir=data_dir, vendored_config=vc, device=device, verbose=False)
        metrics = evaluate_anomalies(result.predicted_sequences, result.labeled_sequences)
        eval_dict = {"chan_id": chan_id, "status": "ok", "predicted_sequences": [list(s) for s in result.predicted_sequences], "labeled_sequences": [list(s) for s in result.labeled_sequences], "normalized_error": float(result.normalized_error), "e_s_mean": float(result.smoothed_errors.mean()), "e_s_std": float(result.smoothed_errors.std()), "e_s_max": float(result.smoothed_errors.max()), "n_test_windows": int(result.n_test_windows), "metrics": metrics, "wall_time_s": time.time() - t0}
    except Exception as e:
        eval_dict = {"chan_id": chan_id, "status": "failed_exception", "exception_type": type(e).__name__, "exception_message": str(e), "traceback": traceback.format_exc(), "wall_time_s": time.time() - t0}
    with open(cache_path, "w") as f:
        json.dump(eval_dict, f, indent=2, default=str)
    return eval_dict, False


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    channels = find_trained_channels(runs_dir)
    if not channels:
        print(f"No trained channels in {runs_dir}", file=sys.stderr)
        return 1
    print(f"All evaluation run")
    print(f"Runs dir: {runs_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Device: {device}")
    print(f"Channels: {len(channels)} trained")
    print()
    vc = VendoredConfig()
    all_results = []
    counts = {"ok": 0, "failed_exception": 0, "cached": 0}
    t_total = time.time()
    for idx, ch in enumerate(channels, 1):
        result, cached = evaluate_one(ch, runs_dir, data_dir, vc, device, args.force)
        all_results.append(result)
        if cached:
            counts["cached"] += 1
        else:
            counts[result["status"]] = counts.get(result["status"], 0) + 1
        tag = "(cached)" if cached else ""
        if result["status"] == "ok":
            m = result["metrics"]
            print(f"[{idx:>3}/{len(channels)}] {ch:<6} {tag:<9} "
                  f"pred={m['n_predicted']:>2}  lab={m['n_labeled']:>2}  "
                  f"tp={m['tp']}  fp={m['fp']}  fn={m['fn']}  "
                  f"P={m['precision']:.2f}  R={m['recall']:.2f}  "
                  f"F0.5={m['f0_5']:.2f}  ({result['wall_time_s']:.1f}s)")
        else:
            print(f"[{idx:>3}/{len(channels)}] {ch:<6} {tag:<9} "
                  f"FAILED ({result['exception_type']}: "
                  f"{result['exception_message'][:60]})")
    elapsed = time.time() - t_total
    ok_results = [r for r in all_results if r["status"] == "ok"]
    tp = sum(r["metrics"]["tp"] for r in ok_results)
    fp = sum(r["metrics"]["fp"] for r in ok_results)
    fn = sum(r["metrics"]["fn"] for r in ok_results)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = (1.25 * p * r / (0.25 * p + r)) if (0.25 * p + r) > 0 else 0.0
    print()
    print(f"Aggregate ({elapsed:.1f}s total)")
    print(f"Successful: {len(ok_results)} / {len(channels)}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<20} {v:>3}")
    print()
    print(f"Total TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"Pooled precision: {p:.4f}")
    print(f"Pooled recall: {r:.4f}")
    print(f"Pooled F0.5: {f05:.4f}")
    print(f"(Paper Table 2: P=0.875  R=0.800  F0.5=0.71)")
    json_out = runs_dir / "all_eval_results.json"
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {json_out}")
    rows = []
    for rr in all_results:
        if rr["status"] == "ok":
            m = rr["metrics"]
            rows.append({"chan_id": rr["chan_id"], "status": "ok", "n_pred": m["n_predicted"], "n_lab": m["n_labeled"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "precision": round(m["precision"], 4), "recall": round(m["recall"], 4), "f0_5": round(m["f0_5"], 4), "norm_err": round(rr["normalized_error"], 4), "e_s_mean": round(rr["e_s_mean"], 4), "e_s_max": round(rr["e_s_max"], 4), "wall_time_s": round(rr["wall_time_s"], 1)})
        else:
            rows.append({"chan_id": rr["chan_id"], "status": rr["status"], "exception_type": rr.get("exception_type", ""), "exception_message": rr.get("exception_message", "")[:100]})
    csv_out = runs_dir / "all_eval_summary.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())