import sys
import json
import time
from dataclasses import asdict
from pathlib import Path
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "smap_msl_data"))
sys.path.insert(0, str(_THIS_DIR))
import numpy as np
import pandas as pd
from pipeline import detect_anomalies_for_channel, evaluate_anomalies
from VENDOR_telemanom import VendoredConfig
RUNS_DIR = _PROJECT_ROOT / "runs"
DATA_DIR = _PROJECT_ROOT / "smap_msl_data"
OUTPUT_DIR = RUNS_DIR

def find_trained_channels(runs_dir: Path) -> list[str]:
    channels = []
    for sub in sorted(runs_dir.iterdir()):
        if sub.is_dir() and (sub / "model.pt").exists():
            channels.append(sub.name)
    return channels

def main():
    print(f"Full Run")
    print(f"Runs directory: {RUNS_DIR}")
    print(f"Data directory: {DATA_DIR}")
    channels = find_trained_channels(RUNS_DIR)
    if not channels:
        print(f"No trained channels found in {RUNS_DIR}. Aborting.")
        return
    print(f"Found {len(channels)} trained channels: {channels}\n")
    vc = VendoredConfig()
    all_results = []
    summary_rows = []

    for ch in channels:
        print(f"- {ch} -")
        model_path = RUNS_DIR / ch / "model.pt"
        t0 = time.time()
        try:
            result = detect_anomalies_for_channel(chan_id=ch, model_path=model_path, data_dir=DATA_DIR, vendored_config=vc, verbose=False)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            summary_rows.append({"chan_id": ch, "status": "error", "error": f"{type(e).__name__}: {e}"})
            continue
        elapsed = time.time() - t0
        metrics = evaluate_anomalies(result.predicted_sequences, result.labeled_sequences)
        es = result.smoothed_errors
        n_test_windows = result.n_test_windows

        print(f"Predicted sequences ({len(result.predicted_sequences)}): "
              f"{result.predicted_sequences}")
        print(f"Labeled sequences  ({len(result.labeled_sequences)}): "
              f"{result.labeled_sequences}")
        print(f"Normalized error: {result.normalized_error:.4f}")
        print(f"e_s: mean={es.mean():.4f}, std={es.std():.4f}, "
              f"max={es.max():.4f}")
        print(f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"F0.5={metrics['f0_5']:.3f}")
        print(f"  Wall time: {elapsed:.1f}s")
        print()
        all_results.append({"chan_id": ch, "predicted_sequences": result.predicted_sequences, "labeled_sequences": result.labeled_sequences, "normalized_error": result.normalized_error, "e_s_mean": float(es.mean()), "e_s_std": float(es.std()), "e_s_max": float(es.max()), "n_test_windows": int(n_test_windows), "metrics": metrics, "wall_time_s": elapsed})
        summary_rows.append({"chan_id": ch,"status": "ok","n_pred": len(result.predicted_sequences),"n_labeled": len(result.labeled_sequences),"tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],"precision": round(metrics["precision"], 4),"recall": round(metrics["recall"], 4),"f0_5": round(metrics["f0_5"], 4),"norm_err": round(result.normalized_error, 4),"e_s_mean": round(float(es.mean()), 4),"e_s_max": round(float(es.max()), 4),"wall_time_s": round(elapsed, 1)})
    ok = [r for r in all_results]
    if ok:
        total_tp = sum(r["metrics"]["tp"] for r in ok)
        total_fp = sum(r["metrics"]["fp"] for r in ok)
        total_fn = sum(r["metrics"]["fn"] for r in ok)
        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_f05 = 0.0
        if (0.25 * total_precision + total_recall) > 0:
            total_f05 = 1.25 * total_precision * total_recall / (0.25 * total_precision + total_recall)
        print("AGGREGATE ACROSS ALL CHANNELS")
        print(f"Total TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
        print(f"Pooled precision: {total_precision:.4f}")
        print(f"Pooled recall:    {total_recall:.4f}")
        print(f"Pooled F0.5:      {total_f05:.4f}")
        print(f"(Paper Table 2 reports: precision=0.875, recall=0.800, F0.5=0.71 total)")
    json_path = OUTPUT_DIR / "telemanon_spot_test_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {json_path}")
    csv_path = OUTPUT_DIR / "telmanon_spot_test_summary.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

if __name__ == "__main__":
    main()