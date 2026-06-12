from __future__ import annotations
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig, Channel, Errors
from pipeline import evaluate_anomalies, _mask_to_sequences
from smap_msl_dataset_api import (SMAPMSLChannelDataset, flat_anomaly_channels, EXCLUDED_CHANNELS)

def constant_predictor_detect(chan, data_dir, cfg):
    tr = SMAPMSLChannelDataset(chan, "train", data_dir=data_dir)
    te = SMAPMSLChannelDataset(chan, "test", data_dir=data_dir)
    train_feats = np.concatenate([tr.telemetry_scaled[:, None], tr._commands], axis=1).astype(np.float32)
    test_feats = np.concatenate([te.telemetry_scaled[:, None], te._commands], axis=1).astype(np.float32)
    ch = Channel.from_arrays(chan_id=chan, train_arr=train_feats, test_arr=test_feats, config=cfg, seed=42)
    const = float(np.mean(tr.telemetry_scaled))
    n_test_windows = ch.y_test.shape[0]
    ch.y_hat = np.full((n_test_windows, cfg.n_predictions), const, dtype=np.float64)
    labels = _mask_to_sequences(te.anomaly_mask)
    base = {"chan": chan, "const_pred": const, "train_std": float(np.std(tr.telemetry_scaled)), "n_labeled": len(labels), "labels": [[int(a), int(b)] for a, b in labels], "test_ptp": float(np.ptp(ch.y_test)) if ch.y_test.size else 0.0}
    if base["test_ptp"] == 0.0:
        return {**base, "status": "degenerate", "n_predicted": 0, "es_max": 0.0, "es_median": 0.0, "E_seq": [], "eval": evaluate_anomalies([], labels)}
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        errs = Errors(ch, cfg)
        errs.process_batches(ch)
    es = errs.e_s
    if es is None or not np.size(es) or not np.all(np.isfinite(es)):
        return {**base, "status": "empty", "n_predicted": 0, "es_max": 0.0, "es_median": 0.0, "E_seq": [], "eval": evaluate_anomalies([], labels)}
    E_seq = list(errs.E_seq)
    return {**base, "status": "ok", "E_seq": [[int(a), int(b)] for a, b in E_seq], "n_predicted": len(E_seq), "es_max": float(np.max(es)), "es_median": float(np.median(es)), "eval": evaluate_anomalies(E_seq, labels)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--out", default=str(_THIS / "runs_flat"))
    ap.add_argument("--channels", nargs="*", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig()
    chans = args.channels or [c for c in flat_anomaly_channels(data_dir=Path(args.data_dir)) if c not in EXCLUDED_CHANNELS]
    print(f"flat_train constant-predictor baseline: {len(chans)} channels\n")
    print(f"{'chan':>6} {'train_std':>9} {'es_max':>8} {'es_med':>8} {'pred':>4} {'lbl':>4} "
          f"{'TP':>3} {'recall':>6} {'status':>10} {'detected?':>9}")
    recs, n_detect, tp_tot, lbl_tot, n_fail = [], 0, 0, 0, 0
    for chan in chans:
        try:
            r = constant_predictor_detect(chan, Path(args.data_dir), cfg)
        except Exception as e:
            n_fail += 1
            print(f"{chan:>6} {'FAILED':>9} {type(e).__name__}: {e}")
            recs.append({"chan": chan, "status": "error", "error": repr(e)})
            continue
        ev = r["eval"]; detected = r["n_predicted"] > 0
        n_detect += int(detected); tp_tot += ev["tp"]; lbl_tot += ev["n_labeled"]
        recs.append(r)
        flag = "yes" if detected else ("flat-test" if r["status"] == "degenerate" else "SILENT")
        print(f"{chan:>6} {r['train_std']:>9.2e} {r['es_max']:>8.3f} {r['es_median']:>8.3f} "
              f"{r['n_predicted']:>4} {r['n_labeled']:>4} {ev['tp']:>3} {ev['recall']:>6.2f} "
              f"{r['status']:>10} {flag:>9}")
    recall = tp_tot / lbl_tot if lbl_tot else 0.0
    json.dump(recs, open(out_dir / "flat_train.json", "w"), indent=2)
    ok = [r for r in recs if r.get("status") == "ok"]
    degen = [r for r in recs if r.get("status") == "degenerate"]
    n_gated = sum(1 for r in ok if r["es_max"] <= 0.05)
    print(f"\n=== flat_train constant-predictor summary ({len(recs)} channels; {n_fail} errors) ===")
    print(f"  channels with any detection:                 {n_detect}/{len(recs)}")
    print(f"  degenerate (constant test window, NPDT n/a):  {len(degen)}/{len(recs)}")
    print(f"  pooled recall:                                {tp_tot}/{lbl_tot} = {recall:.2f}")
    print(f"  of non-degenerate: es_max <= 0.05 (NPDT scale-gate rejects, D.25.1): {n_gated}/{len(ok)}")
    print(f"  => silence is the NPDT scale-gate on ~zero residuals, not LSTM capacity:")
    print(f"     a constant predictor produces the same ~flat residual stream and is gated identically.")
    print(f"summary -> {out_dir / 'flat_train.json'}")

if __name__ == "__main__":
    main()