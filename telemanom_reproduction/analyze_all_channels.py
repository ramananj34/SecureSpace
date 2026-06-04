#THIS FILE WAS MADE WITH HEAVY AI ASSISTANCE

from __future__ import annotations
import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent

PAPER_TABLE_2 = {"overall": {"precision": 0.875, "recall": 0.800, "f0_5": 0.71}, "smap":{"precision": 0.855, "recall": 0.855, "f0_5": 0.71}, "msl":        {"precision": 0.926, "recall": 0.694, "f0_5": 0.69}, "point": {"recall_total": 0.903, "recall_smap": 0.953, "recall_msl": 0.789},"contextual": {"recall_total": 0.690, "recall_smap": 0.760, "recall_msl": 0.588}}

def pooled_metrics(group: list[dict]) -> dict:
    tp = sum(r["metrics"]["tp"] for r in group)
    fp = sum(r["metrics"]["fp"] for r in group)
    fn = sum(r["metrics"]["fn"] for r in group)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = (1.25 * p * r / (0.25 * p + r)) if (0.25 * p + r) > 0 else 0.0
    return {"n_channels": len(group), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f0_5": f05}

def fmt(d: dict, paper_f05: float | None = None) -> str:
    s = (f"channels={d['n_channels']:>3}  "
         f"TP={d['tp']:>3} FP={d['fp']:>3} FN={d['fn']:>3}  "
         f"P={d['precision']:.3f}  R={d['recall']:.3f}  F0.5={d['f0_5']:.3f}")
    if paper_f05 is not None:
        delta = d["f0_5"] - paper_f05
        s += f" [paper F0.5≈{paper_f05:.2f}, Δ={delta:+.2f}]"
    return s

def get_anomaly_classes(labels_path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(labels_path)
    classes: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        cls_str = str(row["class"])
        for word in ("contextual", "point", "collective"):
            cls_str = cls_str.replace(word, f"'{word}'")
        try:
            parsed = ast.literal_eval(cls_str)
            if isinstance(parsed, list):
                classes[row["chan_id"]].extend(str(c) for c in parsed)
            else:
                classes[row["chan_id"]].append(str(parsed))
        except (ValueError, SyntaxError):
            classes[row["chan_id"]].append(str(row["class"]))
    return dict(classes)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(_PROJECT_ROOT / "runs" / "all_eval_results.json"))
    p.add_argument("--manifest", default=str(_PROJECT_ROOT / "smap_msl_data" / "channel_manifest.csv"))
    p.add_argument("--labels", default=str(_PROJECT_ROOT / "smap_msl_data" / "labeled_anomalies.csv"))
    p.add_argument("--out", default=str(_PROJECT_ROOT / "runs" / "final_report.txt"))
    args = p.parse_args()
    with open(args.results) as f:
        results = json.load(f)
    ok_results = [r for r in results if r.get("status") == "ok"]
    manifest = pd.read_csv(args.manifest).set_index("chan_id")
    classes = get_anomaly_classes(Path(args.labels))
    for r in ok_results:
        c = r["chan_id"]
        r["spacecraft"] = manifest.loc[c, "spacecraft"] if c in manifest.index else "?"
        r["manifest_status"] = manifest.loc[c, "status"] if c in manifest.index else "?"
        r["classes"] = classes.get(c, [])
    lines: list[str] = []
    def w(s: str = ""):
        print(s); lines.append(s)

    w("=" * 78)
    w("Final Diagnostic Report")
    w("=" * 78)
    w(f"Channels evaluated: {len(results)}")
    w(f"  successful: {len(ok_results)}")
    w(f"  failed: {len(results) - len(ok_results)}")
    w("")
    overall = pooled_metrics(ok_results)
    w("--- OVERALL (pooled across all evaluated channels) ---")
    w(f"  {fmt(overall, PAPER_TABLE_2['overall']['f0_5'])}")
    w(f"  Paper Table 2: P={PAPER_TABLE_2['overall']['precision']:.3f}  "
      f"R={PAPER_TABLE_2['overall']['recall']:.3f}  "
      f"F0.5={PAPER_TABLE_2['overall']['f0_5']:.3f}")
    w("")
    w("--- BY SPACECRAFT ---")
    for sc, paper_key in (("SMAP", "smap"), ("MSL", "msl")):
        group = [r for r in ok_results if r["spacecraft"] == sc]
        w(f"  {sc:<6}  {fmt(pooled_metrics(group), PAPER_TABLE_2[paper_key]['f0_5'])}")
    w("")
    w("--- BY MANIFEST STATUS ---")
    for status in ("usable", "flat_train", "too_short"):
        group = [r for r in ok_results if r["manifest_status"] == status]
        w(f"  {status:<10}  {fmt(pooled_metrics(group))}")
    w("  (No paper comparison; these splits are AMRCC-specific.)")
    w("")
    w("--- BY ANOMALY CLASS ---")
    w("  Note: channels with multiple classes appear in both groups.")
    w("  Paper Table 4 gives RECALL per anomaly type only (not F0.5).")
    point_results = [r for r in ok_results if "point" in r["classes"]]
    ctx_results   = [r for r in ok_results if "contextual" in r["classes"]]
    p_point = pooled_metrics(point_results)
    p_ctx   = pooled_metrics(ctx_results)
    w(f" point {fmt(p_point)}")
    w(f" paper recall: total={PAPER_TABLE_2['point']['recall_total']:.3f}")
    w(f"  contextual {fmt(p_ctx)}")
    w(f" paper recall: total={PAPER_TABLE_2['contextual']['recall_total']:.3f}")
    w("")
    w("--- CHANNELS OF NOTE ---")
    perfect = [r for r in ok_results if r["metrics"]["f0_5"] == 1.0 and r["metrics"]["n_labeled"] > 0]
    w(f"\n  PERFECT (F0.5 = 1.0, n_labeled > 0): {len(perfect)}")
    for r in perfect:
        w(f" {r['chan_id']:<6}  {r['spacecraft']:<4}  "
          f"status={r['manifest_status']:<10}  "
          f"pred={r['metrics']['n_predicted']} lab={r['metrics']['n_labeled']}")
    silent = [r for r in ok_results if r["metrics"]["n_predicted"] == 0 and r["metrics"]["n_labeled"] > 0]
    w(f"\n  SILENT (no predictions; labels exist): {len(silent)}")
    for r in silent:
        w(f" {r['chan_id']:<6}  {r['spacecraft']:<4}  "
          f"status={r['manifest_status']:<10}  "
          f"lab={r['metrics']['n_labeled']}  "
          f"e_s_max={r['e_s_max']:.3f}  norm_err={r['normalized_error']:.3f}")
    over = [r for r in ok_results if r["metrics"]["n_predicted"] > r["metrics"]["n_labeled"] + 1]
    w(f"\n  OVER-DETECTING (n_pred > n_labeled + 1): {len(over)}")
    for r in over:
        w(f" {r['chan_id']:<6}  {r['spacecraft']:<4}  "
          f"status={r['manifest_status']:<10}  "
          f"pred={r['metrics']['n_predicted']} lab={r['metrics']['n_labeled']}  "
          f"TP={r['metrics']['tp']} FP={r['metrics']['fp']}")
    wrong = [r for r in ok_results if r["metrics"]["n_predicted"] > 0 and r["metrics"]["tp"] == 0 and r["metrics"]["n_labeled"] > 0]
    w(f"\n  PREDICTED BUT MISSED EVERY LABEL: {len(wrong)}")
    for r in wrong:
        w(f" {r['chan_id']:<6}  {r['spacecraft']:<4}  "
          f"status={r['manifest_status']:<10}")
        w(f" predicted: {r['predicted_sequences']}")
        w(f" labeled: {r['labeled_sequences']}")
    w("\n--- ANOMALY COUNT SANITY CHECK ---")
    total_labeled = sum(r["metrics"]["n_labeled"] for r in ok_results)
    total_pred    = sum(r["metrics"]["n_predicted"] for r in ok_results)
    w(f"  Total labeled sequences (post-mask-merge): {total_labeled}")
    w(f"  Total predicted sequences: {total_pred}")
    w(f"  Paper reports: 82 unique labeled anomalies across 81 channels")
    w(f"  Difference between paper's 82 and our number reflects mask-merging of")
    w(f"  adjacent labeled ranges within a channel.")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())