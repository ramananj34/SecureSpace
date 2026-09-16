from __future__ import annotations
import json
import glob
import argparse
from pathlib import Path
import numpy as np

def e9_realized(e9_dir):
    out = {}
    for fn in sorted(glob.glob(str(Path(e9_dir) / "*.json"))):
        d = json.load(open(fn)); best = None
        for lab in d["labels"]:
            for cr in lab.get("by_config", {}).values():
                for e in cr.get("per_eps", []):
                    m = e.get("model")
                    if m and m.get("rate") is not None:
                        cand = (m["rate"], m.get("slot_weight_mean"))
                        if best is None or cand[0] > best[0]:
                            best = cand
        out[d["chan"]] = best
    return out

def operational_bound(e9_dir, e14_dir):
    realized = e9_realized(e9_dir)
    rows = []
    for fn in sorted(glob.glob(str(Path(e14_dir) / "*.json"))):
        d = json.load(open(fn)); ch = d["chan"]
        lab = max(d["labels"], key=lambda l: l["max_eta"])
        curve = sorted((c["w"], c["eta"]) for c in lab["curve"])
        ws = [w for w, _ in curve]; es = [e for _, e in curve]
        rz = realized.get(ch)
        w_real = rz[1] if (rz and rz[1] is not None) else None
        rows.append({"chan": ch, "max_eta": lab["max_eta"], "r_cert": lab["r_cert"], "w_realized": w_real, "eta_at_realized": (float(np.interp(w_real, ws, es)) if w_real is not None else None), "e9_defended": (rz[0] if rz else None), "attack_defeated": rz is None})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e9-dir", default="amrcc/runs_e9")
    ap.add_argument("--e14-dir", default="amrcc/runs_e14")
    args = ap.parse_args()
    rows = operational_bound(args.e9_dir, args.e14_dir)
    print(f"operational Theorem-2 bound   ({args.e9_dir}  x  {args.e14_dir})")
    print(f"{'chan':6}{'max_w eta':>11}{'r_cert':>8}{'w_real':>9}{'eta@real':>10}{'E9 def':>9}  note")
    print("-" * 74)
    for r in sorted(rows, key=lambda r: -(r["eta_at_realized"] if r["eta_at_realized"] is not None else -1)):
        if r["attack_defeated"]:
            print(f"{r['chan']:6}{r['max_eta']:>11.2f}{r['r_cert']:>8}{'--':>9}{'--':>10}{'--':>9}  attack defeated")
        else:
            gap = "  << intrinsic" if (r["max_eta"] - r["eta_at_realized"]) > 0.2 else ""
            print(f"{r['chan']:6}{r['max_eta']:>11.2f}{r['r_cert']:>8}{r['w_realized']:>9.0f}"
                  f"{r['eta_at_realized']:>10.3f}{r['e9_defended']:>9.1%}{gap}")
    print("\neta@real = operational max_w eta_w at the attacker's realizable slot weight (Thm 2);")
    print("tracks E9 defended, <= intrinsic max_w eta_w.  On adv dirs this row IS the combo bound.")

if __name__ == "__main__":
    main()