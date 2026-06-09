from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model, evaluate_anomalies
from fgsm_pgd_attacks import (DEVICE, prepare_model, load_streams, detect, missed, footprint_mask)
from smap_msl_dataset_api import Quantizer

HEADLINE = ["A-2", "A-3", "A-4", "A-7", "D-11", "D-16", "E-2", "E-3", "E-8", "F-2", "F-8", "G-6", "G-7", "M-3", "M-4", "M-5", "P-3", "P-7", "T-12"]
GRADIENT = ["F-5", "M-7", "S-1", "E-7", "D-1", "P-2"]
ALL_CHANNELS = HEADLINE + GRADIENT
DLSB = 2.0 ** -7

def _q_gpu(x, dlsb=DLSB, xmin=-1.0, xmax=1.0, nlev=256):
    xc = torch.clamp(x, xmin, xmax)
    lev = torch.floor((xc - xmin) / (xmax - xmin) * nlev)
    lev = torch.clamp(lev, 0, nlev - 1)
    return xmin + (lev + 0.5) * dlsb

def oracle_replace(model, tele_base, cmds, label, cfg, quantize):
    device = next(model.parameters()).device
    l_s, buf = cfg.l_s, cfg.error_buffer
    T = len(tele_base)
    a, b = int(label[0]), int(label[1])
    lo = max(l_s, a - l_s - buf)
    hi = min(T - 1, b + buf)
    tau = torch.tensor(np.asarray(tele_base, np.float32), device=device)
    cmds_t = torch.tensor(cmds.astype(np.float32), device=device)
    with torch.no_grad():
        for i in range(lo, hi + 1):
            x = torch.cat([tau[i - l_s:i, None], cmds_t[i - l_s:i]], dim=-1)[None]
            pred = model(x)[0, 0]
            tau[i] = _q_gpu(pred) if quantize else pred
    out = tau.detach().cpu().numpy().astype(np.float64)
    return out, (lo, hi)

def run_channel(chan, data_dir, runs_dir, cfg, q):
    tele, cmds, train_features, labels, T = load_streams(chan, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = train_features.shape[1]
    sc = {25: "SMAP", 55: "MSL"}.get(int(n_feat), f"?{n_feat}")
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    tele_q = q.dequantize(q.quantize(tele))
    E0_raw = detect(chan, model, train_features, tele, cmds, cfg)
    E0_q = detect(chan, model, train_features, tele_q, cmds, cfg)
    out_labels = []
    for label in labels:
        a, b = int(label[0]), int(label[1])
        rec = {"label": [a, b]}
        hit_raw = not missed(E0_raw, label)
        rec["clean_hit_raw"] = hit_raw
        if hit_raw:
            orc, (lo, hi) = oracle_replace(model, tele, cmds, label, cfg, quantize=False)
            E = detect(chan, model, train_features, orc, cmds, cfg)
            rec["e1_oracle_missed"] = bool(missed(E, label))
            rec["e1_oracle_pert_lsb"] = float(np.max(np.abs(orc - tele)) / DLSB)
        else:
            rec["e1_oracle_missed"] = None
            rec["e1_oracle_pert_lsb"] = None
        hit_q = not missed(E0_q, label)
        rec["clean_hit_quantized"] = hit_q
        if hit_q:
            orc, (lo, hi) = oracle_replace(model, tele_q, cmds, label, cfg, quantize=True)
            E = detect(chan, model, train_features, orc, cmds, cfg)
            rec["e2_oracle_missed"] = bool(missed(E, label))
            rec["e2_oracle_pert_lsb"] = float(np.max(np.abs(orc - tele_q)) / DLSB)
        else:
            rec["e2_oracle_missed"] = None
            rec["e2_oracle_pert_lsb"] = None
        out_labels.append(rec)
    return {"chan": chan, "spacecraft": sc, "n_timesteps": int(T), "clean_f0_5_raw": float(evaluate_anomalies(E0_raw, labels)["f0_5"]), "clean_f0_5_quantized": float(evaluate_anomalies(E0_q, labels)["f0_5"]), "labels": out_labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--out", default=str(_THIS / "runs_oracle"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig(); q = Quantizer()
    todo = args.channels or ALL_CHANNELS
    print(f"device={DEVICE}  channels={len(todo)}  (epsilon-free oracle, E1 raw + E2 quantized)")
    print(f"{'chan':6} {'sc':4} {'cln_raw':>7} {'E1_orc':>7} {'pert_LSB':>9}   "
          f"{'cln_q':>6} {'E2_orc':>7} {'pert_LSB':>9}")
    recs = []
    e1_brk = e1_tot = e2_brk = e2_tot = 0
    for chan in todo:
        t0 = time.time()
        try:
            r = run_channel(chan, Path(args.data_dir), Path(args.runs_dir), cfg, q)
        except Exception as e:
            print(f"{chan:6} FAILED {type(e).__name__}: {e}")
            continue
        recs.append(r)
        for m in r["labels"]:
            def fmt_missed(v):
                return "MISS" if v is True else ("flag" if v is False else "n/a")
            if m["e1_oracle_missed"] is not None:
                e1_tot += 1; e1_brk += int(m["e1_oracle_missed"])
            if m["e2_oracle_missed"] is not None:
                e2_tot += 1; e2_brk += int(m["e2_oracle_missed"])
            p1 = f"{m['e1_oracle_pert_lsb']:.0f}" if m["e1_oracle_pert_lsb"] is not None else "-"
            p2 = f"{m['e2_oracle_pert_lsb']:.0f}" if m["e2_oracle_pert_lsb"] is not None else "-"
            print(f"{chan:6} {r['spacecraft']:4} "
                  f"{str(m['clean_hit_raw']):>7} {fmt_missed(m['e1_oracle_missed']):>7} {p1:>9}   "
                  f"{str(m['clean_hit_quantized']):>6} {fmt_missed(m['e2_oracle_missed']):>7} {p2:>9}")
    json.dump(recs, open(out_dir / "oracle.json", "w"), indent=2)
    print(f"\nE1 oracle (raw):       breakable {e1_brk}/{e1_tot}  "
          f"(unbreakable: {e1_tot - e1_brk})")
    print(f"E2 oracle (quantized): breakable {e2_brk}/{e2_tot}  "
          f"(unbreakable: {e2_tot - e2_brk})")
    print(f"summary -> {out_dir / 'oracle.json'}")
    print("done")

if __name__ == "__main__":
    main()