from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "nullspace_attack"), str(_ROOT / "ldpc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, detect, missed
from smap_msl_dataset_api import Quantizer
from nullspace_attack import c2_ceiling_attack, CeilingConfig
from frame_ops import (get_long_code, embed_window_in_frame, frame_analysis_over_stream, HEADER_BITS, SLOT_BITS, N_SLOTS)
from keyed_permutation import permutation_for_frame, apply_inverse_perm
from sampling import selection_sample, clopper_pearson

DLSB = 2.0 ** -7
WINDOW = 100
TARGET_SLOT = 0

def load_attackable(runs_e2_dir: Path, only=None):
    out = []
    for jp in sorted(runs_e2_dir.glob("*.json")):
        rec = json.load(open(jp))
        chan = rec["chan"]
        if only and chan not in only:
            continue
        base_cfg, strong_cfg = rec.get("base_config"), rec.get("strong_config")
        for me in rec.get("min_evasion", []):
            if me.get("clean_hit_quantized") and me.get("min_evasion_eps") is not None:
                out.append({"chan": chan, "label": [int(me["label"][0]), int(me["label"][1])], "min_evasion_eps": float(me["min_evasion_eps"]), "config_used": me.get("config_used"), "base_config": base_cfg, "strong_config": strong_cfg})
    return out

def ccfg_from_dict(d):
    return CeilingConfig(**{k: d[k] for k in d})

def _scatter_once(delta_bits, bits_clean, code, mode, key, rng, b):
    T = delta_bits.shape[0]
    k = code.k
    lo = HEADER_BITS + TARGET_SLOT * SLOT_BITS
    bits_pert = bits_clean.copy()
    slot_w = 0
    for wi, w0 in enumerate(range(0, T - WINDOW + 1, WINDOW)):
        wbits = np.asarray(delta_bits[w0:w0 + WINDOW], np.uint8)
        wsum = int(wbits.sum())
        if wsum == 0:
            continue
        if mode == "transplant":
            msg = embed_window_in_frame(wbits.reshape(WINDOW * b), slot=TARGET_SLOT, k=k)
            v = apply_inverse_perm(msg, permutation_for_frame(key, wi, k))
            v_slot = np.asarray(v[lo:lo + SLOT_BITS], np.uint8).reshape(WINDOW, b)
        else:
            pos = selection_sample(k, wsum, rng)
            loc = pos[(pos >= lo) & (pos < lo + SLOT_BITS)] - lo
            v_slot = np.zeros((WINDOW, b), np.uint8)
            if loc.size:
                v_slot[loc // b, loc % b] = 1
        slot_w += int(v_slot.sum())
        bits_pert[w0:w0 + WINDOW] ^= v_slot
    return bits_pert, slot_w

def defended_arm(chan, model, tf, cmds, cfg, label, delta_bits, bits_clean, q, code, mode, n, seed, alpha=0.05):
    b = int(getattr(q, "b", 8))
    rng = np.random.default_rng(seed)
    key_rng = np.random.default_rng(seed ^ 0x9E3779B9)
    misses, slot_ws = 0, []
    for _ in range(int(n)):
        key = key_rng.bytes(32) if mode == "transplant" else b""
        bits_pert, sw = _scatter_once(delta_bits, bits_clean, code, mode, key, rng, b)
        slot_ws.append(sw)
        tele_pert = np.asarray(q.dequantize(q.bits_to_levels(bits_pert)), np.float64)
        if missed(detect(chan, model, tf, tele_pert, cmds, cfg), label):
            misses += 1
    lo, hi = clopper_pearson(misses, int(n), alpha)
    return {"mode": mode, "n": int(n), "misses": int(misses), "rate": misses / float(n), "ci_lo": float(lo), "ci_hi": float(hi), "slot_weight_mean": float(np.mean(slot_ws)) if slot_ws else 0.0, "slot_weight_max": int(np.max(slot_ws)) if slot_ws else 0}

def run_channel(chan, entries, data_dir, runs_dir, cfg, q, code, args):
    tele, cmds, tf, labels, T = load_streams(chan, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = tf.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    levels_clean = q.quantize(tele)
    bits_clean = np.asarray(q.levels_to_bits(levels_clean), np.uint8)
    tele_q = np.asarray(q.dequantize(levels_clean), np.float64)
    b = int(getattr(q, "b", 8))
    out = {"chan": chan, "spacecraft": "SMAP" if n_feat == 25 else "MSL", "n_timesteps": int(T), "window": WINDOW, "target_slot": TARGET_SLOT, "n_model": args.n_model, "n_transplant": args.n_transplant, "eps_mult": list(args.eps_mult), "labels": []}
    for ent in entries:
        label = ent["label"]
        clean_missed = bool(missed(detect(chan, model, tf, tele_q, cmds, cfg), label))
        lab_rec = {"label": label, "min_evasion_eps": ent["min_evasion_eps"], "config_used": ent["config_used"], "clean_missed": clean_missed, "by_config": {}}
        if clean_missed:
            lab_rec["note"] = "anomaly not detected on clean quantized stream -- out of scope"
            out["labels"].append(lab_rec)
            continue
        configs = {"base": ent["base_config"], "strong": ent["strong_config"]}
        for cfg_tag in args.configs:
            ccfg = ccfg_from_dict(configs[cfg_tag])
            per_eps = []
            for mult in args.eps_mult:
                eps = ent["min_evasion_eps"] * float(mult)
                r = c2_ceiling_attack(chan, model, tf, tele, cmds, labels, label, cfg, ccfg, float(eps), q, return_arrays=True)
                delta_bits = np.asarray(r["_delta_info_bits"], np.uint8)
                fa = frame_analysis_over_stream(delta_bits, code=code, window=WINDOW, slot=TARGET_SLOT)
                rec = {"eps": float(eps), "eps_lsb": float(eps) / DLSB, "oracle_missed": bool(r["ceiling_missed_lattice"]), "delta_info_weight_stream": int(r["delta_info_weight_stream"]), "realized_linf_lsb": float(r["realized_linf_lsb"]), "ceiling_collateral": int(r["ceiling_collateral"]), "frame_analysis": {k: v for k, v in fa.items() if k != "per_frame"}, "model": None, "transplant": None}
                if rec["oracle_missed"]:
                    seed = (hash((chan, tuple(label), cfg_tag, round(eps, 9))) & 0x7FFFFFFF)
                    rec["model"] = defended_arm(chan, model, tf, cmds, cfg, label, delta_bits, bits_clean, q, code, "model", args.n_model, seed)
                    if args.n_transplant > 0:
                        rec["transplant"] = defended_arm(chan, model, tf, cmds, cfg, label, delta_bits, bits_clean, q, code, "transplant", args.n_transplant, seed + 1)
                per_eps.append(rec)
            valid = [e for e in per_eps if e["model"] is not None]
            worst = max(valid, key=lambda e: e["model"]["rate"]) if valid else None
            lab_rec["by_config"][cfg_tag] = {"per_eps": per_eps, "oracle_evades": any(e["oracle_missed"] for e in per_eps), "worst_eps_lsb": (worst["eps_lsb"] if worst else None), "defended_rate_model": (worst["model"]["rate"] if worst else None), "defended_ci_hi_model": (worst["model"]["ci_hi"] if worst else None), "defended_rate_transplant": (worst["transplant"]["rate"] if worst and worst["transplant"] else None)}
        out["labels"].append(lab_rec)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-e2", default=str(_ROOT / "nullspace_attack" / "runs_e2"))
    ap.add_argument("--out", default=str(_THIS / "runs_e9"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--channels", nargs="*", default=None, help="subset (smoke test: A-2)")
    ap.add_argument("--configs", nargs="*", default=["base", "strong"], choices=["base", "strong"])
    ap.add_argument("--eps-mult", nargs="*", type=float, default=[1.0], help="multipliers on min_evasion_eps; worst-case sweep e.g. 1.0 2.0 4.0")
    ap.add_argument("--n-model", type=int, default=500, help="Setting-(a) uniform-B_w realizations")
    ap.add_argument("--n-transplant", type=int, default=100, help="real-permutation realizations (0=skip)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig()
    q = Quantizer()
    code = get_long_code()
    assert WINDOW * int(getattr(q, "b", 8)) == SLOT_BITS, "window*b must equal SLOT_BITS=800"
    attackable = load_attackable(Path(args.runs_e2), only=set(args.channels) if args.channels else None)
    by_chan = {}
    for e in attackable:
        by_chan.setdefault(e["chan"], []).append(e)
    chans = sorted(by_chan)
    print(f"device={DEVICE}  attackable channels={len(chans)}  labels={len(attackable)}")
    print(f"configs={args.configs}  eps_mult={args.eps_mult}  "
          f"n_model={args.n_model}  n_transplant={args.n_transplant}\n")
    agg = []
    for i, chan in enumerate(chans, 1):
        cpath = out_dir / f"{chan}.json"
        if cpath.exists() and not args.force:
            print(f"[{i}/{len(chans)}] {chan}: cached, skipping")
            continue
        t0 = time.time()
        try:
            rec = run_channel(chan, by_chan[chan], Path(args.data_dir), Path(args.runs_dir), cfg, q, code, args)
        except Exception as e:
            print(f"[{i}/{len(chans)}] {chan}: FAILED {type(e).__name__}: {e}")
            continue
        tmp = cpath.with_suffix(".json.tmp"); json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(cpath)
        rates = []
        for lab in rec["labels"]:
            for cfg_tag, cr in lab.get("by_config", {}).items():
                if cr["defended_rate_model"] is not None:
                    rates.append((cr["defended_rate_model"], cr["defended_rate_transplant"], lab["label"], cfg_tag))
        worst = max(rates, key=lambda x: x[0]) if rates else None
        agg.extend(rates)
        dt = time.time() - t0
        if worst:
            tp = f"{worst[1]:.1%}" if worst[1] is not None else "n/a"
            print(f"[{i}/{len(chans)}] {chan}: defended worst model={worst[0]:.1%} (transplant={tp}) "
                  f"@ {worst[3]} cfg, label {worst[2]}   {dt:.0f}s")
        else:
            print(f"[{i}/{len(chans)}] {chan}: no evading undefended attack to defend   {dt:.0f}s")
    if agg:
        m = np.array([a[0] for a in agg])
        print(f"\n=== E9 summary over {len(agg)} (channel,label,config) cells ===")
        print(f"defended (model) success: mean={m.mean():.2%}  max={m.max():.2%}  "
              f"<=5%: {int((m <= 0.05).sum())}/{len(m)}")
        tp = np.array([a[1] for a in agg if a[1] is not None])
        if tp.size:
            print(f"transplant (real pi) success: mean={tp.mean():.2%}  max={tp.max():.2%}  "
                  f"(Lemma-1 cross-check vs model)")
        print("target (Sec 8.5): defended success <= max_w eta_w + eps_PRG (<=5%);  "
              "oracle/undefended ~100% by construction")
    print("done")

if __name__ == "__main__":
    main()