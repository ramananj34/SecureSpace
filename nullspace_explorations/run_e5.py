from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "nullspace_attack")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, detect, missed
from smap_msl_dataset_api import Quantizer
from nullspace_attack import c2_ceiling_attack, CeilingConfig

B_GRID = [4, 6, 8, 12, 16]
DLSB8 = 2.0 ** -7
EPS_GRID = tuple(2.0 ** e for e in range(-8, 1))
EXT_MAX_EPS = 32.0
N_BISECT = 4

def in_range_evaders(e2_dir):
    out = []
    for p in sorted(Path(e2_dir).glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if j.get("clipping_affected"):
            continue
        if any(m.get("clean_hit_quantized") and m.get("min_evasion_eps")
               for m in j.get("min_evasion", [])):
            out.append(p.stem)
    return out

def e2_evading_label_and_eps(chan, e2_dir):
    p = Path(e2_dir) / f"{chan}.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    me = next((m for m in j.get("min_evasion", []) if m.get("clean_hit_quantized") and m.get("min_evasion_eps")), None)
    if me is None:
        return None
    return {"label": tuple(me["label"]), "eps8": float(me["min_evasion_eps"]), "config_used": me.get("config_used", "base")}

def e1_ceiling_eps(chan, label, e1_dir):
    p = Path(e1_dir) / f"{chan}.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    for m in j.get("min_evasion", []):
        if tuple(m.get("label", ())) == tuple(label) and m.get("min_evasion_lsb") is not None:
            return float(m["min_evasion_lsb"]) * DLSB8
    return None

def min_evasion_at_b(chan, model, tf, tele, cmds, labels, label, cfg, base_ccfg, strong_ccfg, q):
    tele_q = q.dequantize(q.quantize(np.asarray(tele, np.float64)))
    clean_hit = not missed(detect(chan, model, tf, tele_q, cmds, cfg), label)
    if not clean_hit:
        return None, False, None, 0
    calls = [0]
    def attack(eps, ccfg):
        calls[0] += 1
        r = c2_ceiling_attack(chan, model, tf, tele, cmds, labels, label, cfg, ccfg, float(eps), q, return_arrays=False)
        return bool(r["ceiling_missed_lattice"])
    base_grid = [(eps, attack(eps, base_ccfg)) for eps in EPS_GRID]
    if any(m for _, m in base_grid):
        cfg_used, bisect_ccfg, allg = "base", base_ccfg, list(base_grid)
    else:
        cfg_used, bisect_ccfg = "strong", strong_ccfg
        strong_grid = [(eps, attack(eps, strong_ccfg)) for eps in EPS_GRID]
        ext, e = [], EPS_GRID[-1]
        while e < EXT_MAX_EPS:
            e *= 2.0
            m = attack(e, strong_ccfg)
            ext.append((e, m))
            if m:
                break
        allg = strong_grid + ext
    missed_eps = sorted(eps for eps, m in allg if m)
    if not missed_eps:
        return None, True, cfg_used, calls[0]
    hi = missed_eps[0]
    below = [eps for eps, m in allg if (not m) and eps < hi]
    lo = max(below) if below else hi / 4.0
    for _ in range(N_BISECT):
        mid = float((lo * hi) ** 0.5)
        if attack(mid, bisect_ccfg):
            hi = mid
        else:
            lo = mid
    return hi, True, cfg_used, calls[0]

def run_channel(chan, data_dir, runs_dir, e1_dir, e2_dir, cfg, base_ccfg, strong_ccfg):
    info = e2_evading_label_and_eps(chan, e2_dir)
    if info is None:
        print(f"  {chan}: no evading E2 label -- skipping")
        return None
    label, eps8_e2 = info["label"], info["eps8"]
    tele, cmds, tf, labels, T = load_streams(chan, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = tf.shape[1]
    sc = {25: "SMAP", 55: "MSL"}.get(int(n_feat), f"?{n_feat}")
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    mask = (np.arange(T) < label[0] - 100) | (np.arange(T) > label[1] + 100)
    sigma = float(np.std(tele[mask]))
    e1_eps = e1_ceiling_eps(chan, label, e1_dir)
    e1_over_sigma = (e1_eps / sigma) if (e1_eps and sigma > 0) else None
    print(f"\n=== {chan} ({sc}) label {list(label)}  sigma={sigma/DLSB8:.0f} LSB  "
          f"E1_cont={'%.2f sigma' % e1_over_sigma if e1_over_sigma else 'n/a'}  "
          f"E2 b=8 ref={eps8_e2/DLSB8:.1f} LSB ({eps8_e2/sigma:.2f} sigma) ===")
    rows = []
    for b in B_GRID:
        q = Quantizer(x_min=-1.0, x_max=1.0, b=b)
        me, clean_hit, cfg_used, calls = min_evasion_at_b(
            chan, model, tf, tele, cmds, labels, label, cfg, base_ccfg, strong_ccfg, q)
        me_over_sigma = (me / sigma) if (me and sigma > 0) else None
        pen = ((me_over_sigma - e1_over_sigma)
               if (me_over_sigma is not None and e1_over_sigma is not None) else None)
        rows.append({"b": b, "delta_lsb": q.delta_lsb, "clean_hit_at_b": clean_hit, "config_used": cfg_used, "min_evasion_eps": me, "min_evasion_over_sigma": me_over_sigma, "penalty_over_sigma": pen, "attack_calls": calls})
        if not clean_hit:
            s = "clean detection erased by b"
        elif me is None:
            s = f"robust (no evasion <={EXT_MAX_EPS/sigma:.1f} sigma, {cfg_used})"
        else:
            s = f"min-evas={me_over_sigma:.2f} sigma ({cfg_used})" + (f"  penalty={pen:+.2f} sigma" if pen is not None else "")
        print(f"  b={b:2d} (dlsb={q.delta_lsb:.3e}): {s}  [{calls} calls]")
    return {"chan": chan, "spacecraft": sc, "label": list(label), "sigma_scaled": sigma, "sigma_lsb_b8": sigma / DLSB8, "e1_continuous_eps": e1_eps, "e1_continuous_over_sigma": e1_over_sigma, "e2_b8_ref_eps": eps8_e2, "e2_b8_ref_over_sigma": (eps8_e2 / sigma if sigma > 0 else None), "by_b": rows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--e1-dir", default=str(_ROOT / "baseline_fgsm_pgd" / "runs"))
    ap.add_argument("--e2-dir", default=str(_ROOT / "nullspace_attack" / "runs_e2"))
    ap.add_argument("--out", default=str(_THIS / "runs_e5"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base-ewma-steps", type=int, default=60)
    ap.add_argument("--base-ewma-restarts", type=int, default=3)
    ap.add_argument("--base-square-iters", type=int, default=200)
    ap.add_argument("--strong-ewma-steps", type=int, default=150)
    ap.add_argument("--strong-ewma-restarts", type=int, default=5)
    ap.add_argument("--strong-square-iters", type=int, default=400)
    ap.add_argument("--query-budget", type=int, default=80)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig()
    base_ccfg = CeilingConfig(band_T_steps=args.base_ewma_steps, band_restarts=args.base_ewma_restarts, square_iters=args.base_square_iters, query_budget=args.query_budget)
    strong_ccfg = CeilingConfig(band_T_steps=args.strong_ewma_steps, band_restarts=args.strong_ewma_restarts, square_iters=args.strong_square_iters, query_budget=args.query_budget)
    todo = args.channels if args.channels else in_range_evaders(Path(args.e2_dir))
    print(f"device={DEVICE}  E5 resolution sweep b={B_GRID} (min-evasion in sigma)")
    print(f"in-range evaders ({len(todo)}): {todo}")
    print(f"base {base_ccfg.band_T_steps}x{base_ccfg.band_restarts} sq={base_ccfg.square_iters} | "
          f"strong {strong_ccfg.band_T_steps}x{strong_ccfg.band_restarts} sq={strong_ccfg.square_iters} | "
          f"N_BISECT={N_BISECT} EXT_MAX={EXT_MAX_EPS}\n")
    recs = []
    for i, chan in enumerate(todo, 1):
        cpath = out_dir / f"{chan}.json"
        if cpath.exists() and not args.force:
            print(f"[{i}/{len(todo)}] {chan}: cached"); recs.append(json.load(open(cpath))); continue
        t0 = time.time()
        try:
            r = run_channel(chan, Path(args.data_dir), Path(args.runs_dir), Path(args.e1_dir), Path(args.e2_dir), cfg, base_ccfg, strong_ccfg)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  {chan}: FAILED {type(e).__name__}: {e}"); continue
        if r:
            r["seconds"] = time.time() - t0
            tmp = cpath.with_suffix(".json.tmp"); json.dump(r, open(tmp, "w"), indent=2); tmp.replace(cpath)
            recs.append(r)
            print(f"  [{chan}] {r['seconds']:.0f}s -> {cpath}")
    if recs:
        json.dump(recs, open(out_dir / "e5_summary.json", "w"), indent=2)
        print(f"\n=== E5 summary ({len(recs)} channels) — penalty (C2 - E1_continuous) vs b, in sigma ===")
        print(f"{'b':>3} {'clean OK':>9} {'mean min-evas(sigma)':>20} {'mean penalty(sigma)':>19}")
        for b in B_GRID:
            pts = [x for r in recs for x in r["by_b"] if x["b"] == b]
            n_clean = sum(1 for x in pts if x["clean_hit_at_b"])
            mevs = [x["min_evasion_over_sigma"] for x in pts if x["min_evasion_over_sigma"] is not None]
            pens = [x["penalty_over_sigma"] for x in pts if x["penalty_over_sigma"] is not None]
            mev = f"{np.mean(mevs):.3f}" if mevs else "n/a"
            pen = f"{np.mean(pens):+.3f}" if pens else "n/a"
            print(f"{b:>3} {f'{n_clean}/{len(pts)}':>9} {mev:>20} {pen:>19}")
        print("\nexpect: mean penalty shrinks toward 0 as b grows (C2 -> continuous oracle);")
        print("        E5 b=8 min-evas reproduces each channel's E2 b=8 ref (validation);")
        print("        clean-OK count drops at coarse b (quantization erases marginal detection).")
        print(f"summary -> {out_dir / 'e5_summary.json'}")
    print("done")

if __name__ == "__main__":
    main()