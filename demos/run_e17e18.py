from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "ldpc"), str(_ROOT / "nullspace_attack_utils"),
           str(_ROOT / "amrcc"), str(_ROOT / "fl_core"), str(_ROOT / "baseline_fgsm_pgd"),
           str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bp_decoder as BP
import link_budget as LB
import uplink as UP
from grad_channel import GradientQuantizer, isl_key

def run_e18_bercurves(code, ebn0_grid, n_trials, seed, out_dir, force):
    p = out_dir / "e18_bercurves.json"
    if p.exists() and not force:
        print("  e18 BER curves: cached"); return json.load(open(p))
    t0 = time.time()
    base = LB.run_curve(code, ebn0_grid, n_trials, use_perm=False, max_iter=50, seed=seed)
    amrcc = LB.run_curve(code, ebn0_grid, n_trials, use_perm=True, max_iter=50, seed=seed)
    cmp = LB.compare_curves(base, amrcc)
    rec = {"baseline": base, "amrcc": amrcc, "compare": cmp, "n_trials": n_trials}
    json.dump(rec, open(p, "w"), indent=2)
    print(f"  e18 BER curves ({time.time()-t0:.0f}s): 0-dB (all CI overlap) = {cmp['all_overlap']}")
    return rec

def run_e18_epsdec(code, operating_snrs, n_trials, seed, out_dir, force):
    p = out_dir / "e18_epsdec.json"
    if p.exists() and not force:
        print("  e18 eps_dec: cached"); return json.load(open(p))
    rows = [LB.eps_dec_at(code, s, n_trials, max_iter=50, seed=seed) for s in operating_snrs]
    json.dump({"rows": rows}, open(p, "w"), indent=2)
    for r in rows:
        print(f"  eps_dec @ {r['ebn0_db']}dB = {r['eps_dec']:.4g} (CI {r['ci_lo']:.3g}-{r['ci_hi']:.3g})")
    return {"rows": rows}


def run_e17_uplink(seed, out_dir, force):
    p = out_dir / "e17_uplink.json"
    if p.exists() and not force:
        print("  e17 uplink: cached"); return json.load(open(p))
    rng = np.random.default_rng(seed)
    d_model = 5000
    theta = rng.normal(0, 0.1, d_model)
    sens = np.zeros(d_model); sens[: max(50, d_model // 100)] = rng.normal(0, 1, max(50, d_model // 100))
    sens = rng.normal(0, 1, d_model)
    gq = GradientQuantizer(b=8); key = isl_key(seed, 0)
    res = UP.uplink_experiment(theta, gq, key, w=300, sensitivity=sens, seed=seed)
    json.dump(res, open(p, "w"), indent=2)
    td = res["targeted_damage"]
    print(f"  e17 uplink: targeted damage undef={td['undefended']:.4g} vs amrcc={td['amrcc']:.4g} "
          f"({td['undefended']/max(td['amrcc'],1e-12):.0f}x dilution); L2 undef={res['l2']['undefended']:.3f} "
          f"vs amrcc={res['l2']['amrcc']:.3f}")
    return res

def run_endtoend(code, chan, data_dir, runs_dir, ebn0_db, seed, out_dir, force):
    p = out_dir / f"endtoend_{chan}.json"
    if p.exists() and not force:
        print("  end-to-end: cached"); return json.load(open(p))
    try:
        from VENDOR_telemanom import VendoredConfig
        from pipeline import load_trained_model
        from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, footprint_mask, detect, missed
        from smap_msl_dataset_api import Quantizer
    except Exception as e:
        print(f"  end-to-end skipped (imports: {type(e).__name__})"); return None
    cfg = VendoredConfig(); q = Quantizer()
    tele, cmds, tf, labels, T = load_streams(chan, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = tf.shape[1]
    model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    rate = code.k / code.n
    sigma = LB.ebn0_to_sigma(ebn0_db, rate)
    rng = np.random.default_rng(seed)

    levels0 = q.quantize(tele); bits0 = np.asarray(q.levels_to_bits(levels0)).reshape(-1)
    k = code.k
    n_frames = int(np.ceil(bits0.size / k))
    padded = np.zeros(n_frames * k, np.uint8); padded[:bits0.size] = bits0
    recovered = np.zeros_like(padded)
    frame_fail = 0
    for f in range(n_frames):
        msg = padded[f * k:(f + 1) * k]
        perm = rng.permutation(k); msg_p = msg[perm]
        c = np.asarray(code.encode(msg_p)).astype(np.uint8)
        llr = BP.awgn_llr_seeded(c, sigma, rng)
        chat, iters, ok = BP.bp_decode(code, llr, max_iter=50)
        msg_p_hat = code.info_bit_extract(chat)
        inv = np.empty_like(perm); inv[perm] = np.arange(k)
        recovered[f * k:(f + 1) * k] = msg_p_hat[inv]
        if not ok or np.any(msg_p_hat != msg_p):
            frame_fail += 1
    bits_hat = recovered[:bits0.size]
    tele_hat = np.asarray(q.dequantize(q.bits_to_levels(bits_hat.reshape(-1, q.b))), np.float64)[:len(tele)]

    E_true = detect(chan, model, tf, tele, cmds, cfg)
    E_recv = detect(chan, model, tf, tele_hat, cmds, cfg)
    f1_true = sum(0 if missed(E_true, l) else 1 for l in labels) / len(labels) if labels else 0.0
    f1_recv = sum(0 if missed(E_recv, l) else 1 for l in labels) / len(labels) if labels else 0.0
    rec = {"chan": chan, "ebn0_db": ebn0_db, "n_frames": n_frames, "frame_failures": frame_fail,
           "bit_error_rate": float(np.mean(bits_hat != bits0)),
           "telemetry_mse_roundtrip": float(np.mean((tele - tele_hat) ** 2)),
           "f1_true": f1_true, "f1_recovered": f1_recv,
           "detection_preserved": bool(abs(f1_true - f1_recv) < 1e-9)}
    json.dump(rec, open(p, "w"), indent=2)
    print(f"  end-to-end {chan} @ {ebn0_db}dB: {n_frames} frames, {frame_fail} BP failures, "
          f"BER={rec['bit_error_rate']:.2g}, detection preserved={rec['detection_preserved']} "
          f"(F1 {f1_true}->{f1_recv})")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_e17e18"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--channel", default="D-16")
    ap.add_argument("--ebn0-grid", nargs="*", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--operating-snrs", nargs="*", type=float, default=[3.0, 3.5])
    ap.add_argument("--endtoend-snr", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    code = BP.load_short_code()
    print(f"E17/E18 | code n={code.n} k={code.k} rate={code.k/code.n:.4f} | Eb/N0={args.ebn0_grid} "
          f"trials={args.n_trials}\n")
    print("E18 link-budget (BER curves):")
    run_e18_bercurves(code, args.ebn0_grid, args.n_trials, args.seed, out_dir, args.force)
    print("E18 eps_dec:")
    run_e18_epsdec(code, args.operating_snrs, max(args.n_trials * 2, 400), args.seed, out_dir, args.force)
    print("E17 on-board AI uplink:")
    run_e17_uplink(args.seed, out_dir, args.force)
    print("E18 end-to-end pipeline:")
    run_endtoend(code, args.channel, Path(args.data_dir), Path(args.runs_dir), args.endtoend_snr,
                 args.seed, out_dir, args.force)
    print("\ndone")

if __name__ == "__main__":
    main()