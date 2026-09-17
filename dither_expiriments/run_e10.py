from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "smap_msl_data"), str(_ROOT / "telemanom_reproduction"), str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from smap_msl_dataset_api import (Quantizer, SMAPMSLChannelDataset, working_channels, flat_anomaly_channels, load_manifest, EXCLUDED_CHANNELS)
from keyed_permutation import permutation_for_frame, inverse_permutation
from sa_search import SAConfig, anneal_multi
from recovery_metrics import (score_candidate, observe, m1_random_baseline, m4_random_baseline, spearman)
import discriminator as DISC
from dither import add_dither, sigma_from_lsb
from legit_cost import reconstruction_cost
from info_measures import dither_info_curve

B = 8
SIGMAS_LSB_DEFAULT = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
REAL_SIZES_DEFAULT = [800, 7200, 32400]
SYNTH_SIZES_DEFAULT = [64, 256, 1024]
N_BASELINE = 8
N_TIEBREAK = 4
N_CORPUS_FRAMES = 48

def size_layout(size, b=B):
    if size == 32400:
        header = 400; n_coords = (size - header) // b
    else:
        header = 0; n_coords = size // b
    return {"k": header + n_coords * b, "header_bits": header, "n_coords": n_coords,
            "slot_bits": n_coords * b, "b": b}

def synth_stream(n, seed):
    rng = np.random.default_rng([12345, seed])
    x = np.cumsum(rng.normal(0, 1, n)); x = 0.8 * (x - x.mean()) / (np.abs(x).max() + 1e-9)
    return np.clip(x, -0.95, 0.95)

def block_to_message(block_vals, sigma, rng, q, header_bits):
    d = add_dither(np.asarray(block_vals, np.float64), sigma, rng)
    lv = q.quantize(d)
    bits = np.asarray(q.levels_to_bits(lv), np.uint8).reshape(-1)
    m_in = np.concatenate([np.zeros(header_bits, np.uint8), bits]).astype(np.uint8)
    xs = np.asarray(q.dequantize(lv), np.float64)
    return m_in, xs

def collect_corpus(stream, n_coords, header_bits, sigma, q, n_frames, seed):
    T = len(stream)
    span = max(1, T - n_coords)
    stride = max(1, span // max(1, n_frames))
    starts = list(range(0, span, stride))[:n_frames]
    M = []
    for i, s0 in enumerate(starts):
        m_in, _ = block_to_message(stream[s0:s0 + n_coords], sigma, np.random.default_rng([seed, 1, i]), q, header_bits)
        M.append(m_in)
    k = header_bits + n_coords * B
    return np.stack(M) if M else np.zeros((0, k), np.uint8), len(starts)

def keyed_perm(k, seed):
    key = np.random.default_rng([seed, 777]).bytes(32)
    return permutation_for_frame(key, 0, k)

def _pair_sa(vals, pair_full, pair_scalar, iters, restarts, seed, max_seg, seg_frac):
    k = len(vals)
    v = np.asarray(vals)

    def full(sigma):
        return float(pair_full(v[sigma]))

    def dfn(sigma, i, j):
        if i == j:
            return 0.0
        a = v[sigma[i]]; b = v[sigma[j]]
        if a == b:
            return 0.0
        pairs = set()
        for d in (i, j):
            if d > 0:
                pairs.add((d - 1, d))
            if d < k - 1:
                pairs.add((d, d + 1))
        old = 0.0; new = 0.0
        for (p, q) in pairs:
            vp = v[sigma[p]]; vq = v[sigma[q]]
            old += pair_scalar(vp, vq)
            np_ = b if p == i else (a if p == j else vp)
            nq_ = b if q == i else (a if q == j else vq)
            new += pair_scalar(np_, nq_)
        return float(new - old)

    cfg = SAConfig(iters=int(iters), restarts=int(restarts), seed=int(seed), init="random", max_seg=int(max_seg), seg_move_frac=float(seg_frac))
    res = anneal_multi(k, full, cfg, delta_swap=dfn)
    return res.pi_hat, res.fitness

def bitrun_recover(m_obs, iters, restarts, seed, max_seg=64, seg_frac=0.0):
    v = np.asarray(m_obs, np.int64)
    pi_hat_sigma, fit = _pair_sa(
        v, lambda s: np.sum(s[:-1] == s[1:]), lambda u, w: 1.0 if u == w else 0.0,
        iters, restarts, seed, max_seg, seg_frac)
    return inverse_permutation(pi_hat_sigma), fit

def smoothness_recover(obs_vals, iters, restarts, seed, max_seg=16, seg_frac=0.3):
    x = np.asarray(obs_vals, np.float64)
    sigma, fit = _pair_sa(
        x, lambda s: -np.sum(np.diff(s) ** 2), lambda u, w: -(u - w) ** 2,
        iters, restarts, seed, max_seg, seg_frac)
    return sigma, fit

def arm_baseline(m_in, perm, q, lay, n_draws, seed):
    accs, ratios, m2s, m4s = [], [], [], []
    for d in range(n_draws):
        pj = np.random.default_rng([seed, 2, d]).permutation(lay["k"])
        rec = score_candidate(m_in, perm, pj, quantizer=q, b=lay["b"], header_bits=lay["header_bits"], slot_bits=lay["slot_bits"])
        accs.append(rec["m1_bit_acc"]); ratios.append(rec["m1_ratio"])
        m2s.append(rec["m2_pos_acc"]); m4s.append(rec["m4_msb_loc"])
    return {"m1_bit_acc_mean": float(np.mean(accs)), "m1_ratio_mean": float(np.mean(ratios)),
            "m2_pos_acc_mean": float(np.mean(m2s)), "m4_msb_loc_mean": float(np.mean(m4s))}

def arm_marginal(m_in, perm, p, q, lay, n_tiebreak, seed):
    m_obs = observe(m_in, perm)
    base = m1_random_baseline(lay["k"], int(m_in.sum()))
    accs, m4s, sps = [], [], []
    for t in range(n_tiebreak):
        pi_hat = DISC.marginal_recovery_permutation(m_obs, p, rng=np.random.default_rng([seed, 3, t]))
        rec = score_candidate(m_in, perm, pi_hat, quantizer=q, b=lay["b"], header_bits=lay["header_bits"], slot_bits=lay["slot_bits"])
        accs.append(rec["m1_bit_acc"]); m4s.append(rec["m4_msb_loc"])
        sps.append(rec["telem_order"]["abs_spearman"])
    acc = float(np.mean(accs))
    return {"m1_bit_acc": acc, "m1_lift_over_random": float(acc - base),
            "m1_ratio": float(acc / base) if base > 0 else float("nan"),
            "m4_msb_loc": float(np.mean(m4s)), "telem_order_abs_spearman": float(np.mean(sps))}

def arm_analytic_sa(m_in, perm, q, lay, iters, restarts, seed):
    m_obs = observe(m_in, perm)
    pi_hat, fit = bitrun_recover(m_obs, iters, restarts, seed)
    rec = score_candidate(m_in, perm, pi_hat, quantizer=q, b=lay["b"], header_bits=lay["header_bits"], slot_bits=lay["slot_bits"])
    return {"m1_bit_acc": rec["m1_bit_acc"], "m1_ratio": rec["m1_ratio"],
            "m2_pos_acc": rec["m2_pos_acc"], "m4_msb_loc": rec["m4_msb_loc"],
            "telem_order_abs_spearman": rec["telem_order"]["abs_spearman"], "fitness": float(fit)}

def arm_ceiling(xs_true, iters, restarts, seed, max_seg, seg_frac):
    n = len(xs_true)
    order = np.random.default_rng([seed, 4]).permutation(n)
    obs = np.asarray(xs_true, np.float64)[order]
    sigma, fit = smoothness_recover(obs, iters, restarts, seed, max_seg=max_seg, seg_frac=seg_frac)
    recovered = obs[sigma]
    s = spearman(recovered, np.asarray(xs_true, np.float64)) if n >= 3 else float("nan")
    return {"telem_order_abs_spearman": float(abs(s)) if n >= 3 else float("nan"),
            "fitness": float(fit), "n_coords": int(n)}

def necessary_condition(corpus_bits, seed, do_joint, disc_kw):
    if corpus_bits.shape[0] < 8:
        return {"note": "corpus too small for AUC", "n_frames": int(corpus_bits.shape[0])}
    n = corpus_bits.shape[0]; n_tr = max(4, int(n * 0.7))
    mres = DISC.marginal_discriminator_auc(corpus_bits[:n_tr], corpus_bits[n_tr:], seed=seed)
    out = {"auc_marginal": mres["auc_marginal"], "n_frames": int(n),
           "n_pos": mres["n_pos"], "n_neg": mres["n_neg"]}
    auc_joint = None
    if do_joint and DISC._HAVE_TORCH:
        _, jm = DISC.train_discriminator(corpus_bits, seed=seed, verbose=False, **(disc_kw or {}))
        auc_joint = jm["auc_joint_test"]
        out["auc_joint_test"] = auc_joint; out["auc_joint_train"] = jm["auc_joint_train"]
    out["verdict"] = DISC.necessary_condition_verdict(mres["auc_marginal"], auc_joint)
    return out

def sa_budget(n_coords):
    k = n_coords * B
    bit_iters = int(min(30000, 30 * k)); bit_restarts = 2
    ceil_iters = int(min(60000, 200 * n_coords))
    if n_coords <= 512:
        ceil_restarts = 3; ceil_seg = 0.3; ceil_maxseg = min(32, max(4, n_coords // 4))
    else:
        ceil_restarts = 1; ceil_seg = 0.05; ceil_maxseg = 16
    return bit_iters, bit_restarts, ceil_iters, ceil_restarts, ceil_seg, ceil_maxseg

def run_size(stream, size, sigmas, q, seed, args):
    lay = size_layout(size)
    nc, hdr = lay["n_coords"], lay["header_bits"]
    T = len(stream)
    if T < nc + 4:
        return {"size": size, "layout": lay, "note": f"stream too short ({T}) for n_coords={nc}"}
    target_block = stream[T - nc:T]
    bit_iters, bit_restarts, ceil_iters, ceil_restarts, ceil_seg, ceil_maxseg = sa_budget(nc)
    per_sigma = []
    for i_s, s_lsb in enumerate(sigmas):
        sigma = sigma_from_lsb(s_lsb, q)
        m_in, xs = block_to_message(target_block, sigma, np.random.default_rng([seed, 5, i_s]), q, hdr)
        perm = keyed_perm(lay["k"], seed + i_s)
        corpus, n_corp = collect_corpus(stream, nc, hdr, sigma, q, N_CORPUS_FRAMES, seed)
        p = DISC.position_marginals(corpus) if n_corp > 0 else np.full(lay["k"], 0.5)
        rec = {
            "sigma_lsb": float(s_lsb),
            "m1_baseline": m1_random_baseline(lay["k"], int(m_in.sum())),
            "m4_baseline": m4_random_baseline(nc) if nc > 0 else float("nan"),
            "n_corpus_frames": int(n_corp),
            "baseline": arm_baseline(m_in, perm, q, lay, N_BASELINE, seed + i_s),
            "marginal": arm_marginal(m_in, perm, p, q, lay, N_TIEBREAK, seed + i_s),
            "analytic_sa": arm_analytic_sa(m_in, perm, q, lay, bit_iters, bit_restarts, seed + i_s),
            "ceiling": arm_ceiling(xs, ceil_iters, ceil_restarts, seed + i_s, ceil_maxseg, ceil_seg),
            "necessary_condition": necessary_condition(
                corpus, seed + i_s, args.discriminator,
                {"hidden": tuple(args.disc_hidden), "epochs": args.disc_epochs}),
        }
        per_sigma.append(rec)
    return {"size": size, "layout": lay, "sa_budget":
            {"bit_iters": bit_iters, "ceil_iters": ceil_iters, "ceil_restarts": ceil_restarts},
            "per_sigma": per_sigma}

def channel_curves(stream, sigmas, q, args, chan, data_dir):
    sig_vals = [sigma_from_lsb(s, q) for s in sigmas]
    out = {
        "reconstruction_cost": reconstruction_cost(stream, q, sig_vals, n_reps=args.cost_reps, seed=0),
        "info_measures": dither_info_curve(stream, q, sig_vals, n_reps=args.info_reps, seed=0),
    }
    if args.detection:
        out["detection_cost"] = _detection_curve(chan, stream, sig_vals, q, args, data_dir)
    return out

def _detection_curve(chan, stream, sig_vals, q, args, data_dir):
    from fgsm_pgd_attacks import detect, missed, load_streams, prepare_model
    from pipeline import load_trained_model, evaluate_anomalies
    from VENDOR_telemanom import VendoredConfig
    from legit_cost import detection_cost
    import torch
    cfg = VendoredConfig()
    tele, cmds, tf, labels, T = load_streams(chan, Path(data_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prepare_model(load_trained_model(Path(args.runs_dir) / chan / "model.pt", n_features=tf.shape[1], device=device))
    return detection_cost(chan, model, tf, np.asarray(tele, np.float64), cmds, labels, cfg, q, sig_vals, n_reps=args.det_reps, seed=0, detect_fn=detect, missed_fn=missed, evaluate_fn=evaluate_anomalies)

def channel_entropy_stat(chan, data_dir):
    try:
        tr = SMAPMSLChannelDataset(chan, "train", data_dir=Path(data_dir))
        return float(tr.telemetry_scaled.std())
    except Exception:
        return float("nan")

def run_channel(chan, sizes, sigmas, q, args):
    data_dir = args.data_dir
    if chan.startswith("synth"):
        n = int(chan.replace("synth", "")) if chan[5:].isdigit() else 4096
        stream = synth_stream(max(n, max(sizes) // B + 8), seed=abs(hash(chan)) % (1 << 16))
        spacecraft = "SYNTH"; entropy = float(stream.std())
        size_list = [s for s in sizes]
    else:
        te = SMAPMSLChannelDataset(chan, "test", data_dir=Path(data_dir))
        stream = te.telemetry_scaled
        spacecraft = "MSL" if te._commands.shape[1] >= 54 else "SMAP"
        entropy = channel_entropy_stat(chan, data_dir)
        size_list = [s for s in sizes]
    out = {"chan": chan, "spacecraft": spacecraft, "train_std": entropy,
           "n_timesteps": int(len(stream)), "sigmas_lsb": list(sigmas),
           "channel_curves": channel_curves(stream, sigmas, q, args, chan, data_dir)
                              if not chan.startswith("synth") else
                              {"reconstruction_cost":
                                   reconstruction_cost(stream, q, [sigma_from_lsb(s, q) for s in sigmas], n_reps=args.cost_reps, seed=0),
                               "info_measures":
                                   dither_info_curve(stream, q, [sigma_from_lsb(s, q) for s in sigmas], n_reps=args.info_reps, seed=0)},
           "sizes": {}}
    for size in size_list:
        out["sizes"][str(size)] = run_size(stream, size, sigmas, q, seed=abs(hash((chan, size))) % (1 << 20), args=args)
    return out

def default_channels(data_dir):
    try:
        m = load_manifest(Path(data_dir))
        usable = [c for c in working_channels(data_dir=Path(data_dir))]
        flats = [c for c in flat_anomaly_channels(data_dir=Path(data_dir))]
        stds = {}
        for c in usable:
            stds[c] = channel_entropy_stat(c, data_dir)
        ordered = sorted(usable, key=lambda c: stds.get(c, 0.0))
        picks = []
        if flats:
            picks.append(flats[0])
        if ordered:
            picks += [ordered[len(ordered) // 2]]
            picks += ordered[-2:]
        seen = []
        for c in picks:
            if c not in seen and c not in EXCLUDED_CHANNELS:
                seen.append(c)
        return seen or (usable[:3] if usable else [])
    except Exception:
        return ["A-2", "E-1"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_e10"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--channels", nargs="*", default=None, help="channel ids (real or synthNN). default: entropy-spanning pick.")
    ap.add_argument("--sizes", nargs="*", type=int, default=REAL_SIZES_DEFAULT)
    ap.add_argument("--synthetic", action="store_true", help="also run synthetic sizes as synthNN channels")
    ap.add_argument("--synth-sizes", nargs="*", type=int, default=SYNTH_SIZES_DEFAULT)
    ap.add_argument("--sigmas-lsb", nargs="*", type=float, default=SIGMAS_LSB_DEFAULT)
    ap.add_argument("--discriminator", action="store_true", help="also train joint MLP AUC (needs torch)")
    ap.add_argument("--disc-hidden", nargs="*", type=int, default=[256, 64])
    ap.add_argument("--disc-epochs", type=int, default=40)
    ap.add_argument("--detection", action="store_true", help="NPDT detection-preserved curve (heavy)")
    ap.add_argument("--augment-detection", action="store_true", help="add ONLY the detection curve to already-cached channel JSONs (no arm recompute)")
    ap.add_argument("--cost-reps", type=int, default=8)
    ap.add_argument("--info-reps", type=int, default=8)
    ap.add_argument("--det-reps", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    q = Quantizer()
    sigmas = list(args.sigmas_lsb)

    chans = args.channels if args.channels else default_channels(args.data_dir)
    synth_chans = [f"synth{s}" for s in args.synth_sizes] if args.synthetic else []

    print(f"E10 sweep | channels={chans} synth={synth_chans}")
    print(f"sizes={args.sizes} sigmas_lsb={sigmas} discriminator={args.discriminator} detection={args.detection}\n")

    jobs = [(c, args.sizes) for c in chans] + [(c, [int(c.replace('synth', ''))]) for c in synth_chans]
    for i, (chan, sizes) in enumerate(jobs, 1):
        cpath = out_dir / f"{chan}.json"
        if cpath.exists() and not args.force:
            if args.augment_detection and not chan.startswith("synth"):
                rec = json.load(open(cpath))
                if "detection_cost" not in rec.get("channel_curves", {}):
                    try:
                        te = SMAPMSLChannelDataset(chan, "test", data_dir=Path(args.data_dir))
                        stream = te.telemetry_scaled
                        sig_vals = [sigma_from_lsb(s, q) for s in sigmas]
                        rec["channel_curves"]["detection_cost"] = _detection_curve(
                            chan, stream, sig_vals, q, args, args.data_dir)
                        tmp = cpath.with_suffix(".json.tmp")
                        json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(cpath)
                        print(f"[{i}/{len(jobs)}] {chan}: detection curve added")
                    except Exception as e:
                        print(f"[{i}/{len(jobs)}] {chan}: detection augment FAILED {type(e).__name__}: {e}")
                else:
                    print(f"[{i}/{len(jobs)}] {chan}: detection already present, skipping")
            else:
                print(f"[{i}/{len(jobs)}] {chan}: cached, skipping")
            continue
        t0 = time.time()
        try:
            rec = run_channel(chan, sizes, sigmas, q, args)
        except Exception as e:
            print(f"[{i}/{len(jobs)}] {chan}: FAILED {type(e).__name__}: {e}")
            continue
        tmp = cpath.with_suffix(".json.tmp"); json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(cpath)
        dt = time.time() - t0
        try:
            smallest = str(min(sizes))
            s0 = rec["sizes"][smallest]["per_sigma"][0]
            print(f"[{i}/{len(jobs)}] {chan} (std={rec['train_std']:.3g}): "
                  f"size {smallest} sigma0 -- baseline M1r={s0['baseline']['m1_ratio_mean']:.3f} "
                  f"marginal M1r={s0['marginal']['m1_ratio']:.3f} M4={s0['marginal']['m4_msb_loc']:.3g} "
                  f"(base {s0['m4_baseline']:.3g}) | SA M1r={s0['analytic_sa']['m1_ratio']:.3f} "
                  f"| ceiling |rho|={s0['ceiling']['telem_order_abs_spearman']:.2f} "
                  f"| AUC_marg={s0['necessary_condition'].get('auc_marginal', float('nan')):.3f}   {dt:.0f}s")
        except Exception:
            print(f"[{i}/{len(jobs)}] {chan}: done   {dt:.0f}s")
            
    sa_r, marg_r, marg_lift = [], [], []
    m4_ratio_clean, m4_flat = [], []
    for jp in sorted(out_dir.glob("*.json")):
        r = json.load(open(jp))
        for size, sr in r.get("sizes", {}).items():
            ps = sr.get("per_sigma")
            if not ps:
                continue
            s0 = ps[0]
            if "analytic_sa" in s0 and np.isfinite(s0["analytic_sa"].get("m1_ratio", np.nan)):
                sa_r.append(s0["analytic_sa"]["m1_ratio"])
            if "marginal" in s0:
                marg_r.append(s0["marginal"]["m1_ratio"])
                marg_lift.append(s0["marginal"]["m1_lift_over_random"])
                base = s0.get("m4_baseline", np.nan)
                m4 = s0["marginal"]["m4_msb_loc"]
                if np.isfinite(base) and base > 0:
                    ratio = m4 / base
                    if s0.get("m1_baseline", 0.5) >= 0.90:
                        m4_flat.append(ratio)
                    else:
                        m4_ratio_clean.append(ratio)
    print("\n=== E10 summary (sigma=0) ===")
    if sa_r:
        sa_r = np.array(sa_r)
        print(f"[GATE] attacker (analytic_sa) M1 ratio vs random baseline: "
              f"mean={np.nanmean(sa_r):.3f} max={np.nanmax(sa_r):.3f}  "
              f"(Sec 8.5: < 1.05)  cells passing: {int((sa_r < 1.05).sum())}/{np.isfinite(sa_r).sum()}")
    if marg_r:
        marg_r = np.array(marg_r); marg_lift = np.array(marg_lift)
        print(f"[report] marginal (coordinate-BLIND) M1 ratio: mean={np.nanmean(marg_r):.3f} "
              f"max={np.nanmax(marg_r):.3f}  (lift over random: mean={np.nanmean(marg_lift):+.3f} "
              f"max={np.nanmax(marg_lift):+.3f}) -- NOT a breach; see M4 below")
    if m4_ratio_clean:
        a = np.array(m4_ratio_clean)
        print(f"[SECURITY] coordinate-identity M4 / chance (non-degenerate cells): "
              f"mean={np.nanmean(a):.2f}x max={np.nanmax(a):.2f}x  ({len(a)} cells; ~1x = no recovery)")
    if m4_flat:
        print(f"[note] flat/degenerate cells excluded from M4 ratio: {len(m4_flat)} "
              f"(tiny-count M4 noise, e.g. A-1)")
    print("done")

if __name__ == "__main__":
    main()