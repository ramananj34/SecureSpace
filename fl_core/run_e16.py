from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "fl_core"), str(_ROOT / "amrcc"),
           str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"),
           str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, detect, missed
from smap_msl_dataset_api import Quantizer
import constellation as CN
import federated as FED
import grad_channel as GC
import fl_attack as ATK

try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

REGIMES = ["fedavg", "median", "fedavg_perm", "median_perm"]

def _clean_holdout_windows(features, l_s, n_windows=200, seed=0):
    T = len(features)
    n = min(n_windows, T - l_s)
    start = T - l_s - n
    X = np.stack([features[start + i:start + i + l_s] for i in range(n)])
    y = features[start + l_s:start + l_s + n, 0]
    return X.astype(np.float32), y.astype(np.float32)


def _mse(model, X, y, device):
    import torch
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        Xt = torch.tensor(X, device=device)
        pred = model(Xt)[:, 0].detach().cpu().numpy()
    return float(np.mean((pred - y) ** 2))


def _f1(chan, model, tf, tele, cmds, cfg, labels):
    E = detect(chan, model, tf, np.asarray(tele, np.float64), cmds, cfg)
    tp = sum(0 if missed(E, lab) else 1 for lab in labels)
    n = len(labels)
    return float(tp / n) if n else 0.0


def run_regime(chan, regime, bm_frac, R, con, cfg, q, data, runs_dir, local_epochs, lr, seed,
               scatter_w, unbounded_scale):
    import torch
    tele, cmds, tf, labels, T = load_streams(chan, data)
    tele = np.asarray(tele, np.float64); n_feat = tf.shape[1]
    features = np.concatenate([tele[:, None], cmds], axis=1).astype(np.float32)
    clean_model = prepare_model(load_trained_model(runs_dir / chan / "model.pt", n_features=n_feat, device=DEVICE))
    from telemanom_lstm import TelemanomLSTM
    ctor = lambda: TelemanomLSTM(clean_model.config)
    theta_clean, shapes = FED.flatten_params(clean_model.state_dict())
    Xh, yh = _clean_holdout_windows(features, cfg.l_s)
    mse_clean = _mse(clean_model, Xh, yh, DEVICE)

    global_vec = theta_clean.copy()
    M = con.n_sats
    B = int(round(bm_frac * M))
    B_mask = ATK.byzantine_mask(M, B, seed=seed)
    parts = FED.temporal_partitions(len(features), M, cfg.l_s)
    gq = GC.GradientQuantizer(b=int(q.b))
    agg_name, perm_on = ATK.regime_aggregator(regime)

    traj = []
    for r in range(int(R)):
        gstate = FED.unflatten_params(global_vec, shapes, clean_model.state_dict())
        honest_updates = []
        for s in range(M):
            lo, hi = parts[s]
            if B_mask[s]:
                honest_updates.append(np.zeros_like(global_vec))
            else:
                delta, _ = FED.local_update(ctor, gstate, features[lo:hi], cfg,
                                            local_epochs=local_epochs, lr=lr, device=DEVICE,
                                            seed=seed * 100000 + r * 100 + s)
                honest_updates.append(delta)
        U = ATK.apply_attack(honest_updates, regime, B_mask, gq=gq, base_seed=seed, t=r,
                             unbounded_scale=unbounded_scale, scatter_w=scatter_w)
        f_byz = B if agg_name == "krum" else 0
        global_delta = FED.aggregate(agg_name, U, f=f_byz)
        global_vec = global_vec + global_delta
        # metrics
        upd_state = FED.unflatten_params(global_vec, shapes, clean_model.state_dict())
        upd = ctor().to(DEVICE); upd.load_state_dict(upd_state); upd = prepare_model(upd)
        mse = _mse(upd, Xh, yh, DEVICE)
        finite = bool(np.isfinite(global_vec).all())
        traj.append({"round": r, "mse_ratio": (mse / mse_clean if mse_clean > 0 and np.isfinite(mse) else float("inf")),
                     "theta_dist": float(np.linalg.norm(global_vec - theta_clean)),
                     "finite": finite})
        if not finite:
            break
    final_state = FED.unflatten_params(global_vec, shapes, clean_model.state_dict())
    fm = ctor().to(DEVICE); fm.load_state_dict(final_state); fm = prepare_model(fm)
    final_f1 = _f1(chan, fm, tf, tele, cmds, cfg, labels) if np.isfinite(global_vec).all() else 0.0
    return {"chan": chan, "regime": regime, "bm_frac": bm_frac, "M": M, "B": B, "R": int(R),
            "aggregator": agg_name, "perm_on": perm_on, "mse_clean_baseline": mse_clean,
            "trajectory": traj, "final_mse_ratio": traj[-1]["mse_ratio"] if traj else float("inf"),
            "final_theta_dist": traj[-1]["theta_dist"] if traj else float("inf"),
            "final_f1": final_f1, "collapsed": (not traj[-1]["finite"]) if traj else True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_e16"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--channel", default="A-1")
    ap.add_argument("--regimes", nargs="*", default=REGIMES)
    ap.add_argument("--bm", nargs="*", type=float, default=[0.0, 0.30])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--n-planes", type=int, default=5)
    ap.add_argument("--sats-per-plane", type=int, default=8)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--scatter-w", type=int, default=100)
    ap.add_argument("--unbounded-scale", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig(); q = Quantizer()
    con = CN.Constellation(walker=CN.WalkerDelta(n_planes=args.n_planes, sats_per_plane=args.sats_per_plane))
    print(f"device={DEVICE}  channel={args.channel}  M={con.n_sats}  R={args.rounds}  "
          f"regimes={args.regimes}  B/M={args.bm}\n")
    summary = []
    for regime in args.regimes:
        for bm in args.bm:
            tag = f"{args.channel}_{regime}_bm{int(bm*100)}"
            cpath = out_dir / f"{tag}.json"
            if cpath.exists() and not args.force:
                print(f"[{tag}] cached"); continue
            t0 = time.time()
            try:
                rec = run_regime(args.channel, regime, bm, args.rounds, con, cfg, q,
                                 Path(args.data_dir), Path(args.runs_dir), args.local_epochs, args.lr,
                                 args.seed, args.scatter_w, args.unbounded_scale)
            except Exception as e:
                print(f"[{tag}] FAILED {type(e).__name__}: {e}"); continue
            json.dump(rec, open(cpath, "w"), indent=2)
            summary.append((regime, bm, rec["final_mse_ratio"], rec["collapsed"], rec["final_f1"]))
            print(f"[{tag}] final mse_ratio={rec['final_mse_ratio']:.3g}  collapsed={rec['collapsed']}  "
                  f"F1={rec['final_f1']:.2f}  {time.time()-t0:.0f}s")
    if summary:
        print("\n=== E16 summary (final model quality) ===")
        print(f"{'regime':13} {'B/M':>5} {'mse_ratio':>12} {'collapsed':>10}")
        for regime, bm, mr, col, f1 in summary:
            print(f"{regime:13} {bm*100:>4.0f}% {mr:>12.3g} {str(col):>10}")
    print("done")


if __name__ == "__main__":
    main()