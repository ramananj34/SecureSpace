from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc"), str(_ROOT / "adv_subspace"),
           str(_ROOT / "baseline_fgsm_pgd"), str(_ROOT / "smap_msl_data"),
           str(_ROOT / "telemanom_reproduction")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from VENDOR_telemanom import VendoredConfig
from pipeline import load_trained_model
from fgsm_pgd_attacks import DEVICE, prepare_model, load_streams, footprint_mask, detect, missed
from smap_msl_dataset_api import Quantizer
import constellation as CN
import federated as FED
import grad_channel as GC
import dadv_param as DP
import theorem5_energy as T5
from jacobian_dadv import bit_to_coord_matrix, c_quant_from_quantizer

try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False


def _model_ctor(config):
    from telemanom_lstm import TelemanomLSTM
    def ctor():
        return TelemanomLSTM(config)
    return ctor

def f1_on_channel(chan, model, tf, tele, cmds, cfg, labels):
    tp = fn = 0
    E = detect(chan, model, tf, np.asarray(tele, np.float64), cmds, cfg)
    for label in labels:
        if missed(E, label):
            fn += 1
        else:
            tp += 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"recall": float(recall), "n_detected": int(tp), "n_labels": int(tp + fn)}

def run_clean_round(chan, data_dir, runs_dir, cfg, q, con, aggregators, local_epochs, lr, seed,
                    do_theory=True):
    if not _HAVE_TORCH:
        raise RuntimeError("torch required")
    import torch
    tele, cmds, tf, labels, T = load_streams(chan, data_dir)
    tele = np.asarray(tele, np.float64)
    n_feat = tf.shape[1]
    global_model = prepare_model(load_trained_model(runs_dir / chan / "model.pt",
                                                    n_features=n_feat, device=DEVICE))
    ctor = _model_ctor(global_model.config)
    global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
    g0, shapes = FED.flatten_params(global_state)

    features = np.concatenate([tele[:, None], cmds], axis=1).astype(np.float32)
    M = con.n_sats
    parts = FED.temporal_partitions(len(features), M, cfg.l_s)
    gq = GC.GradientQuantizer(b=int(q.b))
    deltas = []
    for s in range(M):
        lo, hi = parts[s]
        delta, _ = FED.local_update(ctor, global_state, features[lo:hi], cfg,
                                    local_epochs=local_epochs, lr=lr, device=DEVICE, seed=seed * 1000 + s)
        key = GC.isl_key(seed, link_id=s)
        rx = GC.transmit_gradient(delta, gq, key, t=0, poison_w=0, seed=seed * 1000 + s)  # CLEAN
        deltas.append(rx["g_recv"])
    deltas = np.array(deltas)

    out = {"chan": chan, "n_sats": M, "local_epochs": local_epochs, "lr": lr,
           "contact_graph": CN.graph_stats(con.contact(0.0)),
           "aggregators": {}}

    for agg_name in aggregators:
        f_byz = 0
        global_delta = FED.aggregate(agg_name, deltas, f=f_byz)
        new_vec = g0 + global_delta
        new_state = FED.unflatten_params(new_vec, shapes, global_state)
        upd = ctor().to(DEVICE); upd.load_state_dict(new_state); upd = prepare_model(upd)
        f1 = f1_on_channel(chan, upd, tf, tele, cmds, cfg, labels)
        out["aggregators"][agg_name] = {
            "global_delta_norm": float(np.linalg.norm(global_delta)),
            "recall": f1["recall"], "n_detected": f1["n_detected"], "n_labels": f1["n_labels"],
            "update_finite": bool(np.isfinite(new_vec).all())}
    if do_theory:
        anchor_start = parts[0][0]
        x_anchor = features[anchor_start:anchor_start + cfg.l_s]
        vth = DP.v_theta_single_anchor(global_model, x_anchor, tau=0.1, device=DEVICE)
        out["d_adv_prime_single_anchor"] = vth
        anchors = np.stack([features[parts[s][0]:parts[s][0] + cfg.l_s]
                            for s in range(min(M, 20)) if parts[s][0] + cfg.l_s <= len(features)])
        if anchors.shape[0] >= 2:
            vst = DP.v_theta_stacked(global_model, anchors, tau=0.1, device=DEVICE)
            out["d_adv_prime_stacked"] = {"d_adv_prime_int": vst["d_adv_prime_int"],
                                          "d_adv_prime_effective_trace": vst["d_adv_prime_effective_trace"],
                                          "anchors": vst["anchors"]}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_fl_clean"))
    ap.add_argument("--data-dir", default=str(_ROOT / "smap_msl_data"))
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--channel", default="A-1")
    ap.add_argument("--n-planes", type=int, default=5)
    ap.add_argument("--sats-per-plane", type=int, default=8)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aggregators", nargs="*", default=["fedavg", "coord_median", "krum"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VendoredConfig(); q = Quantizer()
    con = CN.Constellation(walker=CN.WalkerDelta(n_planes=args.n_planes, sats_per_plane=args.sats_per_plane))
    cpath = out_dir / f"{args.channel}_clean_round.json"
    if cpath.exists() and not args.force:
        print(f"{args.channel}: cached"); return
    print(f"device={DEVICE}  channel={args.channel}  M={con.n_sats} sats  "
          f"aggregators={args.aggregators}  (ONE CLEAN ROUND)\n")
    t0 = time.time()
    rec = run_clean_round(args.channel, Path(args.data_dir), Path(args.runs_dir), cfg, q, con,
                          args.aggregators, args.local_epochs, args.lr, args.seed)
    json.dump(rec, open(cpath, "w"), indent=2)
    print(f"contact graph: {rec['contact_graph']['n_edges']} edges, "
          f"connected={rec['contact_graph']['connected']}, density={rec['contact_graph']['density']:.2f}")
    for agg, r in rec["aggregators"].items():
        print(f"  {agg:13}: global_delta_norm={r['global_delta_norm']:.4f}  recall={r['recall']:.3f} "
              f"({r['n_detected']}/{r['n_labels']})  update_finite={r['update_finite']}")
    if "d_adv_prime_single_anchor" in rec:
        sa = rec["d_adv_prime_single_anchor"]
        print(f"  d_adv' (single anchor) = {sa['d_adv_prime_int']} "
              f"(effective trace {sa['d_adv_prime_effective_trace']:.3f})")
        if "d_adv_prime_stacked" in rec:
            st = rec["d_adv_prime_stacked"]
            print(f"  d_adv' (stacked {st['anchors']} anchors) = {st['d_adv_prime_int']} "
                  f"(effective trace {st['d_adv_prime_effective_trace']:.3f})")
    print(f"\nclean round OK in {time.time()-t0:.0f}s -> {cpath}")

if __name__ == "__main__":
    main()