from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc"),
           str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "ldpc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import weak_prng as W
from attribution import recover_score

np.seterr(over="ignore")

def build_perm_batch(gen, k, M, seed):
    return np.stack([gen(np.random.default_rng([seed, 3, i]).bytes(32), 0, int(k)) for i in range(int(M))]).astype(np.int64)

def q_vectors_from_batch(perms, weights, k, seed):
    invs = np.argsort(perms, axis=1)
    rng = np.random.default_rng([seed, 1])
    out = {}
    for w in weights:
        D = np.sort(rng.choice(k, int(w), replace=False))
        di = np.zeros(k, np.uint8); di[D] = 1
        out[int(w)] = di[invs].mean(0)
    return out

_FLOOR_CACHE = {}
def mc_floor(k, w, M, seed=0, reps=5):
    key = (int(k), int(w), int(M))
    if key in _FLOOR_CACHE:
        return _FLOOR_CACHE[key]
    vals = []
    for r in range(reps):
        rng = np.random.default_rng([seed, 100, r])
        perms = np.stack([rng.permutation(k) for _ in range(int(M))]).astype(np.int64)
        invs = np.argsort(perms, axis=1)
        D = np.sort(rng.choice(k, int(w), replace=False)); di = np.zeros(k, np.uint8); di[D] = 1
        q = di[invs].mean(0)
        vals.append(float(np.sum(np.abs(q - w / k)) / (2.0 * w)))
    f = float(np.mean(vals)); _FLOOR_CACHE[key] = f
    return f


def eps_from_q(q, k, w):
    dev = np.abs(q - w / k)
    return {"eps_v": float(dev.sum() / (2.0 * w)), "eps_v_max": float(dev.max())}

def eta_w_mock(w, k):
    return float(w) / float(k)

def run_generator(name, gen, k, weights, n_perm, n_keys, seed):
    rec = {"generator": name, "k": k, "weights": list(weights), "n_perm": n_perm, "per_w": []}
    ri = recover_score(gen, k, n_keys=n_keys, seed=seed)
    rec["recover_score"] = float(ri["recover_score"]); rec["n_distinct_perms"] = int(ri["n_distinct"])
    perms = build_perm_batch(gen, k, n_perm, seed)
    qs = q_vectors_from_batch(perms, weights, k, seed)
    for w in weights:
        q = qs[int(w)]
        adv_a = float(q.max())
        eps = eps_from_q(q, k, w)
        floor = mc_floor(k, w, n_perm, seed=seed)
        eta = eta_w_mock(w, k)
        bound_rhs = eta + eps["eps_v_max"]
        nu_above = max(0.0, eps["eps_v"] - floor)
        hi_rec, hi_nu = rec["recover_score"] > 0.30, nu_above > 0.10
        label = ("both" if hi_rec and hi_nu else "recovery" if hi_rec
                 else "nonuniformity" if hi_nu else "neither")
        rec["per_w"].append({
            "w": int(w), "eta_w": eta, "adv_a_worst_target": adv_a, "adv_b_oracle": 1.0,
            "eps_v": eps["eps_v"], "eps_v_max": eps["eps_v_max"], "eps_v_mc_floor": floor,
            "eps_v_above_floor": nu_above, "theorem2_rhs": bound_rhs,
            "theorem2_holds": bool(adv_a <= bound_rhs + 1e-9),
            "theorem2_slack": float(bound_rhs - adv_a), "route_label": label})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS / "runs_e13"))
    ap.add_argument("--k", type=int, default=800)
    ap.add_argument("--weights", nargs="*", type=int, default=[1, 8, 32, 128])
    ap.add_argument("--n-perm", type=int, default=2000, help="permutation batch size (M); 2000 -> q SD ~0.01")
    ap.add_argument("--n-keys", type=int, default=300, help="keys for the recovery probe")
    ap.add_argument("--generators", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ladder = W.ladder()
    if args.generators:
        want = set(args.generators); ladder = [(n, g) for n, g in ladder if n in want]
    print(f"E13 sweep | k={args.k} weights={args.weights} n_perm(M)={args.n_perm} "
          f"generators={len(ladder)}  (mock f=bit-s, eta_w=w/k exact)\n")
    agg = []
    for i, (name, gen) in enumerate(ladder, 1):
        gpath = out_dir / f"{name.replace('/', '_')}.json"
        if gpath.exists() and not args.force:
            print(f"[{i}/{len(ladder)}] {name}: cached, skipping"); continue
        t0 = time.time()
        try:
            rec = run_generator(name, gen, args.k, args.weights, args.n_perm, args.n_keys, args.seed)
        except Exception as e:
            print(f"[{i}/{len(ladder)}] {name}: FAILED {type(e).__name__}: {e}"); continue
        tmp = gpath.with_suffix(".json.tmp"); json.dump(rec, open(tmp, "w"), indent=2); tmp.replace(gpath)
        worst = min(rec["per_w"], key=lambda r: r["theorem2_slack"])
        holds_all = all(r["theorem2_holds"] for r in rec["per_w"])
        agg.append((name, worst["eps_v"], worst["adv_a_worst_target"], worst["route_label"], holds_all))
        print(f"[{i}/{len(ladder)}] {name:16} eps_v={worst['eps_v']:.3f} "
              f"Adv_a={worst['adv_a_worst_target']:.3f} (b_oracle=1.00) route={worst['route_label']:13} "
              f"Thm2={holds_all}   {time.time()-t0:.0f}s")
    if agg:
        n_hold = sum(1 for a in agg if a[4])
        print(f"\n=== E13 summary over {len(agg)} generators (worst-w per gen) ===")
        print(f"Theorem 2 (Adv_a <= eta_w + eps_v_max): holds {n_hold}/{len(agg)} generators\n")
        print("generator        eps_v   Adv_a   route")
        for name, eps, adva, lbl, ok in sorted(agg, key=lambda a: a[1]):
            print(f"  {name:16} {eps:.3f}   {adva:.3f}   {lbl}  {'' if ok else '<-- THM2 FAIL'}")
        print("\nExpect: strong+reduced-round-ChaCha at eps_v~floor/Adv_a~eta_w (robust); biased/local/trunc")
        print("rise in eps_v with Adv_a tracking eta_w+eps_v; identity at eps_v~1/Adv_a~1 (=oracle).")
    print("done")

if __name__ == "__main__":
    main()