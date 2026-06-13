from __future__ import annotations
import sys
import json
import time
import argparse
import platform
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import nullspace_attack_utils.gf2 as gf2
from secret_integrity import (SecretIntegrity, nwords, basis_vec_packed, random_matrix_packed, recover_readout, recover_inverse, witnesses_at, nonuniqueness_log2_count, qpochhammer_span_prob)

SEED = 20250612
N, M = 64800, 16200

def _save(out_dir: Path, name: str, rec: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{name}.json.tmp"
    json.dump(rec, open(tmp, "w"), indent=2, default=float)
    tmp.replace(out_dir / f"{name}.json")

def e7a_deterministic_boundary(n: int, m: int, seed: int, witness_r=(1, 3, 7), do_rank: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    s = SecretIntegrity(n, m, seed)
    rec = {"experiment": "E7a_deterministic_boundary", "n": n, "m": m, "seed": seed, "H_packed_MB": round(s.H_packed.nbytes / 1e6, 1)}
    if verbose:
        print(f"[E7a] H' {m}x{n}  ({rec['H_packed_MB']} MB packed)")
    if do_rank:
        tr = time.time()
        frr = s.full_row_rank()
        rec["full_row_rank"] = bool(frr)
        rec["rank_seconds"] = round(time.time() - tr, 1)
        if verbose:
            print(f"  full_row_rank(H')={frr}  ({rec['rank_seconds']:.1f}s)")
        assert frr, "H' is not full row rank -- regenerate with another seed"
    else:
        rec["full_row_rank"] = None
    tg = time.time()
    Hrec = recover_readout(s, range(n))
    exact = bool(np.array_equal(Hrec, s.H_packed))
    rec["readout_seconds"] = round(time.time() - tg, 2)
    rec["recover_exact_at_t_eq_n"] = exact
    if verbose:
        print(f"  recover_readout exact @ t=n : {exact}  ({rec['readout_seconds']:.2f}s)")
    assert exact, "deterministic read-out did not reproduce H' -- bug"
    wlist = []
    for r in witness_r:
        t = n - r
        A, B, observed, log2c = witnesses_at(s, t, alter_index=n - 1)
        sample = [0, n // 2, t - 1] if t >= 2 else [0]
        indist = all(np.array_equal(gf2.gf2_matvec(A, basis_vec_packed(i, n)), gf2.gf2_matvec(B, basis_vec_packed(i, n))) for i in sample)
        eu = basis_vec_packed(n - 1, n)
        distinct = not np.array_equal(gf2.gf2_matvec(A, eu), gf2.gf2_matvec(B, eu))
        wlist.append({"t": t, "r": r, "log2_consistent_matrices": log2c, "indistinguishable_on_observed_sample": bool(indist), "distinct_on_unobserved_column": bool(distinct)})
        if verbose:
            print(f"  t=n-{r}: 2^{log2c} consistent matrices; "
                  f"indistinguishable_on_observed={indist}, distinct={distinct}")
        assert indist and distinct
    rec["nonuniqueness_witnesses"] = wlist
    rec["seconds_total"] = round(time.time() - t0, 1)
    return rec

def e7b_qpochhammer_sweep(n_list, trials: int, cmax: int, seed: int, recovery_spotcheck: int = 25, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    rec = {"experiment": "E7b_qpochhammer_sweep", "trials": trials, "seed": seed, "qpochhammer_constant": qpochhammer_span_prob(64800, 64800), "curves": {}}
    for n in n_list:
        s = SecretIntegrity(n, max(1, n // 4), seed + n)
        pts = []
        for c in range(-2, cmax + 1):
            t = n + c
            if t < n:
                pts.append({"c": c, "t": t, "p_emp": 0.0, "p_exact": 0.0, "n_trials": 0})
                continue
            spans, spot_ok, spot_n = 0, True, 0
            for j in range(trials):
                X = random_matrix_packed(n, t, rng)
                full = gf2.rank(X, t) == n
                spans += int(full)
                if full and spot_n < recovery_spotcheck and (j % 7 == 0):
                    from secret_integrity import recover_general
                    Hrec, ok = recover_general(s, X, t)
                    spot_n += 1
                    spot_ok &= bool(ok and np.array_equal(Hrec, s.H_packed))
            pts.append({"c": c, "t": t, "p_emp": spans / trials, "p_exact": qpochhammer_span_prob(n, t), "n_trials": trials, "recovery_exact_when_span": spot_ok})
        rec["curves"][str(n)] = pts
        if verbose:
            at0 = next(p for p in pts if p["c"] == 0)
            at7 = next(p for p in pts if p["c"] == 7)
            print(f"  n'={n:5d}: P(span)@t=n' emp {at0['p_emp']:.3f} "
                  f"(exact {at0['p_exact']:.3f});  @t=n'+7 emp {at7['p_emp']:.4f} "
                  f"(exact {at7['p_exact']:.4f})")
    return rec

def e7_prod_confirm(n: int, seed: int, x_mode: str = "invertible", max_tries: int = 8, verbose: bool = True) -> dict:
    from secret_integrity import random_unit_lower_triangular_packed
    m = max(1, n // 4)
    s = SecretIntegrity(n, m, seed)
    rng = np.random.default_rng(seed + 1)
    rec = {"experiment": "E7_prod_confirm", "n": n, "m": m, "seed": seed, "x_mode": x_mode, "X_packed_MB": None, "tries": 0}
    t0 = time.time()
    if x_mode == "invertible":
        X = random_unit_lower_triangular_packed(n, rng)
        rec["X_packed_MB"] = round(X.nbytes / 1e6, 1)
        rec["tries"] = 1
        if verbose:
            print(f"[E7-prod] n={n}  X(unit-lower-triangular, guaranteed invertible) {rec['X_packed_MB']} MB  (inverse + S X^-1, ~O(n^3); one pass)...")
        ti = time.time()
        Hrec, ok = recover_inverse(s, X, n)
        assert ok, "unit-lower-triangular X was not invertible -- impossible"
        rec["recover_seconds"] = round(time.time() - ti, 1)
        rec["recover_exact"] = bool(np.array_equal(Hrec, s.H_packed))
        if verbose:
            print(f"  recovered exact={rec['recover_exact']}  ({rec['recover_seconds']:.1f}s)")
        assert rec["recover_exact"]
    else:
        for attempt in range(max_tries):
            X = random_matrix_packed(n, n, rng)
            rec["X_packed_MB"] = round(X.nbytes / 1e6, 1)
            if verbose:
                print(f"[E7-prod] n={n}  X(random) {rec['X_packed_MB']} MB  attempt {attempt+1} (~O(n^3); invertible w.p. ~0.29)...")
            ti = time.time()
            Hrec, ok = recover_inverse(s, X, n)
            rec["tries"] = attempt + 1
            if ok:
                rec["recover_seconds"] = round(time.time() - ti, 1)
                rec["recover_exact"] = bool(np.array_equal(Hrec, s.H_packed))
                assert rec["recover_exact"]
                break
            elif verbose:
                print("  X singular (did not span); resampling")
    rec["seconds_total"] = round(time.time() - t0, 1)
    return rec

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--out", default=str(_THIS / "runs_e7"))
    ap.add_argument("--seed", type=int, default=SEED)
    a = sub.add_parser("e7a")
    a.add_argument("--n", type=int, default=N)
    a.add_argument("--m", type=int, default=M)
    a.add_argument("--rank", dest="rank", action="store_true", default=True)
    a.add_argument("--no-rank", dest="rank", action="store_false")
    b = sub.add_parser("e7b")
    b.add_argument("--n-list", type=int, nargs="+", default=[256, 512, 1024, 2048])
    b.add_argument("--trials", type=int, default=1000)
    b.add_argument("--cmax", type=int, default=25)
    p = sub.add_parser("prod")
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--x-mode", choices=["invertible", "random"], default="invertible")
    args = ap.parse_args()
    out_dir = Path(args.out)
    print(f"platform: {platform.platform()}  |  numpy {np.__version__}\n")
    if args.cmd == "e7a":
        rec = e7a_deterministic_boundary(args.n, args.m, args.seed, do_rank=args.rank)
        _save(out_dir, f"e7a_n{args.n}", rec)
    elif args.cmd == "e7b":
        rec = e7b_qpochhammer_sweep(args.n_list, args.trials, args.cmax, args.seed)
        _save(out_dir, "e7b_qpochhammer", rec)
    elif args.cmd == "prod":
        rec = e7_prod_confirm(args.n, args.seed, x_mode=args.x_mode)
        _save(out_dir, f"e7_prod_n{args.n}", rec)
    print(f"\nsaved -> {out_dir}")

if __name__ == "__main__":
    main()