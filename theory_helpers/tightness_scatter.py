from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent

def _worst_defended_rate(rec):
    out = {}
    for lab in rec.get("labels", []):
        key = tuple(lab.get("label", []))
        best = None
        for cfg, cr in lab.get("by_config", {}).items():
            v = cr.get("defended_rate_model")
            if v is not None:
                best = v if best is None else max(best, v)
        if best is not None:
            out[key] = float(best)
    return out

def _max_eta(rec):
    out = {}
    labs = rec.get("labels", rec.get("results", None))
    if labs is None and "label" in rec:
        labs = [rec]
    for lab in (labs or []):
        key = tuple(lab.get("label", []))
        m = None
        me = lab.get("max_eta")
        if isinstance(me, dict):
            m = me.get("max_eta_hat", me.get("eta_hat"))
        elif isinstance(me, (int, float)):
            m = float(me)
        if m is None and "max_eta_hat" in lab:
            m = lab["max_eta_hat"]
        if m is None:
            curve = lab.get("curve") or lab.get("eta_curve") or lab.get("per_w")
            if curve:
                vals = [c.get("eta", c.get("eta_hat")) for c in curve]
                vals = [v for v in vals if v is not None]
                m = max(vals) if vals else None
        if m is not None:
            out[key] = float(m)
    return out

def load(runs_e9, runs_e14):
    e9 = {}
    for jp in sorted(Path(runs_e9).glob("*.json")):
        rec = json.load(open(jp)); e9.setdefault(rec["chan"], {}).update(_worst_defended_rate(rec))
    e14 = {}
    for jp in sorted(Path(runs_e14).glob("*.json")):
        rec = json.load(open(jp))
        chan = rec.get("chan", jp.stem); e14.setdefault(chan, {}).update(_max_eta(rec))
    rows = []
    for chan in sorted(set(e9) | set(e14)):
        for key in sorted(set(e9.get(chan, {})) | set(e14.get(chan, {}))):
            adv = e9.get(chan, {}).get(key)
            eta = e14.get(chan, {}).get(key)
            if adv is not None and eta is not None:
                rows.append({"chan": chan, "label": list(key), "adv_measured": adv, "max_eta": eta,
                             "bound_holds": adv <= eta + 0.05})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-e9", default=str(_ROOT / "amrcc" / "runs_e9"))
    ap.add_argument("--runs-e14", default=str(_ROOT / "amrcc" / "runs_e14"))
    ap.add_argument("--out", default=str(_THIS / "tightness_scatter.png"))
    args = ap.parse_args()
    rows = load(args.runs_e9, args.runs_e14)
    if not rows:
        print("no paired (E9,E14) cells found; check --runs-e9/--runs-e14 paths and field names")
        return
    print(f"{'chan':6} {'label':>16} {'Adv_meas':>9} {'max_eta':>8} {'bound<=':>8}")
    n_hold = 0
    for r in rows:
        n_hold += int(r["bound_holds"])
        print(f"{r['chan']:6} {str(r['label']):>16} {r['adv_measured']:>9.3f} "
              f"{r['max_eta']:>8.3f} {'OK' if r['bound_holds'] else 'VIOLATED':>8}")
    adv = np.array([r["adv_measured"] for r in rows]); eta = np.array([r["max_eta"] for r in rows])
    print(f"\nTheorem 2 (Adv <= max_w eta_w): holds {n_hold}/{len(rows)} cells")
    print(f"tightness (Corollary 2.1, binary): mean |Adv - eta_at_or_below| ; median gap "
          f"eta-Adv = {np.median(eta - adv):.3f}  (small gap for small-eta cells = tight; large for D-1)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 5))
        lim = max(0.05, float(max(adv.max(), eta.max())) * 1.1)
        ax.fill_between([0, lim], [0, lim], [lim, lim], color="#2ecc71", alpha=0.08,
                        label="Adv ≤ max$_w$η$_w$ (Thm 2)")
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (tight, Cor 2.1)")
        for r in rows:
            d1 = (r["chan"] == "D-1")
            ax.scatter(r["max_eta"], r["adv_measured"],
                       c=("#c0392b" if d1 else "#2980b9"), s=(60 if d1 else 28),
                       marker=("D" if d1 else "o"), zorder=3,
                       label="D-1 (η$_w$-large exception)" if d1 else None)
        ax.set_xlabel(r"$\max_w \eta_w(f)$  (E14)")
        ax.set_ylabel(r"measured Adv (E9 Setting-(a), worst)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title("Theorem 2 tightness: measured Adv vs η$_w$ bound")
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.out, dpi=130)
        print(f"figure -> {args.out}")
    except Exception as e:
        print(f"(plot skipped: {type(e).__name__}: {e})")

if __name__ == "__main__":
    main()