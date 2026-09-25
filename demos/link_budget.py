from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "ldpc"), str(_ROOT / "nullspace_attack_utils"),
           str(_ROOT / "amrcc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bp_decoder as BP
from sampling import clopper_pearson

def ebn0_to_sigma(ebn0_db, rate):
    ebn0 = 10 ** (ebn0_db / 10.0)
    esn0 = ebn0 * rate
    return float(np.sqrt(1.0 / (2.0 * esn0)))


def _permuted_message(info, code, key_seed, rng):
    perm = rng.permutation(code.k)
    return np.asarray(info, np.uint8)[perm]

def run_curve(code, ebn0_db_list, n_trials, use_perm=False, max_iter=50, seed=0):
    rate = code.k / code.n
    rows = []
    for ebn0_db in ebn0_db_list:
        sigma = ebn0_to_sigma(ebn0_db, rate)
        rng = np.random.default_rng([seed, int(ebn0_db * 100)])
        bit_errs = 0; frame_errs = 0; n_bits = 0
        for _ in range(int(n_trials)):
            info = (rng.random(code.k) < 0.5).astype(np.uint8)
            msg = _permuted_message(info, code, seed, rng) if use_perm else info
            c = np.asarray(code.encode(msg)).astype(np.uint8)
            llr = BP.awgn_llr_seeded(c, sigma, rng)
            chat, iters, ok = BP.bp_decode(code, llr, max_iter=max_iter)
            msg_hat = code.info_bit_extract(chat)
            be = int(np.sum(msg_hat != msg))
            bit_errs += be; n_bits += code.k
            if be > 0 or not ok:
                frame_errs += 1
        ber = bit_errs / n_bits
        bler = frame_errs / n_trials
        lo, hi = clopper_pearson(frame_errs, int(n_trials), 0.05)
        rows.append({"ebn0_db": float(ebn0_db), "sigma": sigma, "ber": float(ber), "bler": float(bler),
                     "bler_ci_lo": float(lo), "bler_ci_hi": float(hi),
                     "frame_errs": int(frame_errs), "n_trials": int(n_trials)})
    return {"use_perm": bool(use_perm), "rate": float(rate), "rows": rows}

def compare_curves(baseline, amrcc):
    rows = []
    for b, a in zip(baseline["rows"], amrcc["rows"]):
        overlap = not (b["bler_ci_hi"] < a["bler_ci_lo"] or a["bler_ci_hi"] < b["bler_ci_lo"])
        rows.append({"ebn0_db": b["ebn0_db"], "baseline_bler": b["bler"], "amrcc_bler": a["bler"],
                     "ci_overlap": bool(overlap)})
    return {"rows": rows, "all_overlap": bool(all(r["ci_overlap"] for r in rows)),
            "verdict": ("AMRCC BLER == baseline BLER within CI at all Eb/N0 -> 0 dB link overhead "
                        "(permutation channel-transparent; confirms Sec-6.11 <0.1 dB)."
                        if all(r["ci_overlap"] for r in rows) else
                        "curves diverge -- investigate (should not happen for a message permutation).")}

def eps_dec_at(code, ebn0_db_operating, n_trials, max_iter=50, seed=0):
    curve = run_curve(code, [ebn0_db_operating], n_trials, use_perm=True, max_iter=max_iter, seed=seed)
    r = curve["rows"][0]
    return {"ebn0_db": ebn0_db_operating, "eps_dec": r["bler"], "ci_lo": r["bler_ci_lo"],
            "ci_hi": r["bler_ci_hi"], "frame_errs": r["frame_errs"], "n_trials": r["n_trials"],
            "note": "eps_dec = BP decoder BLER at operating SNR; validates the project-wide eps_dec~0 "
                    "idealization (algebraic decode used eps_dec=0)."}