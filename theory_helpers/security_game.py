from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "amrcc"),
           str(_ROOT / "nullspace_attack_utils"), str(_ROOT / "nullspace_attack"),
           str(_ROOT / "ldpc")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from keyed_permutation import permutation_for_frame, apply_perm, apply_inverse_perm
from sampling import selection_sample, clopper_pearson

def default_perm_gen(key: bytes, t: int, k: int) -> np.ndarray:
    return permutation_for_frame(key, int(t), int(k))

def default_code():
    from ldpc_ops import LDPCCode
    return LDPCCode.dvbs2_long_rate12()

def enc(x_bits, key, t, code, perm_gen=default_perm_gen):
    x = np.asarray(x_bits, np.uint8).reshape(-1)
    if x.size != code.k:
        raise ValueError(f"x has {x.size} bits, expected code.k={code.k}")
    perm = perm_gen(key, t, code.k)
    m = apply_perm(x, perm).astype(np.uint8)
    return np.asarray(code.encode(m)).astype(np.uint8)

def dec(codeword, key, t, code, perm_gen=default_perm_gen):
    perm = perm_gen(key, t, code.k)
    m_hat = np.asarray(code.info_bit_extract(codeword)).astype(np.uint8)
    return apply_inverse_perm(m_hat, perm).astype(np.uint8)


def delta_star_from_info(delta_info, code):
    di = np.asarray(delta_info, np.uint8).reshape(-1)
    if di.size != code.k:
        raise ValueError(f"delta_info has {di.size} bits, expected k={code.k}")
    return np.asarray(code.encode(di)).astype(np.uint8)

def key_gen(rng) -> bytes:
    return rng.bytes(32)

@dataclass
class EncOracle:
    key: bytes
    code: object
    perm_gen: object = default_perm_gen
    queried_counters: set = field(default_factory=set)
    n_queries: int = 0

    def query(self, t: int, x_bits) -> np.ndarray:
        self.queried_counters.add(int(t))
        self.n_queries += 1
        return enc(x_bits, self.key, int(t), self.code, self.perm_gen)

def play_challenge(oracle: EncOracle, t_star, x_star, y_star, delta_star, r, decision_fn, verify_syndrome=True):
    code = oracle.code
    x_star = np.asarray(x_star, np.uint8).reshape(-1)
    delta_star = np.asarray(delta_star, np.uint8).reshape(-1)
    out = {"t_star": int(t_star), "r": int(r)}
    a_ok = int(t_star) not in oracle.queried_counters
    w_star = int(delta_star.sum())
    b_ok = (w_star <= int(r))
    f_clean = decision_fn(x_star)
    c_ok = (y_star != f_clean)
    out.update({"constraint_a_tstar_fresh": bool(a_ok), "constraint_b_weight_ok": bool(b_ok),
                "constraint_c_target_differs": bool(c_ok),
                "delta_star_weight": w_star, "f_clean": _lab(f_clean)})
    if not (a_ok and b_ok and c_ok):
        out.update({"valid_challenge": False, "win": False,
                    "reason": "constraint " +
                              ("a " if not a_ok else "") + ("b " if not b_ok else "") +
                              ("c" if not c_ok else "")})
        return out
    out["valid_challenge"] = True

    c_star = enc(x_star, oracle.key, int(t_star), code, oracle.perm_gen)
    c_hat = (c_star ^ delta_star).astype(np.uint8)

    synd = np.asarray(code.syndrome(c_hat)).astype(np.int64)
    win_i = (int(synd.sum()) == 0)
    out["win_syndrome_zero"] = bool(win_i)
    if not win_i:
        out.update({"win": False, "reason": "syndrome nonzero (delta* not in N(H))",
                    "syndrome_weight": int(synd.sum())})
        return out

    x_rec = dec(c_hat, oracle.key, int(t_star), code, oracle.perm_gen)
    f_rec = decision_fn(x_rec)
    win_ii = (f_rec == y_star)
    out.update({"win_decision_equals_target": bool(win_ii), "f_recovered": _lab(f_rec),
                "win": bool(win_i and win_ii),
                "w_info": int(np.asarray(x_rec ^ x_star, np.uint8).sum())})
    return out

def _lab(y):
    try:
        return int(y)
    except Exception:
        return str(y)

def setting_a_delta_info(k, w_info, rng):
    di = np.zeros(int(k), np.uint8)
    if w_info > 0:
        di[selection_sample(int(k), int(w_info), rng)] = 1
    return di

def setting_b_delta_info(target_v, key, t_star, code, perm_gen=default_perm_gen):
    tv = np.asarray(target_v, np.uint8).reshape(-1)
    perm = perm_gen(key, int(t_star), code.k)
    return apply_perm(tv, perm).astype(np.uint8)

def measure_setting_a_win_rate(x_star, y_star, w_info, code, decision_fn, n_trials=500, seed=0, perm_gen=default_perm_gen, alpha=0.05):
    x_star = np.asarray(x_star, np.uint8).reshape(-1)
    k = code.k
    rng = np.random.default_rng(seed)
    di = setting_a_delta_info(k, w_info, rng)
    wins, wslots = 0, []
    for i in range(int(n_trials)):
        key = np.random.default_rng([seed, 1, i]).bytes(32)
        perm = perm_gen(key, 0, k)
        v = apply_inverse_perm(di, perm)
        x_rec = (x_star ^ v).astype(np.uint8)
        if decision_fn(x_rec) == y_star:
            wins += 1
        wslots.append(int(v.sum()))
    lo, hi = clopper_pearson(wins, int(n_trials), alpha)
    return {"w_info": int(w_info), "n_trials": int(n_trials), "wins": int(wins),
            "win_rate": wins / float(n_trials), "ci_lo": float(lo), "ci_hi": float(hi),
            "v_weight_mean": float(np.mean(wslots)) if wslots else 0.0}

def compare_settings(x_star, y_star, w_info, code, decision_fn, target_v=None, n_trials_a=300, seed=0, perm_gen=default_perm_gen):
    a = measure_setting_a_win_rate(x_star, y_star, w_info, code, decision_fn, n_trials=n_trials_a, seed=seed, perm_gen=perm_gen)
    x_star = np.asarray(x_star, np.uint8).reshape(-1)
    if target_v is None:
        target_v = _find_flipping_v(x_star, y_star, w_info, decision_fn, seed)
    b_wins, N = 0, n_trials_a
    for i in range(N):
        key = np.random.default_rng([seed, 2, i]).bytes(32)
        di_b = setting_b_delta_info(target_v, key, 0, code, perm_gen)
        v = apply_inverse_perm(di_b, perm_gen(key, 0, code.k))
        if decision_fn((x_star ^ v).astype(np.uint8)) == y_star:
            b_wins += 1
    return {"setting_a": a, "setting_b_win_rate": b_wins / float(N),
            "b_ge_a": (b_wins / float(N)) >= a["win_rate"] - 1e-9}

def _find_flipping_v(x_star, y_star, w_info, decision_fn, seed, tries=200):
    k = x_star.size
    rng = np.random.default_rng([seed, 3])
    for _ in range(tries):
        v = np.zeros(k, np.uint8)
        if w_info > 0:
            v[selection_sample(k, w_info, rng)] = 1
        if decision_fn((x_star ^ v).astype(np.uint8)) == y_star:
            return v
    return v