from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SAConfig:
    iters: int = 20000
    restarts: int = 1
    T0: float = 1.0
    T1: float = 1e-3
    seg_move_frac: float = 0.3
    max_seg: int = 32
    seed: int = 0
    anneal: str = "geometric"
    init: str = "identity"
    track_every: int = 0

@dataclass
class SAResult:
    pi_hat: np.ndarray
    fitness: float
    n_iters: int
    n_accept: int
    trace: list = field(default_factory=list)

def _temps(cfg):
    if cfg.anneal == "linear":
        return np.linspace(cfg.T0, cfg.T1, cfg.iters)
    r = (cfg.T1 / cfg.T0) ** (1.0 / max(cfg.iters - 1, 1))
    return cfg.T0 * (r ** np.arange(cfg.iters))

def anneal(k, fitness_full, cfg: SAConfig, delta_swap=None, init_perm=None, rng=None):
    rng = rng or np.random.default_rng(cfg.seed)
    if init_perm is not None:
        cur = np.asarray(init_perm, np.int64).copy()
    elif cfg.init == "random":
        cur = rng.permutation(k).astype(np.int64)
    else:
        cur = np.arange(k, dtype=np.int64)
    cur_f = float(fitness_full(cur))
    best, best_f = cur.copy(), cur_f
    Ts = _temps(cfg)
    n_accept = 0
    trace = []
    for it in range(cfg.iters):
        T = Ts[it]
        use_seg = (rng.random() < cfg.seg_move_frac) or (delta_swap is None and rng.random() < 0.5)
        if use_seg:
            L = int(rng.integers(2, cfg.max_seg + 1))
            s = int(rng.integers(0, k - L + 1))
            cand = cur.copy()
            cand[s:s + L] = cand[s:s + L][::-1]
            cand_f = float(fitness_full(cand))
            df = cand_f - cur_f
            if df >= 0 or rng.random() < np.exp(df / max(T, 1e-12)):
                cur, cur_f = cand, cand_f
                n_accept += 1
                if cur_f > best_f:
                    best, best_f = cur.copy(), cur_f
        else:
            i, j = rng.integers(0, k), rng.integers(0, k)
            if i == j:
                continue
            if delta_swap is not None:
                df = float(delta_swap(cur, int(i), int(j)))
                if df >= 0 or rng.random() < np.exp(df / max(T, 1e-12)):
                    cur[i], cur[j] = cur[j], cur[i]
                    cur_f += df
                    n_accept += 1
                    if cur_f > best_f:
                        best, best_f = cur.copy(), cur_f
            else:
                cand = cur.copy(); cand[i], cand[j] = cand[j], cand[i]
                cand_f = float(fitness_full(cand)); df = cand_f - cur_f
                if df >= 0 or rng.random() < np.exp(df / max(T, 1e-12)):
                    cur, cur_f = cand, cand_f
                    n_accept += 1
                    if cur_f > best_f:
                        best, best_f = cur.copy(), cur_f
        if cfg.track_every and (it % cfg.track_every == 0):
            trace.append((it, best_f))
    return SAResult(pi_hat=best, fitness=best_f, n_iters=cfg.iters, n_accept=n_accept, trace=trace)

def anneal_multi(k, fitness_full, cfg: SAConfig, delta_swap=None, init_perm=None):
    best = None
    for r in range(cfg.restarts):
        rng = np.random.default_rng([cfg.seed, r])
        ip = init_perm if (r == 0 and init_perm is not None) else None
        if ip is None and cfg.init == "random":
            ip = rng.permutation(k).astype(np.int64)
        res = anneal(k, fitness_full, cfg, delta_swap=delta_swap, init_perm=ip, rng=rng)
        if best is None or res.fitness > best.fitness:
            best = res
    return best