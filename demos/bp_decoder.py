from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for _p in [str(_ROOT), str(_THIS), str(_ROOT / "ldpc"), str(_ROOT / "nullspace_attack_utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from ldpc_ops import LDPCCode
from scipy.sparse import csr_matrix

def load_short_code():
    return LDPCCode.dvbs2_short_rate12()

class _TannerGraph:
    def __init__(self, H: csr_matrix):
        H = H.tocsr()
        self.m, self.n = H.shape
        Hc = H.tocoo()
        order = np.lexsort((Hc.col, Hc.row))
        self.e_chk = Hc.row[order].astype(np.int64)
        self.e_var = Hc.col[order].astype(np.int64)
        self.n_edges = self.e_chk.size
        self.chk_starts = np.searchsorted(self.e_chk, np.arange(self.m)).astype(np.int64)
        self.chk_counts = np.diff(np.append(self.chk_starts, self.n_edges))
        vorder = np.argsort(self.e_var, kind="stable")
        self.vsort = vorder
        self.e_var_sorted = self.e_var[vorder]
        self.var_starts = np.searchsorted(self.e_var_sorted, np.arange(self.n))
        self.var_ends = np.searchsorted(self.e_var_sorted, np.arange(self.n), side="right")

def _min_sum_iter(tg, llr, Mcv):
    total_var = llr.copy()
    np.add.at(total_var, tg.e_var, Mcv)
    Mvc = total_var[tg.e_var] - Mcv
    sign = np.where(Mvc >= 0, 1.0, -1.0)
    absv = np.abs(Mvc)
    m, ne = tg.m, tg.n_edges
    seg = np.clip(tg.chk_starts, 0, ne - 1)
    counts = tg.chk_counts
    chk_sign = np.ones(m); np.multiply.at(chk_sign, tg.e_chk, sign)
    loo_sign = chk_sign[tg.e_chk] * sign
    red1 = np.minimum.reduceat(absv, seg)
    min1 = np.where(counts > 0, red1, np.inf)
    edge_min1 = min1[tg.e_chk]
    is_min = absv <= edge_min1 + 1e-15
    absv2 = np.where(is_min, np.inf, absv)
    red2 = np.minimum.reduceat(absv2, seg)
    min2 = np.where(counts > 0, red2, np.inf)
    loo_min = np.where(is_min, min2[tg.e_chk], min1[tg.e_chk])
    return loo_sign * loo_min

def bp_decode(code, llr, max_iter=50, damping=1.0):
    tg = getattr(code, "_bp_tanner", None)
    if tg is None:
        tg = _TannerGraph(code.H.tocsr())
        try: code._bp_tanner = tg
        except Exception: pass
    llr = np.asarray(llr, np.float64)
    Mcv = np.zeros(tg.n_edges)
    chat = (llr < 0).astype(np.uint8)
    for it in range(int(max_iter)):
        Mcv = _min_sum_iter(tg, llr, Mcv) * damping + Mcv * (1 - damping) if damping < 1 else _min_sum_iter(tg, llr, Mcv)
        total = llr.copy(); np.add.at(total, tg.e_var, Mcv)
        chat = (total < 0).astype(np.uint8)
        if code.is_codeword(chat):
            return chat, it + 1, True
    return chat, int(max_iter), False

def awgn_llr(codeword, sigma):
    tx = 1.0 - 2.0 * np.asarray(codeword, np.float64)
    rx = tx + np.random.default_rng().normal(0, sigma, size=tx.size)
    return 2.0 * rx / sigma ** 2, rx

def awgn_llr_seeded(codeword, sigma, rng):
    tx = 1.0 - 2.0 * np.asarray(codeword, np.float64)
    rx = tx + rng.normal(0, sigma, size=tx.size)
    return 2.0 * rx / sigma ** 2