from __future__ import annotations
from collections import deque
from typing import Iterable
import numpy as np
from scipy.sparse import csr_matrix

def tanner_neighbors(H: csr_matrix) -> tuple[list[list[int]], list[list[int]]]:
    m, n = H.shape
    H_csr = H.tocsr()
    H_csc = H.tocsc()
    cn_neighbors: list[list[int]] = [[] for _ in range(m)]
    for c in range(m):
        cn_neighbors[c] = H_csr.indices[H_csr.indptr[c] : H_csr.indptr[c + 1]].tolist()
    vn_neighbors: list[list[int]] = [[] for _ in range(n)]
    for v in range(n):
        vn_neighbors[v] = H_csc.indices[H_csc.indptr[v] : H_csc.indptr[v + 1]].tolist()
    return vn_neighbors, cn_neighbors

def _shortest_cycle_through_node(start_v: int, vn_neighbors: list[list[int]], cn_neighbors: list[list[int]], n_variables: int, cutoff: int = 16) -> int | None:
    INF = 10**9
    n_total = n_variables + len(cn_neighbors)
    dist = {start_v: 0}
    parent = {start_v: -1}
    best = INF
    queue = deque([start_v])
    while queue:
        u = queue.popleft()
        d_u = dist[u]
        if d_u >= cutoff // 2:
            continue
        if u < n_variables:
            nbrs = (cn + n_variables for cn in vn_neighbors[u])
        else:
            nbrs = cn_neighbors[u - n_variables]
        for w in nbrs:
            if w == parent.get(u, -2):
                continue
            if w in dist:
                cyc = dist[u] + dist[w] + 1
                if cyc < best:
                    best = cyc
            else:
                dist[w] = d_u + 1
                parent[w] = u
                queue.append(w)
    return None if best == INF else best

def compute_girth(H: csr_matrix, cutoff: int = 16, sample_size: int | None = None, rng_seed: int = 0, verbose: bool = False) -> tuple[int | None, int]:
    m, n = H.shape
    vn_neighbors, cn_neighbors = tanner_neighbors(H)
    if sample_size is None:
        starts = list(range(n))
    else:
        rng = np.random.default_rng(rng_seed)
        starts = rng.choice(n, size=min(sample_size, n), replace=False).tolist()
    best_girth = None
    progress_every = max(1, len(starts) // 20)
    for i, v in enumerate(starts):
        c = _shortest_cycle_through_node(v, vn_neighbors, cn_neighbors, n_variables=n, cutoff=cutoff)
        if c is not None:
            if best_girth is None or c < best_girth:
                best_girth = c
                if verbose:
                    print(f"  [{i+1}/{len(starts)}] new best girth: {best_girth}")
                if best_girth <= 4:
                    return best_girth, i + 1
        if verbose and (i + 1) % progress_every == 0:
            print(f"  [{i+1}/{len(starts)}] best girth so far: {best_girth}")
    return best_girth, len(starts)