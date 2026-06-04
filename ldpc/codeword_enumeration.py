from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

def random_codeword_weights(encoder, k: int, n_trials: int = 100, weight_range: tuple[int, int] | None = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    weights: list[int] = []
    low_weight_examples: list[tuple[int, np.ndarray]] = []
    for _ in range(n_trials):
        info = rng.integers(0, 2, size=k, dtype=np.int8)
        cw = encoder(info)
        w = int(cw.sum())
        weights.append(w)
        if weight_range is not None and weight_range[0] <= w <= weight_range[1]:
            if len(low_weight_examples) < 10:
                low_weight_examples.append((w, info.copy()))
    weights_arr = np.array(weights)
    return {"n_trials": n_trials, "min_weight": int(weights_arr.min()), "max_weight": int(weights_arr.max()), "mean_weight": float(weights_arr.mean()), "median_weight": float(np.median(weights_arr)), "weight_counter": Counter(weights), "low_weight_examples": low_weight_examples}

def low_weight_codeword_search_unit_info(encoder, k: int, n_examples: int | None = None) -> list[tuple[int, int]]:
    if n_examples is None:
        n_examples = k
    results: list[tuple[int, int]] = []
    for i in range(n_examples):
        info = np.zeros(k, dtype=np.int8)
        info[i] = 1
        cw = encoder(info)
        results.append((i, int(cw.sum())))
    return results

def low_weight_codeword_search_pair_info(encoder, k: int, n_pairs: int = 1000, seed: int = 0) -> list[tuple[tuple[int, int], int]]:
    rng = np.random.default_rng(seed)
    results: list[tuple[tuple[int, int], int]] = []
    for _ in range(n_pairs):
        i, j = rng.choice(k, size=2, replace=False)
        info = np.zeros(k, dtype=np.int8)
        info[i] = 1
        info[j] = 1
        cw = encoder(info)
        results.append(((int(i), int(j)), int(cw.sum())))
    results.sort(key=lambda x: x[1])
    return results