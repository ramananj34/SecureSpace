from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

B4_HIGH_DEGREE_ROWS = [
    [   54,  9318, 14392, 27561, 26909, 10219,  2534,  8597],
    [   55,  7263,  4635,  2530, 28130,  3033, 23830,  3651],
    [   56, 24731, 23583, 26036, 17299,  5750,   792,  9169],
    [   57,  5811, 26154, 18653, 11551, 15447, 13685, 16264],
    [   58, 12610, 11347, 28768,  2792,  3174, 29371, 12997],
    [   59, 16789, 16018, 21449,  6165, 21202, 15850,  3186],
    [   60, 31016, 21449, 17618,  6213, 12166,  8334, 18212],
    [   61, 22836, 14213, 11327,  5896,   718, 11727,  9308],
    [   62,  2091, 24941, 29966, 23634,  9013, 15587,  5444],
    [   63, 22207,  3983, 16904, 28534, 21415, 27524, 25912],
    [   64, 25687,  4501, 22193, 14665, 14798, 16158,  5491],
    [   65,  4520, 17094, 23397,  4264, 22370, 16941, 21526],
    [   66, 10490,  6182, 32370,  9597, 30841, 25954,  2762],
    [   67, 22120, 22865, 29870, 15147, 13668, 14955, 19235],
    [   68,  6689, 18408, 18346,  9918, 25746,  5443, 20645],
    [   69, 29982, 12529, 13858,  4746, 30370, 10023, 24828],
    [   70,  1262, 28032, 29888, 13063, 24033, 21951,  7863],
    [   71,  6594, 29642, 31451, 14831,  9509,  9335, 31552],
    [   72,  1358,  6454, 16633, 20354, 24598,   624,  5265],
    [   73, 19529,   295, 18011,  3080, 13364,  8032, 15323],
    [   74, 11981,  1510,  7960, 21462,  9129, 11370, 25741],
    [   75,  9276, 29656,  4543, 30699, 20646, 21921, 28050],
    [   76, 15975, 25634,  5520, 31119, 13715, 21949, 19605],
    [   77, 18688,  4608, 31755, 30165, 13103, 10706, 29224],
    [   78, 21514, 23117, 12245, 26035, 31656, 25631, 30699],
    [   79,  9674, 24966, 31285, 29908, 17042, 24588, 31857],
    [   80, 21856, 27777, 29919, 27000, 14897, 11409,  7122],
    [   81, 29773, 23310,   263,  4877, 28622, 20545, 22092],
    [   82, 15605,  5651, 21864,  3967, 14419, 22757, 15896],
    [   83, 30145,  1759, 10139, 29223, 26086, 10556,  5098],
    [   84, 18815, 16575,  2936, 24457, 26738,  6030,   505],
    [   85, 30326, 22298, 27562, 20131, 26390,  6247, 24791],
    [   86,   928, 29246, 21246, 12400, 15311, 32309, 18608],
    [   87, 20314,  6025, 26689, 16302,  2296,  3244, 19613],
    [   88,  6237, 11943, 22851, 15642, 23857, 15112, 20947],
    [   89, 26403, 25168, 19038, 18384,  8882, 12719,  7093],
]

B4_LOW_DEGREE_ROWS = [
    [    0, 14567, 24965],
    [    1,  3908,   100],
    [    2, 10279,   240],
    [    3, 24102,   764],
    [    4, 12383,  4173],
    [    5, 13861, 15918],
    [    6, 21327,  1046],
    [    7,  5288, 14579],
    [    8, 28158,  8069],
    [    9, 16583, 11098],
    [   10, 16681, 28363],
    [   11, 13980, 24725],
    [   12, 32169, 17989],
    [   13, 10907,  2767],
    [   14, 21557,  3818],
    [   15, 26676, 12422],
    [   16,  7676,  8754],
    [   17, 14905, 20232],
    [   18, 15719, 24646],
    [   19, 31942,  8589],
    [   20, 19978, 27197],
    [   21, 27060, 15071],
    [   22,  6071, 26649],
    [   23, 10393, 11176],
    [   24,  9597, 13370],
    [   25,  7081, 17677],
    [   26,  1433, 19513],
    [   27, 26925,  9014],
    [   28, 19202,  8900],
    [   29, 18152, 30647],
    [   30, 20803,  1737],
    [   31, 11804, 25221],
    [   32, 31683, 17783],
    [   33, 29694,  9345],
    [   34, 12280, 26611],
    [   35,  6526, 26122],
    [   36, 26165, 11241],
    [   37,  7666, 26962],
    [   38, 16290,  8480],
    [   39, 11774, 10120],
    [   40, 30051, 30426],
    [   41,  1335, 15424],
    [   42,  6865, 17742],
    [   43, 31779, 12489],
    [   44, 32120, 21001],
    [   45, 14508,  6996],
    [   46,   979, 25024],
    [   47,  4554, 21896],
    [   48,  7989, 21777],
    [   49,  4972, 20661],
    [   50,  6612,  2730],
    [   51, 12742,  4418],
    [   52, 29194,   595],
    [   53, 19267, 20113],
]

@dataclass(frozen=True)
class DVBS2Params:
    rate: str
    frame: str
    n: int
    k: int
    M: int = 360
    @property
    def n_minus_k(self) -> int:
        return self.n - self.k
    @property
    def q(self) -> int:
        return self.n_minus_k // self.M
RATE_1_2_LONG = DVBS2Params(rate="1/2", frame="long", n=64800, k=32400)

def build_dvbs2_h_long_rate12() -> csr_matrix:
    params = RATE_1_2_LONG
    return _build_h_from_tables(high_rows=B4_HIGH_DEGREE_ROWS, low_rows=B4_LOW_DEGREE_ROWS, params=params)

def _build_h_from_tables(high_rows: list[list[int]], low_rows: list[list[int]], params: DVBS2Params) -> csr_matrix:
    n = params.n
    k = params.k
    n_minus_k = params.n_minus_k
    M = params.M
    q = params.q
    assert n_minus_k % M == 0, f"(n-k) must be divisible by M={M}"
    all_rows = high_rows + low_rows
    expected_rows = k // M
    assert len(all_rows) == expected_rows, (f"Expected {expected_rows} table rows, got {len(all_rows)}")
    H = lil_matrix((n_minus_k, n), dtype=np.int8)
    for m in range(k):
        row = m // M
        col = m % M
        for x in all_rows[row]:
            check_idx = (x + col * q) % n_minus_k
            H[check_idx, m] = 1
    for j in range(n_minus_k):
        H[j, k + j] = 1
    for j in range(1, n_minus_k):
        H[j, k + j - 1] = 1
    return H.tocsr()

def verify_h_matrix(H: csr_matrix, params: DVBS2Params, expected_total_ones: int | None = None) -> dict:
    n_minus_k_actual, n_actual = H.shape
    info: dict = {"shape": (n_minus_k_actual, n_actual), "expected_shape": (params.n_minus_k, params.n)}
    assert (n_minus_k_actual, n_actual) == (params.n_minus_k, params.n), (f"Shape mismatch: {(n_minus_k_actual, n_actual)} vs " f"{(params.n_minus_k, params.n)}")
    nnz = H.nnz
    info["nnz"] = int(nnz)
    if expected_total_ones is not None:
        assert nnz == expected_total_ones, (f"Total nonzeros: {nnz} vs expected {expected_total_ones}")
        info["expected_nnz"] = expected_total_ones
    row_weights = np.asarray(H.sum(axis=1)).flatten()
    info["row_weight_min"] = int(row_weights.min())
    info["row_weight_max"] = int(row_weights.max())
    info["row_weight_unique"] = sorted(set(row_weights.tolist()))
    info["row_weight_distribution"] = {int(w): int(np.sum(row_weights == w)) for w in info["row_weight_unique"]}
    col_weights = np.asarray(H.sum(axis=0)).flatten()
    info["col_weight_min"] = int(col_weights.min())
    info["col_weight_max"] = int(col_weights.max())
    info["col_weight_unique"] = sorted(set(col_weights.tolist()))
    info["col_weight_distribution"] = {int(w): int(np.sum(col_weights == w)) for w in info["col_weight_unique"]}
    info["density"] = float(nnz) / (n_minus_k_actual * n_actual)
    assert col_weights.min() >= 1, "Some column has zero weight"
    assert row_weights.min() >= 1, "Some row has zero weight"
    return info

def verify_dvbs2_long_rate12_specifics(H: csr_matrix) -> dict:
    expected_nnz = 226799
    info = verify_h_matrix(H, RATE_1_2_LONG, expected_total_ones=expected_nnz)
    rwd = info["row_weight_distribution"]
    assert rwd == {6: 1, 7: 32399}, f"Row weight distribution mismatch: {rwd}"
    cwd = info["col_weight_distribution"]
    expected_cwd = {1: 1, 2: 32399, 3: 19440, 8: 12960}
    assert cwd == expected_cwd, (f"Column weight distribution mismatch:\n  got: {cwd}\n  exp: {expected_cwd}")
    total_from_cols = 1*1 + 2*32399 + 3*19440 + 8*12960
    assert total_from_cols == 226799
    info["proposal_consistency"] = {"k_LDPC matches proposal §6.7": RATE_1_2_LONG.k == 32400, "n_LDPC matches proposal §6.7": RATE_1_2_LONG.n == 64800, "rate matches proposal §6.7": RATE_1_2_LONG.k / RATE_1_2_LONG.n == 0.5}
    return info