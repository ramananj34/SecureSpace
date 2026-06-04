import sys, time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from ldpc.dvb_s2_ldpc import build_dvbs2_h_long_rate12, RATE_1_2_LONG
from ldpc.tanner_graph import tanner_neighbors, compute_girth

print('Building H matrix...')
H = build_dvbs2_h_long_rate12()
print()
print('Building Tanner-graph adjacency lists...')
t0 = time.time()
vn_neighbors, cn_neighbors = tanner_neighbors(H)
print(f'Done in {time.time()-t0:.2f}s')
print(f'Variable nodes: {len(vn_neighbors)}')
print(f'Check nodes: {len(cn_neighbors)}')
print(f'Avg VN degree: {sum(len(nb) for nb in vn_neighbors) / len(vn_neighbors):.3f}')
print(f'Avg CN degree: {sum(len(nb) for nb in cn_neighbors) / len(cn_neighbors):.3f}')
total_vn_edges = sum(len(nb) for nb in vn_neighbors)
total_cn_edges = sum(len(nb) for nb in cn_neighbors)
print(f'Total edges via VNs: {total_vn_edges}')
print(f'Total edges via CNs: {total_cn_edges}')
print(f'H.nnz: {H.nnz}')
assert total_vn_edges == total_cn_edges == H.nnz
print()
print('Computing girth (sampled, 200 starts)...')
t0 = time.time()
girth, n_searched = compute_girth(H, cutoff=12, sample_size=200, rng_seed=0, verbose=True)
print(f'Done in {time.time()-t0:.1f}s')
print(f'Girth (lower bound from {n_searched} samples): {girth}')