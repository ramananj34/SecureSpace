import sys, time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from ldpc.dvb_s2_ldpc import build_dvbs2_h_long_rate12, RATE_1_2_LONG
from ldpc.dvb_s2_ldpc import encode_dvbs2_long_rate12
from ldpc.codeword_enumeration import (random_codeword_weights, low_weight_codeword_search_unit_info, low_weight_codeword_search_pair_info)

print('Building H matrix...')
H = build_dvbs2_h_long_rate12()
print(f'Done')
print()
print('Random codeword weight distribution (50 trials)')
t0 = time.time()
stats = random_codeword_weights(encode_dvbs2_long_rate12, RATE_1_2_LONG.k, n_trials=50, seed=0)
print(f'Done in {time.time()-t0:.1f}s')
print(f'min weight: {stats["min_weight"]}')
print(f'mean weight: {stats["mean_weight"]:.1f}')
print(f'median weight: {stats["median_weight"]:.1f}')
print(f'max weight: {stats["max_weight"]}')
print()
print('Unit-vector codeword weights (first 100 info positions)')
t0 = time.time()
unit_results = low_weight_codeword_search_unit_info(encode_dvbs2_long_rate12, RATE_1_2_LONG.k, n_examples=100)
print(f' Done in {time.time()-t0:.1f}s')
weights = [w for (_, w) in unit_results]
print(f'  Unit-codeword weight stats over 100 samples:')
print(f'min: {min(weights)}')
print(f'max: {max(weights)}')
print(f'mean: {sum(weights)/len(weights):.1f}')
unit_results.sort(key=lambda x: x[1])
print(f' Lowest 5 unit codewords: {unit_results[:5]}')
print()
print('Pair codeword weights (1000 random pairs)')
t0 = time.time()
pair_results = low_weight_codeword_search_pair_info(encode_dvbs2_long_rate12, RATE_1_2_LONG.k, n_pairs=1000, seed=0)
print(f'Done in {time.time()-t0:.1f}s')
pair_weights = [w for (_, w) in pair_results]
print(f'Pair codeword weight stats:')
print(f'min: {min(pair_weights)}')
print(f'max: {max(pair_weights)}')
print(f'mean: {sum(pair_weights)/len(pair_weights):.1f}')
print(f'Lowest 5 pairs: {pair_results[:5]}')