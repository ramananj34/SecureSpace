import sys
import time
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from ldpc.dvb_s2_ldpc import build_dvbs2_h_long_rate12
from ldpc.dvb_s2_short import build_dvbs2_h_short_rate12
from ldpc.tanner_graph import tanner_neighbors, _shortest_cycle_through_node

def exact_girth(H, label: str, cutoff: int = 8, report_every: int = 5000) -> int:
    print(f"Exhaustive girth: {label}")
    vn, cn = tanner_neighbors(H)
    n_v = len(vn)
    t0 = time.time()
    best = None
    for v in range(n_v):
        c = _shortest_cycle_through_node(v, vn, cn, n_variables=n_v, cutoff=cutoff)
        if c is not None and (best is None or c < best):
            best = c
            if best == 4:
                break
        if (v + 1) % report_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / (v + 1) * (n_v - v - 1)
            print(f"  [{v+1}/{n_v}] best={best} elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. Exact girth = {best}\n")
    return best

def main():
    girth_short = exact_girth(build_dvbs2_h_short_rate12(), "DVB-S2 short-frame rate-1/2 (Table C.4)")
    girth_long  = exact_girth(build_dvbs2_h_long_rate12(), "DVB-S2 long-frame rate-1/2 (Table B.4)")
    assert girth_short == 6, f"Short-frame girth = {girth_short}, expected 6"
    assert girth_long == 6, f"Long-frame girth = {girth_long}, expected 6"
    print(f"Both codes have exact girth = 6 (matches ETSI design target).")

if __name__ == "__main__":
    main()