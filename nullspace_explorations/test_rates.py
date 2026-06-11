from __future__ import annotations
import sys
from pathlib import Path
_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from dvb_s2_rates import ALL_RATES, make_code, verify, LONG_RATE_TABLES

HEADER_BITS = 400
SLOT_BITS = 800

def main() -> int:
    print("=== Week 5 Day 2: DVB-S2 long-frame rate family (n=64800) ===\n")
    print(f"{'rate':>5} {'k':>6} {'q':>4} {'rows':>5} {'nnz':>7} "
          f"{'nnz_ok':>6} {'col_ok':>6} {'Hc=0':>5} {'sys':>4} {'rowwt':>8} {'extra':>16}")
    all_pass = True
    results = {}
    for rate in ALL_RATES:
        try:
            r = verify(rate)
        except Exception as e:
            print(f"{rate:>5}  FAILED {type(e).__name__}: {e}")
            all_pass = False
            continue
        results[rate] = r
        all_pass &= r["all_ok"]
        extra = ("H==frozen:" + str(r.get("frozen_H_identical"))) if rate == "1/2" else ""
        rowwt = f"[{r['rowwt_min']},{r['rowwt_max']}]"
        print(f"{rate:>5} {r['k']:>6} {r['q']:>4} {r['rows']:>5} {r['nnz']:>7} "
              f"{str(r['nnz_ok']):>6} {str(r['col_dist_ok']):>6} {str(r['Hc0_ok']):>5} "
              f"{str(r['systematic_ok']):>4} {rowwt:>8} {extra:>16}")
    print(f"\nALL RATES VERIFIED: {all_pass}")
    print("\n=== E4 rate table (parity floor = syndrome-passing bypass cost) ===")
    print(f"{'rate':>5} {'k':>6} {'(n-k)/2':>8} {'win/frame':>10}")
    for rate in ALL_RATES:
        _, _, p = LONG_RATE_TABLES[rate]
        floor = p.n_minus_k // 2
        win = (p.k - HEADER_BITS) // SLOT_BITS
        print(f"{rate:>5} {p.k:>6} {floor:>8} {win:>10}")
    print("\nmonotone: bypass cost (n-k)/2 decreases with rate; win/frame increases.")
    print("(telemetry-evasion success is rate-INDEPENDENT by code-blindness; shown in Day 3.)")
    return 0 if all_pass else 1

if __name__ == "__main__":
    raise SystemExit(main())