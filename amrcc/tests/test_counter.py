from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve().parent
_AMRCC = _THIS.parent
_ROOT = _AMRCC.parent
for _p in [str(_ROOT), str(_AMRCC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from counter import FrameCounter, AntiReplayWindow, Status

def test_framecounter_monotonic_unique():
    c = FrameCounter()
    seen = [c.next() for _ in range(1000)]
    assert seen == list(range(1000))
    assert len(set(seen)) == 1000

def test_framecounter_start_and_budget():
    c = FrameCounter(start=5)
    assert c.next() == 5 and c.next() == 6
    c2 = FrameCounter(start=(1 << 64) - 1, max_t=(1 << 64) - 1)
    assert c2.next() == (1 << 64) - 1
    try:
        c2.next()
        assert False, "expected budget exhaustion"
    except OverflowError:
        pass

def test_framecounter_restore_guard():
    c = FrameCounter()
    c.next(); c.next(); c.next()
    c.restore(10)
    assert c.next() == 10
    try:
        c.restore(5)
        assert False, "expected reuse guard"
    except RuntimeError:
        pass

def test_antireplay_strictly_increasing():
    w = AntiReplayWindow(size=64)
    for t in range(200):
        assert w.check(t) is Status.ACCEPT
    assert w.highest == 199

def test_antireplay_exact_replay_rejected():
    w = AntiReplayWindow(size=64)
    assert w.check(10) is Status.ACCEPT
    assert w.check(10) is Status.REJECT_REPLAY
    assert w.check(11) is Status.ACCEPT
    assert w.check(11) is Status.REJECT_REPLAY

def test_antireplay_out_of_order_within_window():
    w = AntiReplayWindow(size=64)
    w.check(100)
    assert w.check(98) is Status.ACCEPT
    assert w.check(98) is Status.REJECT_REPLAY
    assert w.check(99) is Status.ACCEPT

def test_antireplay_stale_below_window_rejected():
    w = AntiReplayWindow(size=16)
    w.check(100)
    assert w.check(50) is Status.REJECT_STALE
    assert w.check(84) is Status.REJECT_STALE

def test_antireplay_forward_jump_and_desync():
    w = AntiReplayWindow(size=64, resync_gap=1000)
    w.check(10)
    assert w.check(500) is Status.ACCEPT
    assert w.check(500 + 5000) is Status.DESYNC
    assert w.highest == 5500
    assert w.check(5500) is Status.REJECT_REPLAY

def test_antireplay_no_t_accepted_twice_fuzz():
    rng = np.random.default_rng(0)
    w = AntiReplayWindow(size=128)
    seq = rng.integers(0, 2000, size=20000)
    accepted = []
    for t in seq:
        if w.check(int(t)) in (Status.ACCEPT, Status.DESYNC):
            accepted.append(int(t))
    assert len(accepted) == len(set(accepted))

def test_joint_invariant_tx_to_rx():
    tx = FrameCounter()
    issued = [tx.next() for _ in range(2000)]
    rng = np.random.default_rng(1)
    delivery = list(issued) + list(rng.choice(issued, size=500))
    rng.shuffle(delivery)
    delivery_sorted_ish = sorted(delivery, key=lambda x: (x // 64, rng.random()))
    w = AntiReplayWindow(size=256)
    accepted = [t for t in delivery_sorted_ish if w.check(t) in (Status.ACCEPT, Status.DESYNC)]
    assert sorted(set(accepted)) == sorted(set(issued))
    assert len(accepted) == len(set(accepted))

if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items())
           if n.startswith("test_") and callable(v)]
    for fn in fns:
        t0 = time.time()
        fn()
        print(f"PASS  {fn.__name__:46s} {time.time() - t0:6.2f}s")
    print(f"\nall {len(fns)} counter tests passed")