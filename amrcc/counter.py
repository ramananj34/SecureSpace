from __future__ import annotations
from enum import Enum

_MAX_T = (1 << 64) - 1

class Status(Enum):
    ACCEPT = "accept"
    REJECT_REPLAY = "reject_replay"
    REJECT_STALE = "reject_stale"
    DESYNC = "desync"

class FrameCounter:

    def __init__(self, start: int = 0, max_t: int = _MAX_T):
        if not (0 <= start <= max_t):
            raise ValueError("start out of range")
        self._t = int(start)
        self._max_t = int(max_t)
        self._last = None

    @property
    def value(self) -> int:
        return self._t

    def next(self) -> int:
        if self._t > self._max_t:
            raise OverflowError("frame counter budget exhausted -- rekey required")
        t = self._t
        if self._last is not None and t <= self._last:
            raise RuntimeError("counter regressed -- would reuse (K, t)")
        self._last = t
        self._t = t + 1
        return t

    def restore(self, persisted_t: int) -> None:
        persisted_t = int(persisted_t)
        if self._last is not None and persisted_t <= self._last:
            raise RuntimeError("restore would reuse a counter -- rekey instead")
        if not (0 <= persisted_t <= self._max_t):
            raise ValueError("persisted_t out of range")
        self._t = persisted_t

class AntiReplayWindow:

    def __init__(self, size: int = 1024, resync_gap: int | None = None):
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)
        self.resync_gap = None if resync_gap is None else int(resync_gap)
        self._hi = -1
        self._seen: set[int] = set()

    @property
    def highest(self) -> int:
        return self._hi

    def _prune(self) -> None:
        lo = self._hi - self.size + 1
        if lo > 0:
            self._seen = {t for t in self._seen if t >= lo}

    def check(self, t: int, commit: bool = True) -> Status:
        t = int(t)
        if t < 0:
            raise ValueError("t must be non-negative")
        if self._hi < 0:
            if commit:
                self._hi = t
                self._seen = {t}
            return Status.ACCEPT
        if t > self._hi:
            jump = t - self._hi
            status = (Status.DESYNC if (self.resync_gap is not None and jump > self.resync_gap) else Status.ACCEPT)
            if commit:
                self._hi = t
                self._seen.add(t)
                self._prune()
            return status
        if t < self._hi - self.size + 1:
            return Status.REJECT_STALE
        if t in self._seen:
            return Status.REJECT_REPLAY
        if commit:
            self._seen.add(t)
        return Status.ACCEPT