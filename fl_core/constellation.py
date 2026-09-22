from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

R_EARTH_KM = 6371.0

@dataclass
class WalkerDelta:
    n_planes: int = 5
    sats_per_plane: int = 8
    phasing_F: int = 1
    inclination_deg: float = 53.0
    altitude_km: float = 550.0
    max_isl_range_km: float = 5000.0
    r_earth_km: float = R_EARTH_KM

    @property
    def n_sats(self) -> int:
        return self.n_planes * self.sats_per_plane

    @property
    def orbit_radius_km(self) -> float:
        return self.r_earth_km + self.altitude_km

    def positions(self, t_frac: float = 0.0) -> np.ndarray:
        R = self.orbit_radius_km
        incl = np.deg2rad(self.inclination_deg)
        T = self.n_sats
        pts = np.zeros((T, 3))
        idx = 0
        for p in range(self.n_planes):
            raan = 2 * np.pi * p / self.n_planes
            cr, sr = np.cos(raan), np.sin(raan)
            ci, si = np.cos(incl), np.sin(incl)
            for s in range(self.sats_per_plane):
                M = 2 * np.pi * (s / self.sats_per_plane) + 2 * np.pi * self.phasing_F * p / T + 2 * np.pi * t_frac
                xp, yp = R * np.cos(M), R * np.sin(M)
                x1, y1, z1 = xp, yp * ci, yp * si
                pts[idx] = [x1 * cr - y1 * sr, x1 * sr + y1 * cr, z1]
                idx += 1
        return pts

    def sat_plane(self, sat_idx: int) -> int:
        return sat_idx // self.sats_per_plane


def _los_clear(a: np.ndarray, b: np.ndarray, r_earth: float) -> bool:
    d = b - a
    L2 = float(d @ d)
    if L2 == 0.0:
        return True
    tt = float(np.clip(-(a @ d) / L2, 0.0, 1.0))
    return float(np.linalg.norm(a + tt * d)) > r_earth


def contact_graph(wd: WalkerDelta, t_frac: float = 0.0) -> np.ndarray:
    pts = wd.positions(t_frac)
    n = wd.n_sats
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) <= wd.max_isl_range_km and _los_clear(pts[i], pts[j], wd.r_earth_km):
                A[i, j] = A[j, i] = 1
    return A

def is_connected(A: np.ndarray) -> bool:
    n = A.shape[0]
    if n == 0:
        return True
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in np.where(A[u] > 0)[0]:
            if int(v) not in seen:
                seen.add(int(v)); stack.append(int(v))
    return len(seen) == n

def graph_stats(A: np.ndarray) -> dict:
    n = A.shape[0]
    deg = A.sum(1)
    n_edges = int(A.sum() // 2)
    density = float(A.sum() / (n * (n - 1))) if n > 1 else 0.0
    return {"n_sats": int(n), "n_edges": n_edges, "density": density,
            "mean_degree": float(deg.mean()), "min_degree": int(deg.min()), "max_degree": int(deg.max()),
            "connected": bool(is_connected(A))}

@dataclass
class Constellation:
    walker: WalkerDelta = field(default_factory=WalkerDelta)
    coordinator: int = 0
    
    @property
    def n_sats(self) -> int:
        return self.walker.n_sats

    def contact(self, t_frac: float = 0.0) -> np.ndarray:
        return contact_graph(self.walker, t_frac)

    def reachable_from_coordinator(self, t_frac: float = 0.0) -> np.ndarray:
        A = self.contact(t_frac)
        seen = {self.coordinator}
        stack = [self.coordinator]
        while stack:
            u = stack.pop()
            for v in np.where(A[u] > 0)[0]:
                if int(v) not in seen:
                    seen.add(int(v)); stack.append(int(v))
        mask = np.zeros(self.n_sats, dtype=bool)
        for s in seen:
            mask[s] = True
        return mask