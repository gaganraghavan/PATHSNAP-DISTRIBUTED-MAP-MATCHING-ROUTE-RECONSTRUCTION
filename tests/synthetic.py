"""Small synthetic trips (grid road network + noisy 1 Hz GPS) for API tests.

Used only to exercise the code paths end to end without the real dataset; no
reported result comes from these trips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LON0, LAT0 = 77.59, 12.97
M_PER_DEG_LAT = 111_320.0


def _to_lonlat(x, y):
    return LON0 + x / (M_PER_DEG_LAT * np.cos(np.radians(LAT0))), LAT0 + y / M_PER_DEG_LAT


def write_trip(root: Path, trip_id: str, seed: int = 0, n: int = 8, spacing: float = 150.0,
               n_arcs: int = 24, speed: float = 12.0, noise_m: float = 6.0, with_route: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    ij = np.array([(i, j) for j in range(n) for i in range(n)])
    node_xy = ij * spacing
    node_id = {tuple(p): k for k, p in enumerate(ij)}
    arcs = []
    for (i, j), k in node_id.items():
        for di, dj in ((1, 0), (0, 1)):
            if (i + di, j + dj) in node_id:
                m = node_id[(i + di, j + dj)]
                arcs += [(k, m), (m, k)]
    arcs = np.array(arcs)
    arc_of = {(a, b): idx for idx, (a, b) in enumerate(arcs)}
    out_nb = {k: [b for a, b in arcs if a == k] for k in range(len(node_xy))}

    # Random walk without U-turns.
    node, prev, route = node_id[(n // 2, n // 2)], None, []
    for _ in range(n_arcs):
        nxt = [b for b in out_nb[node] if b != prev]
        b = int(rng.choice(nxt))
        route.append(arc_of[(node, b)])
        prev, node = node, b

    # 1 Hz samples along the route, with one 8 s gap.
    path = np.concatenate([node_xy[[arcs[r][0] for r in route]], node_xy[[arcs[route[-1]][1]]]])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, cum[-1], speed)
    t = np.arange(len(s), dtype=float) + 1
    gap = slice(len(s) // 2, len(s) // 2 + 8)
    s, t = np.delete(s, gap), np.delete(t, gap)
    x = np.interp(s, cum, path[:, 0]) + rng.normal(0, noise_m, len(s))
    y = np.interp(s, cum, path[:, 1]) + rng.normal(0, noise_m, len(s))

    d = Path(root) / trip_id
    d.mkdir(parents=True, exist_ok=True)
    lon, lat = _to_lonlat(x, y)
    np.savetxt(d / f"{trip_id}.track", np.column_stack([lon, lat, t]), fmt="%.7f %.7f %.0f")
    nlon, nlat = _to_lonlat(node_xy[:, 0], node_xy[:, 1])
    np.savetxt(d / f"{trip_id}.nodes", np.column_stack([nlon, nlat]), fmt="%.7f %.7f")
    np.savetxt(d / f"{trip_id}.arcs", arcs, fmt="%d %d")
    if with_route:
        np.savetxt(d / f"{trip_id}.route", np.array(route), fmt="%d")
    return d
