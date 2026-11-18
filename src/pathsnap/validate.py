"""Per-trip inventory and data-quality checks (EDA evidence)."""

from __future__ import annotations

import numpy as np

from .io import EXTENSIONS, Trip, trip_file
from .network import Network

GAP_S = 10.0  # a time step longer than this counts as a GPS gap


def trip_summary(trip: Trip, net: Network, xy: np.ndarray, labels: dict | None = None) -> dict:
    tr, ar = trip.track, trip.arcs
    t = tr.t.to_numpy()
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = step / np.where(dt > 0, dt, np.nan)

    in_range = tr.lon.between(-180, 180) & tr.lat.between(-90, 90)
    deg = np.bincount(np.concatenate([net.u, net.v]), minlength=net.n_nodes)
    weak = net.components(strong=False)
    strong = net.components(strong=True)
    pairs = set(zip(net.u.tolist(), net.v.tolist()))
    two_way = np.mean([(v, u) in pairs for u, v in zip(net.u.tolist(), net.v.tolist())])

    row = {
        "partition": trip.partition,
        "trip": trip.trip_id,
        **{f"bytes_{e}": (trip_file(trip.partition, trip.trip_id, e).stat().st_size
                          if trip_file(trip.partition, trip.trip_id, e).exists() else 0)
           for e in EXTENSIONS},
        "epsg": net.epsg,
        # track
        "n_obs": len(tr),
        "n_missing": int(tr.isna().any(axis=1).sum()),
        "n_out_of_range": int((~in_range).sum()),
        "n_duplicate_rows": int(tr.duplicated().sum()),
        "n_stationary_steps": int((step < 0.5).sum()),
        "t_start": float(t[0]),
        "duration_s": float(t[-1] - t[0]),
        "n_non_increasing_t": int((dt <= 0).sum()),
        "frac_dt_1s": float((dt == 1).mean()),
        "dt_max_s": float(dt.max()),
        "n_gaps_gt10s": int((dt > GAP_S).sum()),
        "gap_time_frac": float(dt[dt > GAP_S].sum() / max(t[-1] - t[0], 1)),
        "path_km": float(step.sum() / 1000),
        "speed_p50_ms": float(np.nanpercentile(speed, 50)),
        "speed_p99_ms": float(np.nanpercentile(speed, 99)),
        "n_speed_gt70ms": int(np.nansum(speed > 70)),
        "lon_min": float(tr.lon.min()), "lon_max": float(tr.lon.max()),
        "lat_min": float(tr.lat.min()), "lat_max": float(tr.lat.max()),
        "extent_km_x": float(np.ptp(xy[:, 0]) / 1000),
        "extent_km_y": float(np.ptp(xy[:, 1]) / 1000),
        # network
        "n_nodes": net.n_nodes,
        "n_arcs": net.n_arcs,
        "n_isolated_nodes": int((deg == 0).sum()),
        "n_self_loops": int((net.u == net.v).sum()),
        "n_duplicate_arcs": int(ar.duplicated().sum()),
        "n_zero_length_arcs": int((net.length < 0.01).sum()),
        "arc_len_p50_m": float(np.median(net.length)),
        "frac_two_way": float(two_way),
        "n_weak_components": int(weak.max() + 1),
        "n_strong_components": int(strong.max() + 1),
        "frac_nodes_largest_scc": float(np.bincount(strong).max() / net.n_nodes),
    }

    if trip.route is not None:
        r = trip.route
        row.update({
            "route_len": len(r),
            "route_out_of_range": int(((r < 0) | (r >= net.n_arcs)).sum()),
            "route_unique_arcs": int(len(np.unique(r))),
            "route_breaks": int((net.v[r[:-1]] != net.u[r[1:]]).sum()),
            "route_km": float(net.length[r].sum() / 1000),
            "route_in_largest_scc": float(np.mean(strong[net.u[r]] == np.bincount(strong).argmax())),
        })
    if labels is not None:
        d = labels["label_dist"]
        row.update({
            "label_dist_p50_m": float(np.median(d)),
            "label_dist_p95_m": float(np.percentile(d, 95)),
            "label_dist_max_m": float(d.max()),
            "frac_label_dist_gt50m": float((d > 50).mean()),
            "route_positions_used": float(len(np.unique(labels["route_pos"])) / len(trip.route)),
        })
    return row
