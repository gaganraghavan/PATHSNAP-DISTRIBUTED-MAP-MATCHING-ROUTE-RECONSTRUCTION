"""Per-observation labels derived from the reference route.

The ``.route`` file lists the arcs the vehicle used, in order, but does not
say which arc each GPS point belongs to. We assign every observation to one
position of the route with a monotonic alignment: observation i gets route
position j_i with j_0 <= j_1 <= ... and the total point-to-arc distance is
minimised (dynamic programming). The label of observation i is
``route[j_i]``.

Labels are only ever built for train and student-test trips. They are used as
classifier targets and for scoring, never as model inputs.
"""

from __future__ import annotations

import numpy as np

from .network import Network, point_segment_distance


def route_distance_matrix(xy: np.ndarray, net: Network, route: np.ndarray, chunk: int = 512) -> np.ndarray:
    """(n_obs, len(route)) distances in metres from each point to each route arc."""
    a, b = net.node_xy[net.u[route]], net.node_xy[net.v[route]]
    out = np.empty((len(xy), len(route)), dtype=np.float32)
    for s in range(0, len(xy), chunk):
        p = xy[s : s + chunk, None, :]
        out[s : s + chunk] = point_segment_distance(p, a[None], b[None])[0]
    return out


def align_to_route(xy: np.ndarray, net: Network, route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic alignment of observations to route positions.

    Returns (route_pos, dist): the chosen route index per observation and the
    distance in metres from the observation to that arc.
    """
    d = route_distance_matrix(xy, net, route)
    n, m = d.shape
    back = np.empty((n, m), dtype=np.int32)
    cost = d[0].astype(np.float64)
    back[0] = np.arange(m)
    for i in range(1, n):
        # best predecessor position k <= j: running minimum over the previous row
        best_k = _running_argmin(cost)
        cost = cost[best_k] + d[i]
        back[i] = best_k
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return pos, d[np.arange(n), pos]


def _running_argmin(x: np.ndarray) -> np.ndarray:
    """idx[j] = argmin(x[:j+1]) (first occurrence on ties)."""
    run_min = np.minimum.accumulate(x)
    is_new = np.empty(len(x), dtype=bool)
    is_new[0] = True
    is_new[1:] = run_min[1:] < run_min[:-1]
    idx = np.where(is_new, np.arange(len(x)), 0)
    return np.maximum.accumulate(idx)


def observation_labels(xy: np.ndarray, net: Network, route: np.ndarray) -> dict[str, np.ndarray]:
    pos, dist = align_to_route(xy, net, route)
    return {"route_pos": pos, "label_arc": route[pos], "label_dist": dist}
