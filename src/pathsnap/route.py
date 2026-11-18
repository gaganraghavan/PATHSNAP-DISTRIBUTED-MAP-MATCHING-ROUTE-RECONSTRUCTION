"""Route reconstruction from per-observation arcs, and sequence-level route metrics.

Reconstruction: consecutive observations with the same arc collapse into one
route element. Between two different consecutive arcs a and b, if a ends where b
starts, b is appended. Otherwise the shortest directed path from a's end node to
b's start node is inserted (bounded by the same speed limit as the decoder). If
there is none, b is appended anyway and the jump is counted as a route break.
The same procedure is applied to classifier-only and HMM predictions, so the
two are compared on equal terms.

Metrics against the reference ``.route`` (evaluation only):
- route precision / recall / F1, length-weighted over the sets of directed arcs
  (an arc driven in the wrong direction counts as wrong);
- route mismatch (Newson and Krumm 2009): (length wrongly added + length wrongly
  missed) / reference length; 0 is perfect and it can exceed 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm import HMMParams, RoadGraph


def reconstruct(graph: RoadGraph, pred: pd.DataFrame, t: np.ndarray,
                params: HMMParams = HMMParams()) -> tuple[list[int], int]:
    """Arc sequence of the driven route from per-observation ``pred`` (obs, arc)."""
    net = graph.net
    p = pred.sort_values("obs")
    arcs, obs = p.arc.to_numpy(), p.obs.to_numpy()
    if len(arcs) == 0:
        return [], 0
    first = np.r_[True, arcs[1:] != arcs[:-1]]
    run_arc = arcs[first]
    run_start = obs[first]
    run_end = np.r_[obs[np.flatnonzero(first)[1:] - 1], obs[-1]]
    route, breaks = [int(run_arc[0])], 0
    for k in range(1, len(run_arc)):
        a, b = int(run_arc[k - 1]), int(run_arc[k])
        if net.v[a] != net.u[b]:
            dt = max(float(t[run_start[k]] - t[run_end[k - 1]]), 1.0)
            path = graph.path_arcs(int(net.v[a]), int(net.u[b]), params.max_speed_mps * dt + params.slack_m)
            if path is None:
                breaks += 1
            else:
                route.extend(path)
        route.append(b)
    return route, breaks


def route_metrics(pred_route, ref_route, length: np.ndarray) -> dict[str, float]:
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    lp = float(length[list(P)].sum()) if P else 0.0
    lr = float(length[list(R)].sum()) if R else 0.0
    li = float(length[list(P & R)].sum()) if P & R else 0.0
    prec = li / lp if lp else 0.0
    rec = li / lr if lr else 0.0
    return {"route_precision": prec, "route_recall": rec,
            "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "route_mismatch": ((lp - li) + (lr - li)) / lr if lr else float("nan"),
            "route_length_km": lp / 1000, "reference_length_km": lr / 1000}
