"""HMM map matching with Viterbi decoding.

Hidden states
    At observation i the states are its candidate arcs (step 02 candidates). A
    state carries the arc id and the position ``t`` (0..1) of the GPS point's
    projection along the arc. Observations with no candidate have no states; the
    chain skips them and the next transition spans the longer time gap.

Emissions
    ``emission_weight * log P(arc | observation)``, where P is the candidate
    ranker's per-observation softmax probability (floored at ``PROB_FLOOR``).

Transitions (from state a at observation i to state b at the next observation j)
    ``route_m``: network distance from a's projected point to b's projected point
    along directed arcs: the rest of a, the shortest path from a's end node to
    b's start node, then the start of b. On the same arc, moving forward (or
    backwards by at most ``backward_tolerance_m``, GPS jitter) costs the distance
    along the arc. ``gps_m`` is the straight-line distance between the two GPS
    points. Following Newson and Krumm (2009), the log transition probability is
    ``-|route_m - gps_m| / (beta_m * max(1, dt))``: it prefers moves whose road
    distance matches the observed displacement, and it flattens as the time gap
    ``dt`` grows, because over a long gap the vehicle's path is less constrained.

    A transition is unreachable when no directed path exists or when
    ``route_m > max_speed_mps * dt + slack_m``. Unreachable transitions get the
    constant log-probability ``disconnect_logp`` instead of -inf, so the decoder
    can still cross a break in the network (disconnected components, off-network
    driving, missing arcs). Such steps are counted as breaks.

The road graph only keeps arcs whose two nodes lie in the track's bounding box
plus ``margin_m``. Shortest paths are computed with Dijkstra on that graph.
Nothing here reads the reference route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from .network import Network

PROB_FLOOR = 1e-9


@dataclass(frozen=True)
class HMMParams:
    beta_m: float = 10.0              # transition scale in metres at a 1 s step
    emission_weight: float = 1.0      # exponent on the classifier probability
    disconnect_logp: float = -30.0    # log-prob of an unreachable transition
    # Reachability, fixed when the lattice is built:
    max_speed_mps: float = 60.0
    slack_m: float = 150.0
    backward_tolerance_m: float = 15.0

    def as_dict(self) -> dict:
        return asdict(self)


class RoadGraph:
    """Directed, length-weighted graph of the arcs near one track."""

    def __init__(self, net: Network, xy: np.ndarray, margin_m: float = 2000.0, chunk: int = 32):
        self.net = net
        self.length = net.length
        self.n = net.n_nodes
        self.chunk = chunk
        lo, hi = xy.min(axis=0) - margin_m, xy.max(axis=0) + margin_m
        inside = np.all((net.node_xy >= lo) & (net.node_xy <= hi), axis=1)
        keep = np.flatnonzero(inside[net.u] & inside[net.v] & (net.u != net.v))
        e = pd.DataFrame({"u": net.u[keep], "v": net.v[keep],
                          "w": np.maximum(self.length[keep], 1e-3), "arc": keep})
        # Parallel arcs: keep the shortest (a CSR matrix would sum duplicates).
        e = e.sort_values(["u", "v", "w"], kind="stable").drop_duplicates(["u", "v"])
        self.graph = sp.csr_matrix((e.w.to_numpy(), (e.u.to_numpy(), e.v.to_numpy())), shape=(self.n, self.n))
        self._edge_arc = dict(zip((e.u.to_numpy() * self.n + e.v.to_numpy()).tolist(), e.arc.tolist()))

    def pair_distances(self, src: np.ndarray, tgt: np.ndarray, limit: np.ndarray) -> np.ndarray:
        """Shortest-path length for each (src, tgt) node pair; inf beyond the pair's ``limit``."""
        out = np.full(len(src), np.inf)
        if len(src) == 0:
            return out
        key = src.astype(np.int64) * self.n + tgt.astype(np.int64)
        uniq, inv = np.unique(key, return_inverse=True)
        ulim = np.zeros(len(uniq))
        np.maximum.at(ulim, inv, limit)
        usrc, utgt = uniq // self.n, uniq % self.n
        vals = np.full(len(uniq), np.inf)
        same = usrc == utgt
        vals[same] = 0.0
        sources, starts = np.unique(usrc, return_index=True)   # uniq is sorted by source
        bounds = np.r_[starts, len(uniq)]
        for c in range(0, len(sources), self.chunk):
            block = sources[c:c + self.chunk]
            lo, hi = bounds[c], bounds[min(c + self.chunk, len(sources))]
            lim = float(ulim[lo:hi].max())
            d = dijkstra(self.graph, directed=True, indices=block, limit=lim)
            row = np.searchsorted(block, usrc[lo:hi])
            vals[lo:hi] = np.minimum(vals[lo:hi], d[row, utgt[lo:hi]])
        res = vals[inv]
        res[res > limit] = np.inf
        return res

    def path_arcs(self, s: int, t: int, limit: float) -> list[int] | None:
        """Arc ids of the shortest directed path from node s to node t, or None."""
        if s == t:
            return []
        d, pred = dijkstra(self.graph, directed=True, indices=s, limit=limit, return_predecessors=True)
        if not np.isfinite(d[t]):
            return None
        nodes = [t]
        while nodes[-1] != s:
            nodes.append(int(pred[nodes[-1]]))
        nodes.reverse()
        return [self._edge_arc[a * self.n + b] for a, b in zip(nodes[:-1], nodes[1:])]


@dataclass
class Lattice:
    obs: np.ndarray                  # observation index of each step
    arcs: list[np.ndarray]           # candidate arcs per step
    emission: list[np.ndarray]       # log P per candidate
    absdiff: list[np.ndarray | None] # |route_m - gps_m| into step k, shape (n_{k-1}, n_k)
    reachable: list[np.ndarray | None]
    dt: np.ndarray                   # seconds from step k-1 to step k (dt[0] = 0)


def build_lattice(graph: RoadGraph, xy: np.ndarray, t: np.ndarray, cands: pd.DataFrame,
                  params: HMMParams = HMMParams()) -> Lattice:
    """``cands``: one row per (obs, arc) with ``position_along_arc`` and ``prob``."""
    net = graph.net
    c = cands.sort_values(["obs", "prob"], ascending=[True, False], kind="stable")
    obs_ids, starts = np.unique(c.obs.to_numpy(), return_index=True)
    bounds = np.r_[starts, len(c)]
    arc_all = c.arc.to_numpy().astype(np.int64)
    pos_all = c.position_along_arc.to_numpy().astype(np.float64)
    em_all = np.log(np.clip(c.prob.to_numpy().astype(np.float64), PROB_FLOOR, 1.0))
    arcs = [arc_all[bounds[k]:bounds[k + 1]] for k in range(len(obs_ids))]
    pos = [pos_all[bounds[k]:bounds[k + 1]] for k in range(len(obs_ids))]
    emission = [em_all[bounds[k]:bounds[k + 1]] for k in range(len(obs_ids))]
    dt = np.zeros(len(obs_ids))
    if len(obs_ids) > 1:
        dt[1:] = np.maximum(np.diff(t[obs_ids]), 1e-3)

    # All transition pairs at once: source node = end of a, target node = start of b.
    src, tgt, lim, shapes = [], [], [], []
    for k in range(1, len(obs_ids)):
        a, b = arcs[k - 1], arcs[k]
        shapes.append((len(a), len(b)))
        src.append(np.repeat(net.v[a], len(b)))
        tgt.append(np.tile(net.u[b], len(a)))
        lim.append(np.full(len(a) * len(b), params.max_speed_mps * dt[k] + params.slack_m))
    if src:
        sp_all = graph.pair_distances(np.concatenate(src), np.concatenate(tgt), np.concatenate(lim))
    absdiff, reachable = [None], [None]
    off = 0
    L = graph.length
    for k in range(1, len(obs_ids)):
        na, nb = shapes[k - 1]
        spd = sp_all[off:off + na * nb].reshape(na, nb)
        off += na * nb
        a, b, ta, tb = arcs[k - 1], arcs[k], pos[k - 1], pos[k]
        route = (1 - ta)[:, None] * L[a][:, None] + spd + (tb * L[b])[None, :]
        same = a[:, None] == b[None, :]
        along = (tb[None, :] - ta[:, None]) * L[a][:, None]
        jitter = same & (along >= -params.backward_tolerance_m)
        route = np.where(jitter, np.abs(along), route)
        gps = float(np.linalg.norm(xy[obs_ids[k]] - xy[obs_ids[k - 1]]))
        ok = np.isfinite(route) & (route <= params.max_speed_mps * dt[k] + params.slack_m)
        absdiff.append(np.where(ok, np.abs(route - gps), 0.0).astype(np.float32))
        reachable.append(ok)
    return Lattice(obs_ids, arcs, emission, absdiff, reachable, dt)


def viterbi(lat: Lattice, params: HMMParams = HMMParams()) -> tuple[pd.DataFrame, int]:
    """Most likely state sequence. Returns (obs, arc) per step and the number of breaks taken."""
    n = len(lat.obs)
    if n == 0:
        return pd.DataFrame({"obs": np.array([], dtype=np.int64), "arc": np.array([], dtype=np.int64)}), 0
    w = params.emission_weight
    delta = w * lat.emission[0]
    back = [None] * n
    for k in range(1, n):
        trans = np.where(lat.reachable[k], -lat.absdiff[k] / (params.beta_m * max(1.0, lat.dt[k])),
                         params.disconnect_logp)
        score = delta[:, None] + trans
        bk = score.argmax(axis=0)
        delta = score[bk, np.arange(score.shape[1])] + w * lat.emission[k]
        back[k] = bk
    state = np.empty(n, dtype=np.int64)
    state[-1] = int(delta.argmax())
    breaks = 0
    for k in range(n - 1, 0, -1):
        state[k - 1] = back[k][state[k]]
        breaks += int(not lat.reachable[k][state[k - 1], state[k]])
    arcs = np.array([lat.arcs[k][state[k]] for k in range(n)], dtype=np.int64)
    return pd.DataFrame({"obs": lat.obs, "arc": arcs}), breaks


def classifier_only(cands: pd.DataFrame) -> pd.DataFrame:
    """Highest-probability candidate per observation, ties to the nearer candidate."""
    c = cands.sort_values(["obs", "prob", "candidate_rank"], ascending=[True, False, True], kind="stable")
    return c.drop_duplicates("obs")[["obs", "arc"]].reset_index(drop=True)
