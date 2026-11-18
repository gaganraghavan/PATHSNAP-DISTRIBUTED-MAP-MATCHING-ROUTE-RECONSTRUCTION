"""Candidate road arcs for each GPS observation.

For every observation we take the arcs whose straight-line segment lies within
``radius`` metres, then keep the ``k`` nearest. Ties at the k-th distance are
kept, so the two directed arcs of a two-way road are always kept or dropped
together (they are the same geometry and only heading can tell them apart).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely import points

from .network import Network, point_segment_distance

DEFAULT_RADIUS_M = 100.0
DEFAULT_K = 32


@dataclass
class Candidates:
    obs: np.ndarray    # observation index per candidate row
    arc: np.ndarray    # arc id
    dist: np.ndarray   # point-to-segment distance (m)
    t: np.ndarray      # position of the projection along the arc, 0..1
    rank: np.ndarray   # 0-based distance rank within the observation (ties share a rank)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(vars(self))


def within_radius(xy: np.ndarray, net: Network, radius: float) -> pd.DataFrame:
    """All (obs, arc) pairs within ``radius`` metres, with exact distances, sorted."""
    obs, arc = net.tree.query(points(xy), predicate="dwithin", distance=radius)
    d, t = point_segment_distance(xy[obs], net.node_xy[net.u[arc]], net.node_xy[net.v[arc]])
    df = pd.DataFrame({"obs": obs, "arc": arc, "dist": d, "t": t})
    df = df.sort_values(["obs", "dist", "arc"], kind="stable", ignore_index=True)
    df["rank"] = df.groupby("obs")["dist"].rank(method="min").astype(np.int64) - 1
    return df


def generate(xy: np.ndarray, net: Network, radius: float = DEFAULT_RADIUS_M, k: int = DEFAULT_K) -> Candidates:
    df = within_radius(xy, net, radius)
    df = df[df["rank"] < k]
    return Candidates(*(df[c].to_numpy() for c in ("obs", "arc", "dist", "t", "rank")))


def label_rank(df: pd.DataFrame, labels: np.ndarray, n_obs: int) -> np.ndarray:
    """Rank of the label arc among an observation's candidates (-1 if absent).

    ``df`` comes from :func:`within_radius`. A label is inside the top-k set
    exactly when 0 <= rank < k.
    """
    out = np.full(n_obs, -1, dtype=np.int64)
    hit = df[df["arc"].to_numpy() == labels[df["obs"].to_numpy()]]
    out[hit["obs"].to_numpy()] = hit["rank"].to_numpy()
    return out
