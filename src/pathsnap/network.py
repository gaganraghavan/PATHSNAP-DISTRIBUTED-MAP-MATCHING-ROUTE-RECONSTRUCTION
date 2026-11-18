"""Projected road network for one trip.

Arcs carry no shape geometry, so each arc is modelled as the straight segment
between its two nodes. Arcs are directed; most roads appear as two arcs, one
per direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from shapely import STRtree, linestrings

from .geo import project, utm_epsg
from .io import Trip


@dataclass
class Network:
    epsg: int
    node_xy: np.ndarray        # (N, 2) metres
    u: np.ndarray              # (M,) from-node per arc
    v: np.ndarray              # (M,) to-node per arc
    _tree: STRtree | None = field(default=None, repr=False)

    @property
    def n_nodes(self) -> int:
        return len(self.node_xy)

    @property
    def n_arcs(self) -> int:
        return len(self.u)

    @property
    def a(self) -> np.ndarray:
        return self.node_xy[self.u]

    @property
    def b(self) -> np.ndarray:
        return self.node_xy[self.v]

    @property
    def length(self) -> np.ndarray:
        return np.linalg.norm(self.b - self.a, axis=1)

    @property
    def heading(self) -> np.ndarray:
        """Arc bearing in radians, clockwise from north."""
        d = self.b - self.a
        return np.arctan2(d[:, 0], d[:, 1])

    @property
    def tree(self) -> STRtree:
        if self._tree is None:
            segs = np.stack([self.a, self.b], axis=1)
            self._tree = STRtree(linestrings(segs))
        return self._tree

    def adjacency(self) -> sp.csr_matrix:
        """Directed node adjacency (node u -> node v for every arc)."""
        n = self.n_nodes
        return sp.csr_matrix((np.ones(self.n_arcs), (self.u, self.v)), shape=(n, n))

    def components(self, strong: bool = False) -> np.ndarray:
        """Component label per node (weak connectivity unless ``strong``)."""
        _, labels = connected_components(
            self.adjacency(), directed=True, connection="strong" if strong else "weak"
        )
        return labels


def build_network(trip: Trip, epsg: int | None = None) -> Network:
    """Project a trip's network. The CRS is chosen from the GPS track only."""
    if epsg is None:
        epsg = utm_epsg(float(trip.track.lon.mean()), float(trip.track.lat.mean()))
    x, y = project(trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy(), epsg)
    return Network(
        epsg=epsg,
        node_xy=np.column_stack([x, y]),
        u=trip.arcs.u.to_numpy(),
        v=trip.arcs.v.to_numpy(),
    )


def track_xy(trip: Trip, epsg: int) -> np.ndarray:
    x, y = project(trip.track.lon.to_numpy(), trip.track.lat.to_numpy(), epsg)
    return np.column_stack([x, y])


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Distance from points ``p`` to segments ``a``-``b`` (row-wise, broadcastable).

    Returns (distance, t) where t in [0, 1] is the position of the closest
    point along the segment.
    """
    ab = b - a
    denom = np.einsum("...i,...i->...", ab, ab)
    t = np.einsum("...i,...i->...", p - a, ab) / np.where(denom > 0, denom, 1.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1), t
