"""Readers for the raw per-trip files.

Each trip folder ``<partition>/<trip_id>/`` holds four headerless,
whitespace-separated files named ``<trip_id>.<ext>``:

- ``.track``  lon, lat, timestamp (s)
- ``.nodes``  lon, lat; the 0-based row index is the node id
- ``.arcs``   from_node, to_node; the 0-based row index is the arc id
- ``.route``  chronological arc ids of the reference route (labels)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_ROOT

EXTENSIONS = ("track", "nodes", "arcs", "route")


def partition_dir(partition: str, root: Path | None = None) -> Path:
    return Path(root or DATA_ROOT) / partition


def list_trips(partition: str, root: Path | None = None) -> list[str]:
    d = partition_dir(partition, root)
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def trip_file(partition: str, trip_id: str, ext: str, root: Path | None = None) -> Path:
    return partition_dir(partition, root) / trip_id / f"{trip_id}.{ext}"


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def read_track(path: Path) -> pd.DataFrame:
    return _read(path, ["lon", "lat", "t"], np.float64)


def read_nodes(path: Path) -> pd.DataFrame:
    return _read(path, ["lon", "lat"], np.float64)


def read_arcs(path: Path) -> pd.DataFrame:
    return _read(path, ["u", "v"], np.int64)


def read_route(path: Path) -> np.ndarray:
    return _read(path, ["arc"], np.int64)["arc"].to_numpy()


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # lon, lat, t
    nodes: pd.DataFrame          # lon, lat (index = node id)
    arcs: pd.DataFrame           # u, v (index = arc id)
    route: np.ndarray | None     # arc ids; None when labels are withheld

    @property
    def n_obs(self) -> int:
        return len(self.track)


def load_trip(
    trip_id: str,
    partition: str,
    with_route: bool = True,
    root: Path | None = None,
) -> Trip:
    """Load one trip. Pass ``with_route=False`` wherever labels must not be seen."""
    f = lambda ext: trip_file(partition, trip_id, ext, root)  # noqa: E731
    route = None
    if with_route and f("route").exists():
        route = read_route(f("route"))
    return Trip(
        trip_id=trip_id,
        partition=partition,
        track=read_track(f("track")),
        nodes=read_nodes(f("nodes")),
        arcs=read_arcs(f("arcs")),
        route=route,
    )
