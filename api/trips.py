"""Discovery and loading of the trips the API serves.

The served folder holds one sub-folder per trip, each with ``.track``,
``.nodes``, ``.arcs`` and (server-side only) ``.route`` files. Files are found
by extension, so a folder does not have to follow the ``<trip>/<trip>.<ext>``
naming exactly.

Two loaders keep the reference route away from the prediction pipeline:
:meth:`TripStore.load_for_prediction` never reads ``.route``, and
:meth:`TripStore.load_reference` is called only after prediction, to score it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pathsnap.io import Trip, read_arcs, read_nodes, read_route, read_track


@dataclass(frozen=True)
class TripInfo:
    trip_id: str
    n_obs: int
    duration_s: float
    has_reference: bool


class TripStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"trip folder not found: {self.root}")
        self._dirs = {p.name: p for p in sorted(self.root.iterdir())
                      if p.is_dir() and self._find(p, "track", required=False)}
        self._info: dict[str, TripInfo] = {}

    @staticmethod
    def _find(folder: Path, ext: str, required: bool = True) -> Path | None:
        exact = folder / f"{folder.name}.{ext}"
        if exact.is_file():
            return exact
        found = sorted(folder.glob(f"*.{ext}"))
        if len(found) > 1:
            raise ValueError(f"{folder.name}: more than one .{ext} file")
        if not found and required:
            raise FileNotFoundError(f"{folder.name}: no .{ext} file")
        return found[0] if found else None

    @property
    def trip_ids(self) -> list[str]:
        return list(self._dirs)

    def __contains__(self, trip_id: str) -> bool:
        return trip_id in self._dirs

    def info(self, trip_id: str) -> TripInfo:
        if trip_id not in self._info:
            folder = self._dirs[trip_id]
            t = read_track(self._find(folder, "track")).t.to_numpy()
            self._info[trip_id] = TripInfo(
                trip_id=trip_id, n_obs=len(t),
                duration_s=float(t[-1] - t[0]) if len(t) else 0.0,
                has_reference=self._find(folder, "route", required=False) is not None)
        return self._info[trip_id]

    def load_for_prediction(self, trip_id: str) -> Trip:
        """GPS track and road network only. The ``.route`` file is not opened."""
        folder = self._dirs[trip_id]
        return Trip(trip_id=trip_id, partition=self.root.name,
                    track=read_track(self._find(folder, "track")),
                    nodes=read_nodes(self._find(folder, "nodes")),
                    arcs=read_arcs(self._find(folder, "arcs")),
                    route=None)

    def load_reference(self, trip_id: str) -> np.ndarray | None:
        """Server-side ground truth, used for scoring only. Never sent to the client."""
        path = self._find(self._dirs[trip_id], "route", required=False)
        return None if path is None else read_route(path)
