"""Coordinate handling.

Raw coordinates are (lon, lat) in WGS84 degrees. All distances are computed
in metres after projecting to the UTM zone that contains the trip's GPS
centroid. UTM distortion within one zone is < 0.1 %, which is negligible next
to GPS noise of several metres.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer

EARTH_RADIUS_M = 6_371_008.8


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=None)
def _transformer(epsg: int) -> Transformer:
    return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)


def project(lon, lat, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat degrees to metres (x east, y north) in the given CRS."""
    x, y = _transformer(epsg).transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
