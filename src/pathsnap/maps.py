"""Folium maps: raw GPS, road network, reference route, predicted route."""

from __future__ import annotations

from collections.abc import Sequence

import folium
import numpy as np
from shapely import points

from .io import Trip
from .network import Network, track_xy

COLORS = {"gps": "#d62728", "network": "#7f7f7f", "reference": "#2ca02c", "predicted": "#1f77b4"}


def _arc_latlon(trip: Trip, arcs: Sequence[int]) -> list[list[tuple[float, float]]]:
    lon, lat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    u, v = trip.arcs.u.to_numpy()[arcs], trip.arcs.v.to_numpy()[arcs]
    return [[(lat[a], lon[a]), (lat[b], lon[b])] for a, b in zip(u, v)]


def _route_polyline(trip: Trip, route: Sequence[int]) -> list[tuple[float, float]]:
    """Node chain of a route; consecutive duplicates collapse, breaks stay as jumps."""
    lon, lat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    pts: list[int] = []
    for a in route:
        for n in (u[a], v[a]):
            if not pts or pts[-1] != n:
                pts.append(int(n))
    return [(lat[n], lon[n]) for n in pts]


def nearby_arcs(trip: Trip, net: Network, buffer_m: float = 150.0) -> np.ndarray:
    """Arcs within ``buffer_m`` of any GPS point (keeps maps light on big networks)."""
    xy = track_xy(trip, net.epsg)[::5]
    _, arc = net.tree.query(points(xy), predicate="dwithin", distance=buffer_m)
    return np.unique(arc)


def trip_map(
    trip: Trip,
    net: Network | None = None,
    reference: Sequence[int] | None = None,
    predicted: Sequence[int] | None = None,
    network_buffer_m: float = 150.0,
    gps_every: int = 1,
) -> folium.Map:
    tr = trip.track
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])

    if net is not None:
        fg = folium.FeatureGroup(name="Road network (near track)")
        segs = _arc_latlon(trip, nearby_arcs(trip, net, network_buffer_m))
        folium.PolyLine(segs, color=COLORS["network"], weight=1.5, opacity=0.7).add_to(fg)
        fg.add_to(m)

    if reference is not None:
        fg = folium.FeatureGroup(name="Reference route")
        folium.PolyLine(_route_polyline(trip, reference), color=COLORS["reference"], weight=6, opacity=0.6).add_to(fg)
        fg.add_to(m)

    if predicted is not None:
        fg = folium.FeatureGroup(name="Predicted route")
        folium.PolyLine(_route_polyline(trip, predicted), color=COLORS["predicted"], weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name="Raw GPS")
    sub = tr.iloc[::gps_every]
    for lat, lon, t in zip(sub.lat, sub.lon, sub.t):
        folium.CircleMarker((lat, lon), radius=2, color=COLORS["gps"], fill=True, weight=0,
                            fill_opacity=0.8, tooltip=f"t = {t:.0f} s").add_to(fg)
    fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    _add_legend(m, net is not None, reference is not None, predicted is not None)
    return m


def _add_legend(m: folium.Map, net: bool, ref: bool, pred: bool) -> None:
    items = [("Raw GPS", COLORS["gps"])]
    if net:
        items.append(("Road network", COLORS["network"]))
    if ref:
        items.append(("Reference route", COLORS["reference"]))
    if pred:
        items.append(("Predicted route", COLORS["predicted"]))
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{n}</div>' for n, c in items
    )
    html = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #999;border-radius:4px;font:12px sans-serif">'
        f"<b>Legend</b>{rows}</div>"
    )
    m.get_root().html.add_child(folium.Element(html))
