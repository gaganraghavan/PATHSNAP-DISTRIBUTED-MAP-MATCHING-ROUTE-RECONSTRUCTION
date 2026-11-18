"""Folium maps of the reconstructed routes on the student-test trips.

Run after ``scripts/06_hmm_decoding.py``. Each map has layer control and a
legend with: raw GPS observations, the road network near the track, the
reference route, the classifier-only route and the classifier + HMM route
(both reconstructed by ``pathsnap.route``), plus an info box with that trip's
metrics from ``results/hmm/per_trip.csv``. A static PNG of each trip (projected
metres) is drawn too, for the report.

Outputs
- results/maps/route_<trip>.html   interactive maps
- results/maps/route_<trip>.png    static views
- results/maps/index.html          links to all maps with their metrics

Usage: uv run python scripts/07_route_maps.py [--partition student_test] [--trips 00000007 ...]
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap.config import REPO_ROOT, RESULTS_DIR  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.maps import COLORS, PREDICTION_STYLES, nearby_arcs, trip_map  # noqa: E402
from pathsnap.network import build_network, track_xy  # noqa: E402

PRED_DIR = REPO_ROOT / "artifacts" / "hmm"
OUT = RESULTS_DIR / "maps"
LAYERS = {"hmm": "Classifier + HMM route", "classifier": "Classifier-only route"}


def info(trip_id: str, rows: pd.DataFrame) -> str:
    h, c = rows.loc["hmm"], rows.loc["classifier"]
    return (f"<b>Trip {trip_id}</b> (student-test)<br>"
            "<table style='font:12px sans-serif'><tr><th></th><th>clf only</th><th>clf + HMM</th></tr>"
            f"<tr><td>Top-1</td><td>{c.top1_accuracy:.3f}</td><td>{h.top1_accuracy:.3f}</td></tr>"
            f"<tr><td>Route F1</td><td>{c.route_f1:.3f}</td><td>{h.route_f1:.3f}</td></tr>"
            f"<tr><td>Mismatch</td><td>{c.route_mismatch:.3f}</td><td>{h.route_mismatch:.3f}</td></tr>"
            f"<tr><td>Breaks</td><td>{int(c.route_breaks)}</td><td>{int(h.route_breaks)}</td></tr></table>")


def static_view(trip, net, routes: dict, ref, title: str, path) -> None:
    xy = track_xy(trip, net.epsg)
    a, b = net.node_xy[net.u], net.node_xy[net.v]
    fig, ax = plt.subplots(figsize=(8, 8))
    near = nearby_arcs(trip, net, 150.0)
    for i in near:
        ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color=COLORS["network"], lw=0.4, alpha=0.6, zorder=1)

    def draw(route, **kw):
        pts = np.concatenate([np.stack([a[r], b[r]]) for r in route]) if len(route) else np.empty((0, 2))
        ax.plot(pts[:, 0], pts[:, 1], **kw)

    draw(ref, color=COLORS["reference"], lw=5, alpha=0.45, label="Reference route", zorder=2)
    for (name, route), (color, weight, dash) in zip(routes.items(), PREDICTION_STYLES):
        draw(route, color=color, lw=weight / 1.5, ls="-" if dash is None else "--", label=name, zorder=3)
    ax.scatter(xy[:, 0], xy[:, 1], s=1.5, color=COLORS["gps"], label="Raw GPS", zorder=4)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UTM easting (m)")
    ax.set_ylabel("UTM northing (m)")
    ax.legend(loc="best", fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trips", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per_trip = pd.read_csv(RESULTS_DIR / "hmm" / "per_trip.csv", dtype={"trip": str})
    per_trip = per_trip[per_trip.partition == args.partition]
    links = []
    for tid in args.trips or list_trips(args.partition):
        trip = load_trip(tid, args.partition)
        net = build_network(trip)
        saved = json.loads((PRED_DIR / args.partition / f"{tid}_routes.json").read_text())
        routes = {LAYERS[k]: saved[k]["arcs"] for k in ("hmm", "classifier")}
        rows = per_trip[per_trip.trip == tid].set_index("method")
        m = trip_map(trip, net, reference=trip.route, predicted=routes,
                     gps_every=max(1, trip.n_obs // 2000), info_html=info(tid, rows))
        m.save(OUT / f"route_{tid}.html")
        h, c = rows.loc["hmm"], rows.loc["classifier"]
        static_view(trip, net, routes, trip.route,
                    f"Trip {tid}: top-1 {c.top1_accuracy:.3f} -> {h.top1_accuracy:.3f}, "
                    f"route F1 {c.route_f1:.3f} -> {h.route_f1:.3f} (classifier-only -> HMM)",
                    OUT / f"route_{tid}.png")
        links.append((tid, c, h))
        print(f"{tid}: map written", flush=True)
    rows_html = "".join(
        f"<tr><td><a href='route_{t}.html'>{t}</a></td><td>{c.top1_accuracy:.3f}</td><td>{h.top1_accuracy:.3f}</td>"
        f"<td>{c.route_f1:.3f}</td><td>{h.route_f1:.3f}</td></tr>" for t, c, h in links)
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>PathSnap route maps</title>"
        "<style>body{font:14px sans-serif;margin:24px}td,th{padding:4px 10px;border-bottom:1px solid #ddd}</style>"
        f"<h2>PathSnap route maps ({args.partition})</h2><table><tr><th>Trip</th><th>Top-1 clf</th>"
        f"<th>Top-1 HMM</th><th>Route F1 clf</th><th>Route F1 HMM</th></tr>{rows_html}</table>")


if __name__ == "__main__":
    main()
