"""Regenerates notebooks/01_eda.ipynb (run from the repo root). Execute it afterwards with nbconvert."""
import nbformat as nbf
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 01 · Dataset exploration

Inventory, data quality, trajectories, road networks, reference routes and candidate coverage
for the **train** (80) and **student_test** (10) partitions. Evaluator trips are not read here.

Run `scripts/01_inventory.py` and `scripts/02_candidate_sweep.py` first; this notebook reads their outputs.
Figures are written to `results/eda/`, maps to `results/maps/`.
""")
code("""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathsnap.config import RESULTS_DIR, REPO_ROOT
from pathsnap.io import load_trip, list_trips
from pathsnap.network import build_network, track_xy
from pathsnap.maps import trip_map

pd.set_option("display.width", 200, "display.max_columns", 50)
FIG = RESULTS_DIR / "eda"; FIG.mkdir(parents=True, exist_ok=True)
MAPS = RESULTS_DIR / "maps"; MAPS.mkdir(parents=True, exist_ok=True)
inv = pd.read_csv(FIG / "inventory.csv", dtype={"trip": str})
def save(fig, name): fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
""")

md("""
## 1. Inventory

Every trip folder has exactly one `.track`, `.nodes`, `.arcs` and `.route` file named after the trip id.
Each trip carries **its own road network** (a bounding box around the track), so node and arc ids are local to a trip.
""")
code("""
files = inv.groupby("partition").agg(
    trips=("trip", "size"),
    obs_total=("n_obs", "sum"),
    obs_median=("n_obs", "median"),
    nodes_median=("n_nodes", "median"),
    arcs_median=("n_arcs", "median"),
    route_arcs_median=("route_len", "median"),
    track_MB=("bytes_track", lambda s: s.sum() / 1e6),
    network_MB=("bytes_nodes", lambda s: s.sum() / 1e6),
)
files
""")
code("""
cols = ["n_obs", "duration_s", "path_km", "frac_dt_1s", "dt_max_s", "n_gaps_gt10s",
        "speed_p50_ms", "n_nodes", "n_arcs", "arc_len_p50_m", "route_len", "route_km"]
inv[cols].describe(percentiles=[.1, .5, .9]).T.round(2)
""")

md("""
## 2. Data types and missing / invalid values

All four file types are headerless and whitespace-separated. Track and node coordinates parse as float64,
arcs and routes as int64. No trip has missing values, out-of-range coordinates, duplicate rows or non-increasing timestamps.
""")
code("""
t = load_trip("00000001", "train")
print({name: df.dtypes.to_dict() for name, df in [("track", t.track), ("nodes", t.nodes), ("arcs", t.arcs)]},
      "route:", t.route.dtype)
inv[["n_missing", "n_out_of_range", "n_duplicate_rows", "n_non_increasing_t", "route_out_of_range",
     "n_isolated_nodes", "n_self_loops", "n_duplicate_arcs", "n_zero_length_arcs"]].agg(["sum", "max"]).T
""")

md("""
## 3. Sampling intervals and GPS gaps

Most trips are 1 Hz, but a minority have long gaps. A step longer than 10 s counts as a gap.
""")
code("""
dts = []
for part in ("train", "student_test"):
    for tid in list_trips(part):
        tt = load_trip(tid, part, with_route=False).track.t.to_numpy()
        dts.append(np.diff(tt))
dts = np.concatenate(dts)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
vals, cnt = np.unique(np.clip(np.round(dts), 0, 11), return_counts=True)
axes[0].bar(vals, cnt / cnt.sum() * 100); axes[0].set_yscale("log")
axes[0].set_xlabel("Δt (s), 11 = longer"); axes[0].set_ylabel("% of steps"); axes[0].set_title("Sampling interval")
axes[1].hist(inv["frac_dt_1s"] * 100, bins=20); axes[1].set_xlabel("% of steps at exactly 1 s"); axes[1].set_ylabel("trips")
axes[1].set_title("Per-trip regularity")
for part, d in inv.groupby("partition"):
    axes[2].scatter(d["n_obs"], d["dt_max_s"], s=14, label=part)
axes[2].legend()
axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlabel("observations"); axes[2].set_ylabel("largest gap (s)")
axes[2].set_title("Largest gap per trip")
fig.tight_layout(); save(fig, "sampling_intervals.png")
print(f"steps at 1 s: {(dts == 1).mean():.1%}   steps > 10 s: {(dts > 10).mean():.2%}   "
      f"non-integer steps: {(np.abs(dts - np.round(dts)) > 1e-6).sum()} of {len(dts)}")
inv.sort_values("dt_max_s", ascending=False)[["partition", "trip", "n_obs", "frac_dt_1s", "dt_max_s", "n_gaps_gt10s", "gap_time_frac"]].head(8)
""")

md("## 4. Geographical extent and trajectory patterns")
code("""
fig, ax = plt.subplots(figsize=(11, 5))
for part, mk in [("train", "o"), ("student_test", "s")]:
    d = inv[inv.partition == part]
    ax.scatter((d.lon_min + d.lon_max) / 2, (d.lat_min + d.lat_max) / 2, marker=mk, s=25, label=part)
ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Trip locations (centroid of each GPS track)"); ax.grid(alpha=.3); ax.legend()
save(fig, "trip_locations.png")
print("UTM zones used:", inv.epsg.nunique())
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
axes[0].hist(inv.n_obs, bins=30); axes[0].set_xlabel("observations per trip"); axes[0].set_ylabel("trips")
axes[1].hist(inv.path_km, bins=30); axes[1].set_xlabel("GPS path length (km)")
axes[2].hist(inv.speed_p50_ms * 3.6, bins=30); axes[2].set_xlabel("median speed (km/h)")
fig.tight_layout(); save(fig, "trajectory_stats.png")
""")
code("""
sample = ["00000001", "00000010", "00000054", "00000033"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, tid in zip(axes, sample):
    tr = load_trip(tid, "train"); net = build_network(tr); xy = track_xy(tr, net.epsg) / 1000
    lo, hi = xy.min(0) - .3, xy.max(0) + .3
    a, b = net.a / 1000, net.b / 1000
    keep = ((a >= lo) & (a <= hi)).all(1)
    ax.plot(np.stack([a[keep, 0], b[keep, 0]]), np.stack([a[keep, 1], b[keep, 1]]), color="#bbb", lw=.4)
    ra, rb = net.a[tr.route] / 1000, net.b[tr.route] / 1000
    ax.plot(np.stack([ra[:, 0], rb[:, 0]]), np.stack([ra[:, 1], rb[:, 1]]), color="#2ca02c", lw=2.5)
    ax.scatter(xy[:, 0], xy[:, 1], s=1, color="#d62728")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_title(f"trip {tid}"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("GPS (red), reference route (green), road network (grey); UTM metres"); fig.tight_layout()
save(fig, "trajectory_examples.png")
""")

md("""
## 5. Road networks: size and connectivity

Arcs are directed straight segments; most roads appear as two arcs (one per direction).
Weak components ignore direction; strong components respect it and matter for routing in the HMM.
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
axes[0].scatter(inv.n_nodes, inv.n_arcs, s=12); axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel("nodes"); axes[0].set_ylabel("arcs"); axes[0].set_title("Network size")
axes[1].hist(inv.frac_two_way * 100, bins=20); axes[1].set_xlabel("% arcs with a reverse twin"); axes[1].set_title("Two-way roads")
axes[2].hist(inv.frac_nodes_largest_scc * 100, bins=20); axes[2].set_xlabel("% nodes in largest strong component")
axes[2].set_title("Directed connectivity")
fig.tight_layout(); save(fig, "network_stats.png")
inv[["n_nodes", "n_arcs", "arc_len_p50_m", "frac_two_way", "n_weak_components",
     "n_strong_components", "frac_nodes_largest_scc", "route_in_largest_scc"]].describe(percentiles=[.1, .5, .9]).T.round(3)
""")

md("""
## 6. Reference routes

The `.route` file lists arc ids in travel order. It does not say which arc each GPS point is on, so we build
per-observation labels with a monotonic alignment (`pathsnap.labels`): each point gets a route position, positions never
decrease, and the total point-to-arc distance is minimised. The distance from a point to its label arc is a direct
measure of GPS noise plus reference-route error.
""")
code("""
inv[["route_len", "route_unique_arcs", "route_breaks", "route_km", "route_positions_used",
     "label_dist_p50_m", "label_dist_p95_m", "frac_label_dist_gt50m"]].describe(percentiles=[.1, .5, .9]).T.round(3)
""")
code("""
labs = pd.concat([pd.read_parquet(REPO_ROOT / "artifacts" / "labels" / f"{t}.parquet").assign(trip=t, partition=p)
                  for p, t in inv[["partition", "trip"]].itertuples(index=False)])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
d = labs.label_dist.clip(upper=100)
axes[0].hist(d, bins=100, density=True); axes[0].set_xlabel("distance GPS → label arc (m, clipped at 100)")
axes[0].set_ylabel("density"); axes[0].set_title("GPS offset from the reference route")
per = inv.set_index("trip").label_dist_p50_m.sort_values()
axes[1].bar(range(len(per)), per.values); axes[1].set_yscale("log"); axes[1].set_xlabel("trip (sorted)")
axes[1].set_ylabel("median offset (m)"); axes[1].set_title("Per-trip median offset")
fig.tight_layout(); save(fig, "label_offset.png")
print(labs.label_dist.describe(percentiles=[.5, .9, .95, .99]).round(2))
""")
code("""
inv.sort_values("frac_label_dist_gt50m", ascending=False)[
    ["partition", "trip", "n_obs", "path_km", "route_km", "label_dist_p50_m", "frac_label_dist_gt50m", "route_breaks"]].head(6)
""")

md("""
## 7. Candidate coverage

The candidate set for an observation is the `k` nearest arcs within `r` metres. The **candidate miss rate** is the share
of observations whose label arc is not in that set, an upper bound on what any classifier or decoder can recover.
The "clean" rate excludes observations more than 50 m from their label arc (see §6).
""")
code("""
sweep = pd.read_csv(RESULTS_DIR / "candidates" / "sweep.csv")
show = sweep[sweep.k.isin([8, 16, 32, 10000])].pivot_table(index=["radius", "k"], columns="partition",
        values=["miss_rate", "miss_rate_clean", "mean_candidates"]).round(4)
show
""")
code("""
from IPython.display import Image
Image(str(RESULTS_DIR / "candidates" / "sweep.png"))
""")

md("## 8. Interactive maps")
code("""
for part, tid in [("train", "00000001"), ("train", "00000054"), ("student_test", "00000052")]:
    tr = load_trip(tid, part); net = build_network(tr)
    trip_map(tr, net, reference=tr.route, gps_every=max(1, tr.n_obs // 2000)).save(MAPS / f"eda_{tid}.html")
sorted(p.name for p in MAPS.glob("eda_*.html"))
""")

md("""
## 9. Data-quality issues, assumptions and decisions

| Finding | Evidence | Decision |
|---|---|---|
| Trips span ~25 UTM zones worldwide | §4 | Project each trip to the UTM zone of its GPS centroid; all distances in metres |
| Each trip has its own network; ids are trip-local | §1 | Load the network per trip in training, decoding and the API |
| No missing values, invalid coordinates or out-of-order timestamps | §2 | No row dropping needed |
| Most steps are 1 s, but a few trips have very long gaps (up to ~42 h in trip 90) | §3 | Use Δt as a feature and in HMM transitions; break or relax transitions across large gaps |
| ~92 % of arcs have a reverse twin with the same geometry | §5 | Heading/travel direction features are needed to separate the two directions |
| Networks contain many small strong components | §5 | Decoder must survive unreachable transitions (fallback / restart) |
| Reference route has occasional breaks (non-adjacent consecutive arcs) | §6 | Treat as reference, not perfect truth; report it |
| 12 of ~227k time steps are non-integer (trips 2, 71, 97) | §3 | Keep timestamps as float seconds |
| Labels are not per-observation | §6 | Monotonic alignment of points to route positions |
| Trips 33 and 54: most GPS is > 50 m from the reference route (long stretches in areas with no mapped roads) | §6, map `eda_00000054.html` | Exclude observations > 50 m from their label from training; report metrics with and without them |
| ~10 % of route arcs get no GPS point (short arcs passed within 1 s) | §6 | Observation-level metrics ignore them; sequence-level metrics include them |
""")

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/01_eda.ipynb")
