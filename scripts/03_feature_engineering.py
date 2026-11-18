"""Build observation-candidate features for train and student-test trips.

Run after ``scripts/01_inventory.py`` so the monotonic route-alignment labels
exist. This step reuses the project's UTM projection, candidate search, and
label alignment; it does not change or rerun the existing EDA outputs.

Outputs
- artifacts/features/<partition>/<trip>.parquet  regenerable per-candidate rows
- results/features/feature_summary.csv           compact partition-level summary
- results/features/feature_schema.csv            model feature definitions

Usage: uv run python scripts/03_feature_engineering.py [--workers N]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from pathsnap.candidates import DEFAULT_K, DEFAULT_RADIUS_M, generate
from pathsnap.config import DEV_PARTITIONS, REPO_ROOT, RESULTS_DIR
from pathsnap.features import MODEL_FEATURES, candidate_feature_frame, observation_candidate_misses
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import observation_labels
from pathsnap.network import build_network, track_xy

ARTIFACTS = REPO_ROOT / "artifacts" / "features"
FEATURE_DEFINITIONS = {
    "point_to_arc_m": "Projected point-to-segment distance for this GPS/candidate pair (metres).",
    "arc_length_m": "Length of the directed candidate arc between its two nodes (metres).",
    "candidate_rank": "Zero-based distance rank within the observation's candidate set; ties share a rank.",
    "position_along_arc": "Projection fraction along the directed candidate arc, from 0 at its start to 1 at its end.",
    "heading_delta_sin": "Sine of candidate-arc bearing minus local GPS movement bearing.",
    "heading_delta_cos": "Cosine of candidate-arc bearing minus local GPS movement bearing.",
    "gps_step_m": "Distance from the previous GPS observation in the trip's UTM projection (metres).",
    "gps_gap_s": "Timestamp difference from the previous observation (seconds).",
    "gps_speed_mps": "GPS step distance divided by the elapsed seconds, with a 1-second floor.",
    "start_node_degree": "In-degree plus out-degree of the candidate arc's start node.",
    "end_node_degree": "In-degree plus out-degree of the candidate arc's end node.",
}


def process(job: tuple[str, str]) -> dict:
    partition, trip_id = job
    trip = load_trip(trip_id, partition)
    if trip.route is None:
        raise ValueError(f"{partition}/{trip_id} has no .route reference file")
    net = build_network(trip)
    xy = track_xy(trip, net.epsg)
    labels = observation_labels(xy, net, trip.route)
    candidates = generate(xy, net, DEFAULT_RADIUS_M, DEFAULT_K)
    frame = candidate_feature_frame(trip, net, xy, candidates, labels)
    miss, reliable = observation_candidate_misses(trip.n_obs, candidates, labels)
    out = ARTIFACTS / partition
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / f"{trip_id}.parquet", index=False)

    n_clean = int(reliable.sum())
    return {
        "partition": partition,
        "trip": trip_id,
        "n_observations": trip.n_obs,
        "n_candidates": len(candidates.obs),
        "mean_candidates": len(candidates.obs) / max(trip.n_obs, 1),
        "candidate_miss_count": int(miss.sum()),
        "candidate_miss_rate": float(miss.mean()),
        "reliable_observations_le_50m": n_clean,
        "candidate_miss_count_le_50m": int((miss & reliable).sum()),
        "candidate_miss_rate_le_50m": float((miss & reliable).sum() / n_clean) if n_clean else np.nan,
        "positive_candidate_rows": int(frame["is_label"].sum()) if len(frame) else 0,
        "label_reliable_fraction": float(reliable.mean()) if len(reliable) else np.nan,
        "mean_candidate_distance_m": float(frame["point_to_arc_m"].mean()) if len(frame) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = [(p, t) for p in DEV_PARTITIONS for t in list_trips(p)]
    with ProcessPoolExecutor(a.workers) as executor:
        rows = [r for r in executor.map(process, jobs)]
    out = RESULTS_DIR / "features"
    out.mkdir(parents=True, exist_ok=True)
    per_trip = pd.DataFrame(rows).sort_values(["partition", "trip"])
    per_trip.to_csv(out / "per_trip.csv", index=False)
    summary = per_trip.groupby("partition", sort=True).agg(
        trips=("trip", "count"),
        observations=("n_observations", "sum"),
        candidates=("n_candidates", "sum"),
        candidate_miss_count=("candidate_miss_count", "sum"),
        reliable_observations_le_50m=("reliable_observations_le_50m", "sum"),
        reliable_candidate_miss_count=("candidate_miss_count_le_50m", "sum"),
        positive_candidate_rows=("positive_candidate_rows", "sum"),
        mean_candidates_per_observation=("mean_candidates", "mean"),
        mean_candidate_distance_m=("mean_candidate_distance_m", "mean"),
    ).reset_index()
    summary["candidate_miss_rate"] = summary.candidate_miss_count / summary.observations
    summary["candidate_miss_rate_le_50m"] = (
        summary.reliable_candidate_miss_count / summary.reliable_observations_le_50m
    )
    summary.to_csv(out / "feature_summary.csv", index=False)
    pd.DataFrame([{"feature": name, "definition": FEATURE_DEFINITIONS[name]}
                  for name in MODEL_FEATURES]).to_csv(out / "feature_schema.csv", index=False)
    print(f"Built candidate features for {len(per_trip)} trips -> {out}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
