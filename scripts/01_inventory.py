"""Build the dataset inventory and per-observation labels for all dev trips.

Outputs
- results/eda/inventory.csv           one row per trip (EDA / data-quality table)
- artifacts/labels/<trip>.parquet     per-observation route alignment (train + student_test)

Usage: uv run python scripts/01_inventory.py [--workers N]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from pathsnap.config import DEV_PARTITIONS, REPO_ROOT, RESULTS_DIR
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import observation_labels
from pathsnap.network import build_network, track_xy
from pathsnap.validate import trip_summary

LABEL_DIR = REPO_ROOT / "artifacts" / "labels"


def process(args: tuple[str, str]) -> dict:
    partition, tid = args
    t0 = time.time()
    trip = load_trip(tid, partition)
    net = build_network(trip)
    xy = track_xy(trip, net.epsg)
    labels = observation_labels(xy, net, trip.route) if trip.route is not None else None
    if labels is not None:
        pd.DataFrame(labels).to_parquet(LABEL_DIR / f"{tid}.parquet", index=False)
    row = trip_summary(trip, net, xy, labels)
    row["process_s"] = round(time.time() - t0, 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, t) for p in DEV_PARTITIONS for t in list_trips(p)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(process, jobs))
    df = pd.DataFrame(rows).sort_values(["partition", "trip"])
    df.to_csv(out / "inventory.csv", index=False)
    print(f"{len(df)} trips -> {out / 'inventory.csv'}")


if __name__ == "__main__":
    main()
