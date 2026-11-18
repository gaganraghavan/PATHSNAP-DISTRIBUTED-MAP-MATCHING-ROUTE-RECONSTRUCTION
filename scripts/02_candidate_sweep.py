"""Candidate miss rate as a function of search radius and k.

The miss rate is the fraction of observations whose label arc is not in the
candidate set; no classifier or decoder can recover those. Parameters are
chosen on train trips; student-test is reported alongside.

Outputs
- results/candidates/sweep.csv     miss rate and set size per (partition, radius, k)
- results/candidates/per_trip.csv  miss rate per trip at the default setting
- results/candidates/sweep.png

Usage: uv run python scripts/02_candidate_sweep.py   (needs scripts/01_inventory.py first)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.candidates import DEFAULT_K, DEFAULT_RADIUS_M, label_rank, within_radius
from pathsnap.config import DEV_PARTITIONS, REPO_ROOT, RESULTS_DIR
from pathsnap.io import list_trips, load_trip
from pathsnap.network import build_network, track_xy

RADII = [25, 50, 75, 100, 150, 200]
KS = [2, 4, 8, 16, 32, 64, 10_000]
MAX_R = max(RADII)
UNRELIABLE_LABEL_M = 50.0  # observations this far from their label arc are excluded from the "clean" rate


def process(args):
    partition, tid = args
    trip = load_trip(tid, partition)
    net = build_network(trip)
    xy = track_xy(trip, net.epsg)
    lab = pd.read_parquet(REPO_ROOT / "artifacts" / "labels" / f"{tid}.parquet")
    df = within_radius(xy, net, MAX_R)
    rank = label_rank(df, lab["label_arc"].to_numpy(), len(xy))
    ldist = lab["label_dist"].to_numpy()
    rows = []
    for r in RADII:
        sub = df[df["dist"] <= r]
        ranks_r = np.where(ldist <= r, rank, -1)  # label further than r is a miss
        for k in KS:
            size = (sub["rank"] < k).groupby(sub["obs"]).sum().reindex(range(len(xy)), fill_value=0)
            rows.append({
                "partition": partition, "trip": tid, "radius": r, "k": k, "n_obs": len(xy),
                "n_miss": int(((ranks_r < 0) | (ranks_r >= k)).sum()),
                "n_clean": int((ldist <= UNRELIABLE_LABEL_M).sum()),
                "n_miss_clean": int((((ranks_r < 0) | (ranks_r >= k)) & (ldist <= UNRELIABLE_LABEL_M)).sum()),
                "n_empty": int((size == 0).sum()),
                "cand_sum": int(size.sum()),
            })
    return rows


def main() -> None:
    out = RESULTS_DIR / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, t) for p in DEV_PARTITIONS for t in list_trips(p)]
    with ProcessPoolExecutor(6) as ex:
        per = pd.DataFrame([r for rows in ex.map(process, jobs) for r in rows])

    per.query("radius == @DEFAULT_RADIUS_M and k == @DEFAULT_K").assign(
        miss_rate=lambda d: d.n_miss / d.n_obs
    ).to_csv(out / "per_trip.csv", index=False)

    g = per.groupby(["partition", "radius", "k"])[["n_obs", "n_miss", "n_clean", "n_miss_clean", "n_empty", "cand_sum"]].sum()
    sweep = pd.DataFrame({
        "miss_rate": g.n_miss / g.n_obs,
        "miss_rate_clean": g.n_miss_clean / g.n_clean,
        "empty_rate": g.n_empty / g.n_obs,
        "mean_candidates": g.cand_sum / g.n_obs,
    }).reset_index()
    sweep.to_csv(out / "sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    tr = sweep[sweep.partition == "train"]
    for r, d in tr.groupby("radius"):
        axes[0].plot(d["mean_candidates"], 100 * d["miss_rate"], marker="o", label=f"r = {r} m")
        axes[1].plot(d["mean_candidates"], 100 * d["miss_rate_clean"], marker="o", label=f"r = {r} m")
    for ax, title in zip(axes, ["All observations", f"Label within {UNRELIABLE_LABEL_M:.0f} m"]):
        ax.set_xscale("log")
        ax.set_xlabel("Mean candidates per observation (k = 2 … all)")
        ax.set_ylabel("Candidate miss rate (%)")
        ax.set_title(f"Train trips: {title}")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=130)

    pd.set_option("display.width", 200)
    print(sweep.query("k in [8, 16, 32, 10000]").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
