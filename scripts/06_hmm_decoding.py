"""HMM + Viterbi route reconstruction, compared with classifier-only predictions.

Run after ``scripts/05_bayesian_optimisation.py`` (it writes the emission
probabilities to ``artifacts/scores/``).

Procedure
1. Decoder tuning on the 80 training trips only. Emissions there are the tuned
   ranker's out-of-fold probabilities (each trip scored by a model that did not
   train on it), so they are as noisy as on unseen trips. A grid over beta_m,
   emission_weight and disconnect_logp is decoded, and the configuration with
   the best observation-level top-1 accuracy on train is kept (ties: fewer
   breaks). Student-test is not used for this choice.
2. Final comparison with the chosen configuration on train (out-of-fold) and on
   student-test (final tuned model): classifier-only vs classifier + HMM on
   observation-level top-1, macro P/R/F1, route continuity, and sequence-level
   route metrics after the same route reconstruction (``pathsnap.route``).

The decoder never reads the reference route; labels and ``.route`` files are
read in this script for scoring only.

Outputs
- results/hmm/grid.csv                 every decoder configuration on train
- results/hmm/summary.csv              partition x method, all metrics
- results/hmm/per_trip.csv             per trip and method
- results/hmm/decoder_config.json      chosen parameters and the selection rule
- results/hmm/figures/*.png
- models/hmm_decoder.json              decoder parameters used by the API
- artifacts/hmm/<partition>/<trip>.parquet   obs, classifier arc, HMM arc
- artifacts/hmm/<partition>/<trip>_routes.json  reconstructed routes (for the maps)

Usage: uv run python scripts/06_hmm_decoding.py [--workers 6]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap.config import REPO_ROOT, RESULTS_DIR  # noqa: E402
from pathsnap.evaluate import FEATURE_DIR, LABEL_DIR, RELIABLE_LABEL_M, arc_prediction_metrics  # noqa: E402
from pathsnap.hmm import HMMParams, RoadGraph, build_lattice, classifier_only, viterbi  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.network import build_network, track_xy  # noqa: E402
from pathsnap.route import reconstruct, route_metrics  # noqa: E402

OUT = RESULTS_DIR / "hmm"
FIG = OUT / "figures"
SCORE_DIR = REPO_ROOT / "artifacts" / "scores"
PRED_DIR = REPO_ROOT / "artifacts" / "hmm"
SCORE_FILES = {"train": "train_oof.parquet", "student_test": "student_test.parquet"}

GRID = {"beta_m": [2.0, 5.0, 10.0, 25.0, 60.0],
        "emission_weight": [0.5, 1.0, 2.0],
        "disconnect_logp": [-10.0, -30.0]}


def grid_configs() -> list[HMMParams]:
    keys = list(GRID)
    return [HMMParams(**dict(zip(keys, vals))) for vals in itertools.product(*GRID.values())]


def candidates_with_scores(partition: str, trip_id: str, scores: pd.DataFrame) -> pd.DataFrame:
    f = pd.read_parquet(FEATURE_DIR / partition / f"{trip_id}.parquet",
                        columns=["obs", "arc", "position_along_arc", "candidate_rank"])
    c = f.merge(scores[["obs", "arc", "prob"]], on=["obs", "arc"], how="left", validate="one_to_one")
    if c.prob.isna().any():
        raise ValueError(f"{partition}/{trip_id}: candidates without a score")
    return c


def decode_trip(job) -> dict:
    partition, trip_id, scores, configs, final = job
    trip = load_trip(trip_id, partition, with_route=False)   # the decoder never sees the route
    net = build_network(trip)
    xy = track_xy(trip, net.epsg)
    t = trip.track.t.to_numpy(dtype=np.float64)
    cands = candidates_with_scores(partition, trip_id, scores)
    t0 = time.perf_counter()
    graph = RoadGraph(net, xy)
    lat = build_lattice(graph, xy, t, cands, configs[0])
    build_s = time.perf_counter() - t0
    clf = classifier_only(cands)
    out = {"partition": partition, "trip": trip_id, "n_obs": trip.n_obs, "u": net.u, "v": net.v,
           "classifier": clf, "lattice_seconds": build_s, "decoded": []}
    for cfg in configs:
        t1 = time.perf_counter()
        pred, breaks = viterbi(lat, cfg)
        out["decoded"].append((pred, breaks, time.perf_counter() - t1))
    if final:
        out["routes"] = {}
        for name, pred in (("classifier", clf), ("hmm", out["decoded"][0][0])):
            out["routes"][name] = reconstruct(graph, pred, t, configs[0])
        out["length"] = net.length
    return out


def load_labels(trip_id: str) -> pd.DataFrame:
    lab = pd.read_parquet(LABEL_DIR / f"{trip_id}.parquet", columns=["label_arc", "label_dist"])
    lab.insert(0, "obs", np.arange(len(lab)))
    return lab


def observation_rows(trip_id: str, labels: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    lab = labels[labels.label_dist <= RELIABLE_LABEL_M]
    o = lab.merge(pred.rename(columns={"arc": "pred_arc"}), on="obs", how="left")
    o.insert(0, "trip", trip_id)
    return o


def continuity_counts(pred: pd.DataFrame, u: np.ndarray, v: np.ndarray) -> tuple[int, int]:
    a = pred.sort_values("obs").arc.to_numpy()
    if len(a) < 2:
        return 0, 0
    ok = (a[1:] == a[:-1]) | (v[a[:-1]] == u[a[1:]])
    return int(ok.sum()), len(ok)


def run(jobs, workers):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(decode_trip, jobs))


def make_jobs(partition, scores, configs, final):
    by_trip = {k: g for k, g in scores.groupby("trip", sort=False)}
    return [(partition, tid, by_trip[tid], configs, final) for tid in list_trips(partition)]


def plot_grid(grid: pd.DataFrame, clf_top1: float, path) -> None:
    fig, axes = plt.subplots(1, len(GRID["disconnect_logp"]), figsize=(11, 3.8), sharey=True)
    for ax, dl in zip(np.atleast_1d(axes), GRID["disconnect_logp"]):
        g = grid[grid.disconnect_logp == dl].pivot(index="emission_weight", columns="beta_m", values="top1_accuracy")
        im = ax.imshow(g.values, cmap="viridis", aspect="auto", vmin=grid.top1_accuracy.min(),
                       vmax=grid.top1_accuracy.max())
        ax.set_xticks(range(g.shape[1]), [f"{c:g}" for c in g.columns])
        ax.set_yticks(range(g.shape[0]), [f"{r:g}" for r in g.index])
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g.values[i, j]:.4f}", ha="center", va="center", fontsize=7,
                        color="white" if g.values[i, j] < grid.top1_accuracy.median() else "black")
        ax.set_xlabel("beta_m")
        ax.set_title(f"disconnect_logp = {dl:g}", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("emission_weight")
    fig.colorbar(im, ax=axes, shrink=0.8, label="train top-1 (OOF emissions)")
    fig.suptitle(f"HMM decoder grid on the 80 training trips (classifier-only top-1 = {clf_top1:.4f})")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, path) -> None:
    metrics = [("top1_accuracy", "Top-1 accuracy"), ("macro_f1", "Macro F1"),
               ("route_continuity", "Route continuity"), ("route_f1", "Route F1 (length-weighted)"),
               ("route_mismatch", "Route mismatch (lower is better)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.6))
    parts = ["train", "student_test"]
    for ax, (col, title) in zip(axes, metrics):
        x = np.arange(len(parts))
        for i, (method, color) in enumerate((("classifier", "#7f7f7f"), ("hmm", "#1f77b4"))):
            vals = [summary[(summary.partition == p) & (summary.method == method)][col].iloc[0] for p in parts]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=color,
                          label="classifier only" if method == "classifier" else "classifier + HMM")
            ax.bar_label(bars, fmt="%.3f", fontsize=7)
        ax.set_xticks(x, ["train (OOF)", "student-test"])
        ax.set_title(title, fontsize=9)
        if col != "route_mismatch":
            lo = summary[col].min()
            ax.set_ylim(max(0, lo - 0.08), 1.0)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Classifier-only vs classifier + HMM/Viterbi (tuned LambdaRank emissions)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_trip(per_trip: pd.DataFrame, path) -> None:
    p = per_trip.pivot_table(index=["partition", "trip"], columns="method", values="top1_accuracy").reset_index()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for part, color in (("train", "#7f7f7f"), ("student_test", "#d62728")):
        s = p[p.partition == part]
        ax.scatter(s.classifier, s.hmm, s=22, color=color, alpha=0.8,
                   label="train (OOF)" if part == "train" else "student-test")
    lo = float(np.nanmin(p[["classifier", "hmm"]].values)) - 0.02
    ax.plot([lo, 1], [lo, 1], color="black", lw=0.8, ls="--")
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
    ax.set_xlabel("classifier-only top-1 (per trip)")
    ax.set_ylabel("classifier + HMM top-1 (per trip)")
    ax.set_title("Per-trip effect of HMM decoding (above the line = HMM helps)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    for d in (OUT, FIG, PRED_DIR / "train", PRED_DIR / "student_test", REPO_ROOT / "models"):
        d.mkdir(parents=True, exist_ok=True)
    started = time.time()
    scores = {p: pd.read_parquet(SCORE_DIR / f) for p, f in SCORE_FILES.items()}
    labels = {p: {tid: load_labels(tid) for tid in list_trips(p)} for p in SCORE_FILES}

    # 1. Grid on train.
    configs = grid_configs()
    res = run(make_jobs("train", scores["train"], configs, final=False), args.workers)
    clf_rows = pd.concat([observation_rows(r["trip"], labels["train"][r["trip"]], r["classifier"]) for r in res])
    clf_top1 = arc_prediction_metrics(clf_rows)["top1_accuracy"]
    grid = []
    for i, cfg in enumerate(configs):
        rows = pd.concat([observation_rows(r["trip"], labels["train"][r["trip"]], r["decoded"][i][0]) for r in res])
        m = arc_prediction_metrics(rows)
        grid.append({**cfg.as_dict(), "top1_accuracy": m["top1_accuracy"], "macro_f1": m["macro_f1"],
                     "breaks": sum(r["decoded"][i][1] for r in res),
                     "decode_seconds": sum(r["decoded"][i][2] for r in res)})
        print(f"beta={cfg.beta_m:5.1f} w={cfg.emission_weight:3.1f} disc={cfg.disconnect_logp:5.1f} "
              f"top1={m['top1_accuracy']:.4f} breaks={grid[-1]['breaks']}", flush=True)
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "grid.csv", index=False)
    best_row = grid.sort_values(["top1_accuracy", "breaks"], ascending=[False, True]).iloc[0]
    best = HMMParams(**{k: float(best_row[k]) for k in HMMParams().as_dict()})
    plot_grid(grid, clf_top1, FIG / "decoder_grid.png")
    print("chosen:", best, flush=True)
    decoder = {"params": best.as_dict(),
               "selection": "best observation-level top-1 on the 80 training trips with out-of-fold "
                            "LambdaRank emissions (ties: fewer breaks); student-test not used",
               "grid": GRID, "classifier_only_train_top1": clf_top1,
               "chosen_train_top1": float(best_row.top1_accuracy)}
    (OUT / "decoder_config.json").write_text(json.dumps(decoder, indent=2))
    (REPO_ROOT / "models" / "hmm_decoder.json").write_text(json.dumps(best.as_dict(), indent=2))

    # 2. Final comparison on train and student-test.
    summary, per_trip = [], []
    for part in ("train", "student_test"):
        res = run(make_jobs(part, scores[part], [best], final=True), args.workers)
        for method in ("classifier", "hmm"):
            obs_all, linked, total, routes = [], 0, 0, []
            for r in res:
                pred = r["classifier"] if method == "classifier" else r["decoded"][0][0]
                o = observation_rows(r["trip"], labels[part][r["trip"]], pred)
                obs_all.append(o)
                lk, tt = continuity_counts(pred, r["u"], r["v"])
                linked, total = linked + lk, total + tt
                ref = load_trip(r["trip"], part).route
                route, rbreaks = r["routes"][method]
                rm = route_metrics(route, ref, r["length"])
                tm = arc_prediction_metrics(o)
                row = {"partition": part, "trip": r["trip"], "method": method, "n_obs": r["n_obs"],
                       "top1_accuracy": tm["top1_accuracy"], "macro_f1": tm["macro_f1"],
                       "route_continuity": lk / tt if tt else np.nan, **rm, "route_breaks": rbreaks,
                       "decoder_breaks": r["decoded"][0][1] if method == "hmm" else np.nan,
                       "decode_seconds": (r["lattice_seconds"] + r["decoded"][0][2]) if method == "hmm" else np.nan}
                per_trip.append(row)
                routes.append(rm)
            om = arc_prediction_metrics(pd.concat(obs_all))
            pt = pd.DataFrame([x for x in per_trip if x["partition"] == part and x["method"] == method])
            n_obs = pt.n_obs.sum()
            summary.append({
                "partition": part, "method": method, **om, "route_continuity": linked / total,
                "route_precision": pt.route_precision.mean(), "route_recall": pt.route_recall.mean(),
                "route_f1": pt.route_f1.mean(), "route_mismatch": pt.route_mismatch.median(),
                "route_mismatch_mean": pt.route_mismatch.mean(),
                "trips_route_f1_ge_0.9": int((pt.route_f1 >= 0.9).sum()), "trips": len(pt),
                "route_breaks": int(pt.route_breaks.sum()),
                "decoder_breaks": pt.decoder_breaks.sum() if method == "hmm" else np.nan,
                "decode_ms_per_1k_obs": 1e6 * pt.decode_seconds.sum() / n_obs if method == "hmm" else np.nan})
        for r in res:
            d = r["classifier"].rename(columns={"arc": "classifier_arc"}).merge(
                r["decoded"][0][0].rename(columns={"arc": "hmm_arc"}), on="obs", how="outer")
            d.to_parquet(PRED_DIR / part / f"{r['trip']}.parquet", index=False)
            (PRED_DIR / part / f"{r['trip']}_routes.json").write_text(json.dumps(
                {k: {"arcs": list(map(int, v[0])), "breaks": v[1]} for k, v in r["routes"].items()}))
    summary = pd.DataFrame(summary)
    per_trip = pd.DataFrame(per_trip)
    summary.to_csv(OUT / "summary.csv", index=False)
    per_trip.to_csv(OUT / "per_trip.csv", index=False)
    plot_summary(summary, FIG / "classifier_vs_hmm.png")
    plot_per_trip(per_trip, FIG / "per_trip_top1.png")
    decoder["runtime_seconds"] = round(time.time() - started, 1)
    (OUT / "decoder_config.json").write_text(json.dumps(decoder, indent=2))
    cols = ["partition", "method", "top1_accuracy", "macro_f1", "route_continuity", "route_precision",
            "route_recall", "route_f1", "route_mismatch", "route_breaks", "decode_ms_per_1k_obs"]
    print(summary[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
