"""Compare seven candidate-ranking models with trip-grouped CV and on student-test.

Run after ``scripts/03_feature_engineering.py``.

Procedure
1. The 80 training trips are split into 5 GroupKFold folds (group = trip).
   In each fold, features are selected and every model is fitted on the
   training trips only, then scored on the held-out trips.
2. Features are selected again on all 80 training trips, every model is refitted
   on them, and the refitted models are scored on the 10 student-test trips.

Training rows: candidates of observations whose reference label is within 50 m
and present in the candidate set. At most ``--max-train-obs`` observations are
sampled per fit (all candidates of a sampled observation are kept, so the class
balance stays as it is at prediction time). Scoring always uses every candidate
of the held-out trips. Metric definitions are in ``pathsnap.evaluate``.

Primary metric: observation-level top-1 accuracy, mean over the CV folds.

Outputs
- results/models/cv_folds.csv            per fold x model metrics, fit/predict time
- results/models/model_comparison.csv    CV mean/std + student-test metrics per model
- results/models/student_test_per_trip.csv
- results/models/confusion_matrices.csv  candidate-level, student-test, threshold 0.5
- results/models/histories/<model>.csv   per-epoch train/validation log loss (HGB, MLP)
- results/models/run_config.json         seed, sizes, model configs, selected features, versions
- results/models/figures/*.png
- artifacts/models/<model>.joblib        refitted models (git-ignored)

Usage: uv run python scripts/04_model_comparison.py [--folds 5] [--max-train-obs 20000]
       uv run python scripts/04_model_comparison.py --feature-set pair   # ablation without context features
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from pathsnap.config import REPO_ROOT, RESULTS_DIR, SEED
from pathsnap.evaluate import (
    Partition,
    candidate_confusion,
    evaluate,
    load_partition,
    observation_table,
    reference_continuity,
)
from pathsnap.features import CONTEXT_FEATURES, MODEL_FEATURES
from pathsnap.models import fit, history, model_specs, positive_proba, select_features

FEATURE_SETS = {
    "all": list(MODEL_FEATURES),
    # Ablation: per-pair features only, without the step-03 context features.
    "pair": [f for f in MODEL_FEATURES if f not in CONTEXT_FEATURES],
}


def training_rows(part: Partition, max_obs: int, rng: np.random.Generator) -> pd.DataFrame:
    """Candidates of sampled reliable observations whose reference arc is a candidate."""
    c = part.candidates
    c = c[c.label_reliable_50m == 1]
    hit = c.loc[c.is_label == 1, ["trip", "obs"]].drop_duplicates()
    if len(hit) > max_obs:
        hit = hit.iloc[np.sort(rng.choice(len(hit), size=max_obs, replace=False))]
    return c.merge(hit, on=["trip", "obs"], how="inner")


def train_fit_metrics(rows: pd.DataFrame, part: Partition, scores: np.ndarray) -> dict[str, float]:
    """Top-1 and log loss on the fitted observations themselves (for over/underfitting)."""
    keys = rows[["trip", "obs"]].drop_duplicates()
    sub = Partition(rows.reset_index(drop=True),
                    part.labels.merge(keys, on=["trip", "obs"]), part.arc_ends)
    obs = observation_table(sub, scores)
    return {"train_top1_accuracy": float(obs.correct.mean()),
            "train_observation_log_loss": float(-np.log(np.clip(obs.true_prob, 1e-6, 1)).mean())}


def run_models(train: Partition, test: Partition, args, rng, fold_label) -> tuple[list[dict], dict]:
    rows = training_rows(train, args.max_train_obs, rng)
    X_all = rows[FEATURE_SETS[args.feature_set]].astype(np.float32)
    y = rows.is_label.to_numpy()
    groups = rows.trip.astype(str).to_numpy()
    query = rows.groupby(["trip", "obs"], observed=True, sort=False).ngroup().to_numpy()
    test_query = test.candidates.groupby(["trip", "obs"], observed=True, sort=False).ngroup().to_numpy()
    feats = select_features(X_all, y, args.seed)
    X = X_all[feats]
    X_test = test.candidates[feats].astype(np.float32)
    n_test_obs = int(test.labels.shape[0])
    results, fitted = [], {}
    for spec in model_specs(args.seed):
        t0 = time.perf_counter()
        model = fit(spec, X, y, groups, query)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = positive_proba(model, X_test, test_query)
        predict_s = time.perf_counter() - t0
        m = evaluate(test, scores)
        m.update(train_fit_metrics(rows, train, positive_proba(model, X, query)))
        m.update({"fold": fold_label, "model": spec.name, "family": spec.family,
                  "train_observations": int(rows[["trip", "obs"]].drop_duplicates().shape[0]),
                  "train_rows": len(rows), "fit_seconds": fit_s, "predict_seconds": predict_s,
                  "predict_ms_per_1k_obs": 1000 * predict_s / max(n_test_obs, 1) * 1000,
                  "features": ",".join(feats)})
        results.append(m)
        fitted[spec.name] = (model, scores)
        print(f"[{fold_label}] {spec.name:28s} top1={m['top1_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"obsLL={m['observation_log_loss']:.3f} cont={m['route_continuity']:.3f} "
              f"fit={fit_s:.1f}s pred={predict_s:.1f}s", flush=True)
    return results, {"features": feats, "models": fitted}


def summarise(cv: pd.DataFrame, test: pd.DataFrame, specs) -> pd.DataFrame:
    keys = ["top1_accuracy", "macro_f1", "macro_precision", "macro_recall", "observation_log_loss",
            "candidate_log_loss", "route_continuity", "train_top1_accuracy"]
    agg = cv.groupby("model")[keys].agg(["mean", "std"])
    agg.columns = [f"cv_{k}_{s}" for k, s in agg.columns]
    t = test.set_index("model")
    tcols = ["top1_accuracy", "macro_f1", "macro_precision", "macro_recall", "observation_log_loss",
             "candidate_log_loss", "candidate_roc_auc", "candidate_precision", "candidate_recall",
             "candidate_f1", "route_continuity", "train_top1_accuracy", "fit_seconds",
             "predict_seconds", "predict_ms_per_1k_obs"]
    # Paired view: folds share trips across models, so compare within each fold.
    per_fold = cv.pivot(index="fold", columns="model", values="top1_accuracy")
    leader = per_fold.mean().idxmax()
    agg["cv_top1_folds_best"] = per_fold.eq(per_fold.max(axis=1), axis=0).sum()
    agg["cv_top1_paired_gap_to_leader_mean"] = per_fold.sub(per_fold[leader], axis=0).mean()
    agg["cv_top1_paired_gap_to_leader_std"] = per_fold.sub(per_fold[leader], axis=0).std()
    out = agg.join(t[tcols].add_prefix("test_"))
    meta = pd.DataFrame([{"model": s.name, "family": s.family, "configuration": s.config} for s in specs])
    out = meta.set_index("model").join(out).reset_index()
    return out.sort_values("cv_top1_accuracy_mean", ascending=False, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-train-obs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="all")
    ap.add_argument("--output", default=None,
                    help="results folder (default results/models, or results/models/ablation_<set>)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    default = "models" if args.feature_set == "all" else f"models/ablation_{args.feature_set}_features"
    OUT = RESULTS_DIR / default if args.output is None else Path(args.output)
    MODEL_DIR = REPO_ROOT / "artifacts" / default
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "histories").mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train = load_partition("train")
    test = load_partition("student_test")
    trips = train.trips
    folds = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    cv_rows, fold_features = [], {}
    for k, (tr, va) in enumerate(folds.split(trips, groups=trips), start=1):
        res, info = run_models(train.subset(trips[tr]), train.subset(trips[va]), args, rng, f"fold{k}")
        for r in res:
            r["validation_trips"] = ",".join(trips[va])
        cv_rows += res
        fold_features[f"fold{k}"] = info["features"]
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT / "cv_folds.csv", index=False)

    test_rows, final = run_models(train, test, args, rng, "student_test")
    test_df = pd.DataFrame(test_rows)
    specs = model_specs(args.seed)
    summary = summarise(cv, test_df, specs)
    summary.to_csv(OUT / "model_comparison.csv", index=False)

    per_trip, confusion = [], []
    for name, (model, scores) in final["models"].items():
        obs = observation_table(test, scores)
        acc = obs.groupby("trip").correct.agg(["mean", "size"]).reset_index()
        acc.columns = ["trip", "top1_accuracy", "observations"]
        acc.insert(0, "model", name)
        per_trip.append(acc)
        (tn, fp), (fn, tp) = candidate_confusion(test, scores)
        confusion.append({"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        h = history(model)
        if h is not None:
            h.to_csv(OUT / "histories" / f"{name.lower().replace(' ', '_')}.csv", index=False)
        joblib.dump({"model": model, "features": final["features"], "model_name": name},
                    MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")
    pd.concat(per_trip).to_csv(OUT / "student_test_per_trip.csv", index=False)
    pd.DataFrame(confusion).to_csv(OUT / "confusion_matrices.csv", index=False)

    best = summary.iloc[0].model
    config = {
        "seed": args.seed,
        "folds": args.folds,
        "fold_assignment": "GroupKFold(shuffle=True) over the 80 training trips, group = trip",
        "max_train_obs_per_fit": args.max_train_obs,
        "feature_set": args.feature_set,
        "candidate_features": FEATURE_SETS[args.feature_set],
        "primary_metric": "CV mean observation-level top-1 accuracy",
        "selected_model": best,
        "feature_selection": "ExtraTrees impurity importance, median threshold, refitted per fold",
        "features_per_fold": fold_features,
        "features_final": final["features"],
        "reference_label_continuity": {"train": reference_continuity(train),
                                       "student_test": reference_continuity(test)},
        "models": [{"name": s.name, "family": s.family, "configuration": s.config,
                    "params": {k: repr(v) for k, v in s.estimator.get_params(deep=False).items()}}
                   for s in specs],
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2))

    from plot_model_comparison import plot_all  # noqa: E402  (sibling script)
    plot_all(OUT)
    print(summary[["model", "cv_top1_accuracy_mean", "cv_top1_accuracy_std", "cv_macro_f1_mean",
                   "test_top1_accuracy", "test_macro_f1", "test_observation_log_loss",
                   "test_route_continuity"]].round(4).to_string(index=False))
    print("Selected (CV top-1):", best)


if __name__ == "__main__":
    main()
