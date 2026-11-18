"""Bayesian hyperparameter optimisation of the selected model, LightGBM LambdaRank.

Run after ``scripts/03_feature_engineering.py``.

Procedure
1. The same 5 trip-grouped folds, training samples and per-fold feature
   selection as step 04 (same seed, same order of draws from the generator), so
   the untuned numbers here reproduce step 04's LambdaRank row.
2. The untuned configuration is scored on those folds as the baseline.
3. Optuna's TPE sampler (multivariate, seeded) searches the space in
   ``SEARCH_SPACE``. The objective is the mean observation-level top-1 accuracy
   over the 5 held-out folds, the Phase 1 primary metric. A median pruner stops
   a trial early when its running fold mean falls below the median of earlier
   trials at the same fold. Only the 80 training trips are used.
4. The untuned and the best configuration are refitted on all 80 training trips
   and scored on the 10 student-test trips with the full metric set.
5. Inputs for the HMM (step 06): out-of-fold probabilities of the tuned model on
   the training trips (each trip scored by the fold model that did not see it),
   and the final tuned model's probabilities on student-test.

Outputs
- results/tuning/trials.csv              every trial: params, state, fold scores, mean
- results/tuning/search_space.json       distributions, sampler, pruner, objective
- results/tuning/best_params.json
- results/tuning/cv_folds.csv            per-fold metrics, untuned vs tuned
- results/tuning/untuned_vs_tuned.csv    CV and student-test metrics, fit/inference time
- results/tuning/run_config.json
- results/tuning/figures/*.png
- models/lambdarank_tuned.joblib, models/lambdarank_tuned.txt, models/lambdarank_tuned.json
- artifacts/scores/{train_oof,student_test}.parquet   (trip, obs, arc, prob) for step 06
- results/tuning/optuna.db               study storage; rerun with a larger --trials to resume

Usage: uv run python scripts/05_bayesian_optimisation.py [--trials 30]
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import time

import joblib
import lightgbm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.config import REPO_ROOT, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.evaluate import Partition, evaluate, load_partition, observation_metrics  # noqa: E402
from pathsnap.features import MODEL_FEATURES  # noqa: E402
from pathsnap.models import SoftmaxRanker, select_features  # noqa: E402

training_rows = importlib.import_module("04_model_comparison").training_rows

OUT = RESULTS_DIR / "tuning"
FIG = OUT / "figures"
MODEL_DIR = REPO_ROOT / "models"
SCORE_DIR = REPO_ROOT / "artifacts" / "scores"
# Committed, so the search can be resumed on another machine.
STUDY_DIR = OUT

UNTUNED = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
           "truncation_level": 5, "colsample_bytree": 1.0, "subsample": 1.0,
           "reg_alpha": 0.0, "reg_lambda": 0.0}

# name -> (kind, low, high, log scale)
SEARCH_SPACE = {
    "n_estimators": ("int", 100, 1500, True),
    "learning_rate": ("float", 0.01, 0.2, True),
    "num_leaves": ("int", 7, 255, True),
    "min_child_samples": ("int", 5, 500, True),
    "truncation_level": ("int", 1, 32, False),
    "colsample_bytree": ("float", 0.4, 1.0, False),
    "subsample": ("float", 0.4, 1.0, False),
    "reg_alpha": ("float", 1e-4, 10.0, True),
    "reg_lambda": ("float", 1e-4, 30.0, True),
}


class Fold:
    """Training rows and held-out candidates of one CV fold (or of the final fit)."""

    def __init__(self, train: Partition, held_out: Partition, max_obs: int, rng, seed: int):
        rows = training_rows(train, max_obs, rng)
        X_all = rows[list(MODEL_FEATURES)].astype(np.float32)
        self.y = rows.is_label.to_numpy()
        self.features = select_features(X_all, self.y, seed)
        self.X = X_all[self.features]
        self.query = rows.groupby(["trip", "obs"], observed=True, sort=False).ngroup().to_numpy()
        self.held_out = held_out
        self.X_test = held_out.candidates[self.features].astype(np.float32)
        self.test_query = held_out.candidates.groupby(["trip", "obs"], observed=True,
                                                      sort=False).ngroup().to_numpy()

    def fit(self, params: dict, seed: int) -> tuple[SoftmaxRanker, float]:
        t0 = time.perf_counter()
        model = SoftmaxRanker(**params, random_state=seed).fit(self.X, self.y, self.query)
        return model, time.perf_counter() - t0

    def score(self, model: SoftmaxRanker) -> np.ndarray:
        return model.predict_proba(self.X_test, self.test_query)[:, 1]


def suggest(trial: optuna.Trial) -> dict:
    p = {}
    for name, (kind, lo, hi, log) in SEARCH_SPACE.items():
        f = trial.suggest_int if kind == "int" else trial.suggest_float
        p[name] = f(name, lo, hi, log=log)
    return p


def cross_validate(folds: list[Fold], params: dict, seed: int, trial: optuna.Trial | None = None):
    rows = []
    for k, fold in enumerate(folds, start=1):
        model, fit_s = fold.fit(params, seed)
        m = observation_metrics(fold.held_out, fold.score(model))
        m.update({"fold": k, "fit_seconds": fit_s})
        rows.append(m)
        if trial is not None:
            running = float(np.mean([r["top1_accuracy"] for r in rows]))
            trial.report(running, step=k)
            if trial.should_prune():
                trial.set_user_attr("folds", rows)
                raise optuna.TrialPruned()
    return rows


def predict_timed(model: SoftmaxRanker, fold: Fold, repeats: int = 3) -> tuple[np.ndarray, float]:
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        scores = fold.score(model)
        best = min(best, time.perf_counter() - t0)
    return scores, best


def score_frame(part: Partition, scores: np.ndarray) -> pd.DataFrame:
    out = part.candidates[["trip", "obs", "arc"]].copy()
    out["trip"] = out.trip.astype(str)
    out["prob"] = scores.astype(np.float32)
    return out


def plot_history(trials: pd.DataFrame, baseline: float, path) -> None:
    trials = trials[trials.state.isin(["COMPLETE", "PRUNED"])]
    done = trials[trials.state == "COMPLETE"]
    pruned = trials[trials.state == "PRUNED"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(done.number, done.value, s=18, color="#1f77b4", label="completed trial (CV top-1)")
    best = done.set_index("number").value.reindex(trials.number).ffill().cummax()
    ax.step(trials.number, best, where="post", color="#d62728", label="best so far")
    ax.axhline(baseline, color="#2ca02c", ls="--", lw=1, label=f"untuned ({baseline:.4f})")
    lo = min(baseline, done.value.min()) - 0.002
    ax.set_ylim(lo, done.value.max() + 0.002)
    if len(pruned):
        # Pruned trials have no 5-fold score (their running mean covers only the
        # easier first folds), so they are marked along the bottom edge only.
        ax.scatter(pruned.number, np.full(len(pruned), lo + 0.0004), s=22, marker="x", color="#999999",
                   label="pruned trial (stopped after 2-3 folds)")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV mean top-1 accuracy (5 trip folds)")
    ax.set_title("Bayesian optimisation of LightGBM LambdaRank (TPE)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importance: dict, path) -> None:
    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#1f77b4")
    ax.set_xlabel("fANOVA importance for CV top-1 (completed trials)")
    ax.set_title("Hyperparameter importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(table: pd.DataFrame, path) -> None:
    metrics = [("cv_macro_f1_mean", "CV macro F1"), ("test_top1_accuracy", "Student-test top-1"),
               ("test_macro_f1", "Student-test macro F1"), ("test_observation_log_loss", "Student-test obs log loss"),
               ("test_predict_ms_per_1k_obs", "Inference ms / 1k obs")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.4))
    colors = {"untuned": "#7f7f7f", "tuned": "#1f77b4"}
    for ax, (col, title) in zip(axes, metrics):
        vals = table.set_index("configuration")[col]
        bars = ax.bar(vals.index, vals.values, color=[colors[i] for i in vals.index])
        ax.bar_label(bars, fmt="%.4f" if vals.max() < 10 else "%.1f", fontsize=8)
        ax.set_title(title, fontsize=9)
        if vals.max() < 1.01 and "loss" not in col:
            ax.set_ylim(vals.min() - 0.03, min(1.0, vals.max() + 0.02))
    fig.suptitle("LightGBM LambdaRank: untuned vs tuned (same folds, same partitions)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-train-obs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    for d in (OUT, FIG, MODEL_DIR, SCORE_DIR, STUDY_DIR):
        d.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(args.seed)

    train = load_partition("train")
    test = load_partition("student_test")
    trips = train.trips
    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = [Fold(train.subset(trips[tr]), train.subset(trips[va]), args.max_train_obs, rng, args.seed)
             for tr, va in splitter.split(trips, groups=trips)]
    final = Fold(train, test, args.max_train_obs, rng, args.seed)
    print("features per fold:", [f.features for f in folds], "final:", final.features, flush=True)

    # Baseline: untuned configuration on the same folds.
    untuned_cv = cross_validate(folds, UNTUNED, args.seed)
    base = float(np.mean([r["top1_accuracy"] for r in untuned_cv]))
    print(f"untuned CV top-1 = {base:.4f}", flush=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, n_startup_trials=15)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(study_name="lambdarank_top1", direction="maximize", sampler=sampler,
                                pruner=pruner, storage=f"sqlite:///{(STUDY_DIR / 'optuna.db').as_posix()}",
                                load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        rows = cross_validate(folds, suggest(trial), args.seed, trial)
        trial.set_user_attr("folds", rows)
        return float(np.mean([r["top1_accuracy"] for r in rows]))

    # A run that was stopped mid-trial leaves that trial RUNNING in storage: mark it failed.
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    remaining = args.trials - len(study.get_trials(deepcopy=False, states=finished))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining,
                       callbacks=[lambda s, t: print(f"trial {t.number:3d} {t.state.name:8s} value={t.value} "
                                                     f"best={s.best_value:.4f}", flush=True)])

    # Trial record.
    recs = []
    for t in study.trials:
        fr = t.user_attrs.get("folds", [])
        rec = {"number": t.number, "state": t.state.name, "value": t.value,
               "duration_s": t.duration.total_seconds() if t.duration else None,
               "folds_run": len(fr),
               "last_running_mean": float(np.mean([r["top1_accuracy"] for r in fr])) if fr else None}
        rec.update(t.params)
        for r in fr:
            rec[f"fold{r['fold']}_top1"] = r["top1_accuracy"]
            rec[f"fold{r['fold']}_macro_f1"] = r["macro_f1"]
            rec[f"fold{r['fold']}_obs_log_loss"] = r["observation_log_loss"]
        recs.append(rec)
    trials = pd.DataFrame(recs)
    trials.to_csv(OUT / "trials.csv", index=False)

    best_params = dict(study.best_params)
    tuned_cv = study.best_trial.user_attrs["folds"]
    (OUT / "best_params.json").write_text(json.dumps(
        {"trial": study.best_trial.number, "cv_top1_mean": study.best_value, "params": best_params}, indent=2))
    (OUT / "search_space.json").write_text(json.dumps({
        "objective": "maximise mean observation-level top-1 accuracy over 5 trip-grouped CV folds (80 training trips)",
        "sampler": "optuna TPESampler(multivariate=True, n_startup_trials=15, seed)",
        "pruner": "MedianPruner(n_startup_trials=10, n_warmup_steps=2) on the running fold mean",
        "n_trials_requested": args.trials, "n_trials_recorded": len(study.trials), "seed": args.seed,
        "space": {k: {"type": kind, "low": lo, "high": hi, "log": log}
                  for k, (kind, lo, hi, log) in SEARCH_SPACE.items()},
        "fixed": {"objective": "lambdarank", "min_split_gain": 0.0, "random_state": args.seed},
        "untuned": UNTUNED}, indent=2))

    cvf = pd.concat([pd.DataFrame(untuned_cv).assign(configuration="untuned"),
                     pd.DataFrame(tuned_cv).assign(configuration="tuned")])
    cvf.to_csv(OUT / "cv_folds.csv", index=False)

    # Refit both on all training trips, score student-test.
    rows, models = [], {}
    n_test_obs = len(test.labels)
    for name, params, cv in (("untuned", UNTUNED, untuned_cv), ("tuned", best_params, tuned_cv)):
        model, fit_s = final.fit(params, args.seed)
        scores, pred_s = predict_timed(model, final)
        m = evaluate(test, scores)
        cvd = pd.DataFrame(cv)
        row = {"configuration": name, "params": json.dumps(params)}
        for k in ("top1_accuracy", "macro_f1", "macro_precision", "macro_recall", "observation_log_loss"):
            row[f"cv_{k}_mean"] = cvd[k].mean()
            row[f"cv_{k}_std"] = cvd[k].std()
        row.update({f"test_{k}": v for k, v in m.items()})
        row.update({"test_fit_seconds": fit_s, "test_predict_seconds": pred_s,
                    "test_predict_ms_per_1k_obs": 1000 * pred_s / n_test_obs * 1000})
        rows.append(row)
        models[name] = (model, scores)
        print(f"[{name}] CV top-1 {row['cv_top1_accuracy_mean']:.4f}  test top-1 {m['top1_accuracy']:.4f}  "
              f"macro F1 {m['macro_f1']:.4f}  obs LL {m['observation_log_loss']:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "untuned_vs_tuned.csv", index=False)

    tuned, tuned_scores = models["tuned"]
    joblib.dump({"model": tuned, "features": final.features, "params": best_params},
                MODEL_DIR / "lambdarank_tuned.joblib")
    tuned.model_.booster_.save_model(str(MODEL_DIR / "lambdarank_tuned.txt"))
    (MODEL_DIR / "lambdarank_tuned.json").write_text(json.dumps({
        "model": "LightGBM LambdaRank with per-observation softmax (pathsnap.models.SoftmaxRanker)",
        "features": final.features, "params": best_params, "seed": args.seed,
        "trained_on": "80 training trips, sample of %d reliable observations" % args.max_train_obs,
        "student_test": {k: rows[1][f"test_{k}"] for k in ("top1_accuracy", "macro_f1", "observation_log_loss")},
    }, indent=2))

    # HMM inputs: out-of-fold scores on train, final-model scores on student-test.
    oof = []
    for fold in folds:
        model, _ = fold.fit(best_params, args.seed)
        oof.append(score_frame(fold.held_out, fold.score(model)))
    pd.concat(oof, ignore_index=True).to_parquet(SCORE_DIR / "train_oof.parquet", index=False)
    score_frame(test, tuned_scores).to_parquet(SCORE_DIR / "student_test.parquet", index=False)

    done = study.trials_dataframe(attrs=("state",)).state.eq("COMPLETE").sum()
    try:
        importance = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=args.seed))
        pd.Series(importance, name="importance").rename_axis("param").to_csv(OUT / "param_importance.csv")
        plot_importance(importance, FIG / "param_importance.png")
    except Exception as e:  # too few completed trials
        print("importance skipped:", e)
    plot_history(trials, base, FIG / "optimisation_history.png")
    plot_comparison(table, FIG / "untuned_vs_tuned.png")

    (OUT / "run_config.json").write_text(json.dumps({
        "seed": args.seed, "folds": args.folds, "max_train_obs_per_fit": args.max_train_obs,
        "fold_assignment": "GroupKFold(shuffle=True) over the 80 training trips, group = trip (same as step 04)",
        "features_per_fold": [f.features for f in folds], "features_final": final.features,
        "trials_total": len(study.trials), "trials_completed": int(done),
        "best_trial": study.best_trial.number,
        "versions": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                     "optuna": optuna.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "runtime_seconds": round(time.time() - started, 1)}, indent=2))
    print(table[["configuration", "cv_top1_accuracy_mean", "cv_macro_f1_mean", "test_top1_accuracy",
                 "test_macro_f1", "test_observation_log_loss", "test_predict_ms_per_1k_obs"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
