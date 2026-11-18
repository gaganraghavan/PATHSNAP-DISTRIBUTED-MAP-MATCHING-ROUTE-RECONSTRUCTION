"""Bayesian hyperparameter optimisation of the LightGBM LambdaRank candidate ranker.

Run after ``scripts/03_feature_engineering.py``. Step 04 selected LambdaRank on
CV mean top-1; this step tunes it.

Procedure
1. Rebuild the step-04 CV folds: 5-fold GroupKFold over the 80 training trips
   (group = trip), the same seed, the same training-row sample per fold
   (at most ``--max-train-obs`` observations) and the same per-fold feature
   selection. The student-test training sample is drawn next from the same
   random stream, also as in step 04.
2. Score the untuned step-04 configuration on these folds (the CV baseline).
3. Run an Optuna TPE search (multivariate, seeded) over ``pathsnap.tuning.SEARCH_SPACE``.
   Objective: CV mean observation-level top-1 accuracy, the step-04 primary
   metric. A median pruner stops trials whose running fold mean falls below the
   median of earlier trials at the same fold.
4. Refit the untuned and the best configuration on the 80 training trips and
   score both on the 10 student-test trips. Student-test is used only here,
   after the configuration is fixed. The evaluator partition is never read.

Outputs
- results/tuning/trials.csv              every trial: state, params, per-fold metrics, time
- results/tuning/param_importance.csv    fANOVA importance over completed trials
- results/tuning/cv_folds.csv            per-fold untuned vs tuned
- results/tuning/tuned_vs_untuned.csv    CV + student-test metrics, fit/predict cost
- results/tuning/student_test_per_trip.csv
- results/tuning/run_config.json         search space, sampler, seed, best params, cost, versions
- results/tuning/figures/*.png
- models/candidate_ranker/{model.txt,config.json}  tuned final model (committed)
- artifacts/tuning/study.db              Optuna storage (git-ignored; enables --resume)

Usage: uv run python scripts/05_bayesian_tuning.py [--trials 60] [--timeout SECONDS] [--resume]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import lightgbm
import numpy as np
import optuna
import pandas as pd
import sklearn

from pathsnap.config import REPO_ROOT, RESULTS_DIR, SEED
from pathsnap.evaluate import evaluate, load_partition, observation_table
from pathsnap.tuning import (
    OBJECTIVE,
    SEARCH_SPACE,
    UNTUNED_PARAMS,
    create_study,
    cv_evaluate,
    fit_ranker,
    fitted_top1,
    make_objective,
    prepare_cv_folds,
    prepare_split,
    trials_table,
)

MODEL_DIR = REPO_ROOT / "models" / "candidate_ranker"
STUDY_DIR = REPO_ROOT / "artifacts" / "tuning"
N_STARTUP_TRIALS = 10
METRICS = ["top1_accuracy", "macro_f1", "macro_precision", "macro_recall", "observation_log_loss",
           "candidate_log_loss", "candidate_roc_auc", "route_continuity"]


def run_search(folds, args) -> optuna.Study:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    db = STUDY_DIR / "study.db"
    if db.exists() and not args.resume:
        db.unlink()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = create_study(args.seed, N_STARTUP_TRIALS, storage=f"sqlite:///{db}", load_if_exists=args.resume)
    remaining = args.trials - len(study.trials)
    if remaining > 0:
        def log(study: optuna.Study, t: optuna.trial.FrozenTrial) -> None:
            val = (f"{t.value:.4f}" if t.state == optuna.trial.TrialState.COMPLETE
                   else f"after {len(t.intermediate_values)} folds")
            print(f"trial {t.number:3d} {t.state.name:8s} {val}  best={study.best_value:.4f}  "
                  f"{t.duration.total_seconds():.0f}s  {t.params}", flush=True)

        study.optimize(make_objective(folds, args.seed), n_trials=remaining, timeout=args.timeout,
                       callbacks=[log], gc_after_trial=True)
    return study


def final_fit(name: str, params: dict, split, test, seed: int) -> tuple[dict, pd.DataFrame, object]:
    model, fit_s = fit_ranker(params, split, seed)
    t0 = time.perf_counter()
    scores = model.predict_proba(split.X_heldout, split.heldout_query)[:, 1]
    predict_s = time.perf_counter() - t0
    m = evaluate(test, scores)
    n_obs = int(test.labels.shape[0])
    m.update({"configuration": name, "train_top1_accuracy": fitted_top1(model, split),
              "fit_seconds": fit_s, "predict_seconds": predict_s,
              "predict_ms_per_1k_obs": 1e6 * predict_s / max(n_obs, 1)})
    per_trip = observation_table(test, scores).groupby("trip").correct.agg(["mean", "size"]).reset_index()
    per_trip.columns = ["trip", "top1_accuracy", "observations"]
    per_trip.insert(0, "configuration", name)
    return m, per_trip, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60, help="total trials in the study (incl. resumed ones)")
    ap.add_argument("--timeout", type=float, default=None, help="stop starting new trials after this many seconds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-train-obs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true", help="continue artifacts/tuning/study.db instead of restarting")
    ap.add_argument("--output", default=str(RESULTS_DIR / "tuning"))
    args = ap.parse_args()
    OUT = Path(args.output)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train = load_partition("train")
    test = load_partition("student_test")
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    folds = prepare_cv_folds(train, args.folds, args.max_train_obs, rng, args.seed)
    final_split = prepare_split(train, test, args.max_train_obs, rng, args.seed, "student_test")
    prep_s = time.perf_counter() - t0
    print(f"prepared {len(folds)} folds + final split in {prep_s:.0f}s; "
          f"features fold1={folds[0].features}", flush=True)

    # 1. Untuned baseline on the identical folds.
    t0 = time.perf_counter()
    base_cv = cv_evaluate(UNTUNED_PARAMS, folds, args.seed)
    base_s = time.perf_counter() - t0
    print(f"untuned CV top-1 {base_cv.top1_accuracy.mean():.4f} ± {base_cv.top1_accuracy.std():.4f} "
          f"({base_s:.0f}s)", flush=True)

    # 2. Bayesian search.
    t0 = time.perf_counter()
    study = run_search(folds, args)
    search_s = time.perf_counter() - t0
    trials = trials_table(study)
    trials.to_csv(OUT / "trials.csv", index=False)
    best = study.best_trial
    best_params = dict(best.params)
    print(f"best trial {best.number}: CV top-1 {best.value:.4f}  {best_params}", flush=True)

    complete = trials[trials.state == "COMPLETE"]
    if len(complete) >= 2:
        imp = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=args.seed))
        pd.DataFrame({"param": list(imp), "importance": list(imp.values())}).to_csv(
            OUT / "param_importance.csv", index=False)

    # 3. Per-fold untuned vs tuned (the best trial completed every fold).
    tuned_cv = pd.DataFrame([{"fold": f.name, **{k: best.user_attrs[f"{f.name}_{k}"] for k in
                              ("top1_accuracy", "observation_log_loss", "train_top1_accuracy", "fit_seconds")}}
                             for f in folds])
    cv_folds = pd.concat([base_cv.assign(configuration="untuned"), tuned_cv.assign(configuration="tuned")],
                         ignore_index=True)
    cv_folds["validation_trips"] = cv_folds.fold.map({f.name: ",".join(f.heldout_trips) for f in folds})
    cv_folds.to_csv(OUT / "cv_folds.csv", index=False)
    gap = tuned_cv.top1_accuracy.to_numpy() - base_cv.top1_accuracy.to_numpy()

    # 4. Refit on the 80 training trips and score once on student-test.
    test_rows, per_trip, models = [], [], {}
    for name, params in [("untuned", UNTUNED_PARAMS), ("tuned", best_params)]:
        m, pt, model = final_fit(name, params, final_split, test, args.seed)
        test_rows.append(m)
        per_trip.append(pt)
        models[name] = model
        print(f"student-test {name:8s} top1={m['top1_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"obsLL={m['observation_log_loss']:.3f} cont={m['route_continuity']:.3f}", flush=True)
    pd.concat(per_trip).to_csv(OUT / "student_test_per_trip.csv", index=False)

    summary = []
    for name, cv, m in [("untuned", base_cv, test_rows[0]), ("tuned", tuned_cv, test_rows[1])]:
        row = {"configuration": name}
        for k in ("top1_accuracy", "observation_log_loss", "train_top1_accuracy"):
            row[f"cv_{k}_mean"] = float(cv[k].mean())
            row[f"cv_{k}_std"] = float(cv[k].std())
        row["cv_fit_seconds_per_fold"] = float(cv.fit_seconds.mean())
        row.update({f"test_{k}": m[k] for k in METRICS + ["train_top1_accuracy", "fit_seconds",
                                                          "predict_seconds", "predict_ms_per_1k_obs"]})
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary["cv_top1_paired_gain_mean"] = [0.0, float(gap.mean())]
    summary["cv_top1_paired_gain_std"] = [0.0, float(gap.std(ddof=1))]
    summary["cv_top1_folds_improved"] = [0, int((gap > 0).sum())]
    summary.to_csv(OUT / "tuned_vs_untuned.csv", index=False)

    # 5. Persist the tuned model (LightGBM text format) with what is needed to use it.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models["tuned"].save(MODEL_DIR / "model.txt")
    model_config = {
        "model": "LightGBM LambdaRank, softmax over each observation's candidates",
        "params": best_params, "random_state": args.seed, "features": final_split.features,
        "trained_on": "80 training trips", "train_observations": final_split.train_observations,
        "selected_by": f"trial {best.number}: {OBJECTIVE}",
        "cv_top1_accuracy_mean": best.value,
        "load": "pathsnap.models.SoftmaxRanker.load('models/candidate_ranker/model.txt', **params)",
    }
    (MODEL_DIR / "config.json").write_text(json.dumps(model_config, indent=2))

    states = trials.state.value_counts().to_dict()
    config = {
        "seed": args.seed,
        "objective": OBJECTIVE,
        "primary_metric_reason": "Same metric as the step-04 model selection, so tuned and untuned are "
                                 "compared on one scale; it is the per-observation decision the HMM refines.",
        "folds": args.folds,
        "fold_assignment": "GroupKFold(shuffle=True) over the 80 training trips, group = trip (as step 04)",
        "max_train_obs_per_fit": args.max_train_obs,
        "feature_selection": "ExtraTrees impurity importance, median threshold, once per fold before the search",
        "features_per_fold": {f.name: f.features for f in folds},
        "features_final": final_split.features,
        "search_space": SEARCH_SPACE,
        "untuned_params": UNTUNED_PARAMS,
        "sampler": {"name": "TPESampler", "multivariate": True, "n_startup_trials": N_STARTUP_TRIALS,
                    "seed": args.seed},
        "pruner": {"name": "MedianPruner", "n_startup_trials": N_STARTUP_TRIALS, "n_warmup_steps": 1,
                   "reported_value": "running mean of fold top-1 after each fold"},
        "trials_requested": args.trials, "timeout_seconds": args.timeout, "resumed": args.resume,
        "trial_states": states,
        "best_trial": best.number, "best_params": best_params, "best_cv_top1_accuracy": best.value,
        "partitions_used": {"search": "train (CV)", "final evaluation": "student_test",
                            "evaluator_test": "never read"},
        "cost_seconds": {"data_preparation": round(prep_s, 1), "untuned_baseline_cv": round(base_s, 1),
                         "search": round(search_s, 1),
                         "mean_per_completed_trial": round(float(
                             trials.loc[trials.state == "COMPLETE", "duration_seconds"].mean()), 1),
                         "mean_per_pruned_trial": round(float(
                             trials.loc[trials.state == "PRUNED", "duration_seconds"].mean()), 1)
                         if "PRUNED" in states else None,
                         "total_runtime": round(time.time() - started, 1)},
        "versions": {"python": platform.python_version(), "optuna": optuna.__version__,
                     "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2, default=str))

    from plot_tuning import plot_all  # noqa: E402  (sibling script)
    plot_all(OUT)
    print(summary.round(4).T.to_string(header=False))


if __name__ == "__main__":
    main()
