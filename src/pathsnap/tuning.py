"""Bayesian hyperparameter search for the LightGBM LambdaRank candidate ranker (step 05).

The search reuses the step-04 protocol so tuned and untuned numbers are directly
comparable: the same 5 GroupKFold folds over the 80 training trips (group =
trip), the same training-row sampling, the same per-fold feature selection and
the same primary metric (observation-level top-1 accuracy on the held-out fold
trips, averaged over folds).

Feature selection does not depend on the ranker's hyperparameters, so it is run
once per fold before the search and shared by every trial.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from .evaluate import PROB_EPS, Partition, observation_table, training_rows
from .features import MODEL_FEATURES
from .models import SoftmaxRanker, select_features

# Each entry: Optuna distribution plus the reason it is in the space.
# Ranges were set from the step-04 results on training/CV data only.
SEARCH_SPACE: dict[str, dict] = {
    "n_estimators": {
        "type": "int", "low": 100, "high": 1200, "step": 50,
        "why": "Number of boosting rounds; trades off against learning_rate. Untuned: 400.",
    },
    "learning_rate": {
        "type": "float", "low": 0.01, "high": 0.3, "log": True,
        "why": "Shrinkage per tree; log scale because its effect is multiplicative. Untuned: 0.05.",
    },
    "num_leaves": {
        "type": "int", "low": 7, "high": 127, "log": True,
        "why": "Tree complexity. Step 04 showed a 0.956 fit vs 0.881 held-out top-1 gap, "
               "so the range reaches well below the untuned 31 as well as above it.",
    },
    "min_child_samples": {
        "type": "int", "low": 5, "high": 500, "log": True,
        "why": "Minimum candidate rows per leaf (an observation has about 20 candidates); "
               "larger values smooth leaves and limit overfitting. Untuned: 20.",
    },
    "reg_lambda": {
        "type": "float", "low": 1e-3, "high": 30.0, "log": True,
        "why": "L2 penalty on leaf values, a second overfitting control. Untuned: 0 (off).",
    },
    "colsample_bytree": {
        "type": "float", "low": 0.4, "high": 1.0,
        "why": "Share of the selected features each tree sees; decorrelates trees. "
               "Only about 9 features are selected per fold, so the floor keeps at least 4. Untuned: 1.0.",
    },
    "truncation_level": {
        "type": "int", "low": 1, "high": 30,
        "why": "LambdaRank only builds pairs that involve its top-k positions. With one reference arc "
               "per observation, small k focuses on the top-1 decision the metric scores. Untuned: 5.",
    },
}

# Step-04 configuration of the ranker (everything except the seed).
UNTUNED_PARAMS: dict = {k: v for k, v in SoftmaxRanker().get_params().items() if k != "random_state"}

OBJECTIVE = "maximise CV mean observation-level top-1 accuracy (5-fold GroupKFold over the 80 training trips)"


def suggest_params(trial: optuna.Trial) -> dict:
    """Draw one configuration from ``SEARCH_SPACE``."""
    params = {}
    for name, s in SEARCH_SPACE.items():
        if s["type"] == "int":
            params[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1),
                                             log=s.get("log", False))
        else:
            params[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
    return params


@dataclass
class SplitData:
    """Fixed training rows, selected features and held-out data for one fit/score split."""

    name: str
    features: list[str]
    X: pd.DataFrame
    y: np.ndarray
    query: np.ndarray
    heldout: Partition
    X_heldout: pd.DataFrame
    heldout_query: np.ndarray
    train_trips: list[str]
    heldout_trips: list[str]

    @property
    def train_observations(self) -> int:
        return int(len(np.unique(self.query)))


def _query_ids(candidates: pd.DataFrame) -> np.ndarray:
    return candidates.groupby(["trip", "obs"], observed=True, sort=False).ngroup().to_numpy()


def prepare_split(train: Partition, heldout: Partition, max_obs: int, rng: np.random.Generator,
                  seed: int, name: str, features: list[str] | None = None) -> SplitData:
    """Sample training rows and select features exactly as step 04 does."""
    rows = training_rows(train, max_obs, rng)
    X_all = rows[list(MODEL_FEATURES)].astype(np.float32)
    y = rows.is_label.to_numpy()
    feats = features if features is not None else select_features(X_all, y, seed)
    return SplitData(
        name=name, features=feats, X=X_all[feats], y=y, query=_query_ids(rows),
        heldout=heldout, X_heldout=heldout.candidates[feats].astype(np.float32),
        heldout_query=_query_ids(heldout.candidates),
        train_trips=[str(t) for t in train.trips], heldout_trips=[str(t) for t in heldout.trips],
    )


def prepare_cv_folds(train: Partition, n_folds: int, max_obs: int, rng: np.random.Generator,
                     seed: int) -> list[SplitData]:
    """The step-04 folds: GroupKFold(shuffle) over trips, sampled in fold order from ``rng``."""
    trips = train.trips
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [prepare_split(train.subset(trips[tr]), train.subset(trips[va]), max_obs, rng, seed, f"fold{k}")
            for k, (tr, va) in enumerate(folds.split(trips, groups=trips), start=1)]


def fit_ranker(params: dict, split: SplitData, seed: int) -> tuple[SoftmaxRanker, float]:
    t0 = time.perf_counter()
    model = SoftmaxRanker(**params, random_state=seed).fit(split.X, split.y, query=split.query)
    return model, time.perf_counter() - t0


def fitted_top1(model: SoftmaxRanker, split: SplitData) -> float:
    """Top-1 on the fitted observations themselves (ties go to the lower candidate rank)."""
    p = model.predict_proba(split.X, split.query)[:, 1]
    df = pd.DataFrame({"q": split.query, "p": p, "y": split.y})
    return float(df.y.to_numpy()[df.groupby("q", sort=False).p.idxmax().to_numpy()].mean())


def score_split(model: SoftmaxRanker, split: SplitData) -> dict[str, float]:
    """Held-out top-1 and observation log loss, with the step-04 definitions."""
    t0 = time.perf_counter()
    scores = model.predict_proba(split.X_heldout, split.heldout_query)[:, 1]
    predict_s = time.perf_counter() - t0
    obs = observation_table(split.heldout, scores)
    return {"top1_accuracy": float(obs.correct.mean()),
            "observation_log_loss": float(-np.log(np.clip(obs.true_prob, PROB_EPS, 1.0)).mean()),
            "predict_seconds": predict_s}


def cv_evaluate(params: dict, folds: list[SplitData], seed: int,
                on_fold: Callable[[int, dict, float], None] | None = None) -> pd.DataFrame:
    """Fit and score one configuration on every fold.

    ``on_fold(step, fold_result, running_mean_top1)`` is called after each fold;
    the Optuna objective uses it to record per-fold results and to prune.
    """
    rows = []
    for step, split in enumerate(folds):
        model, fit_s = fit_ranker(params, split, seed)
        r = {"fold": split.name, **score_split(model, split), "fit_seconds": fit_s,
             "train_top1_accuracy": fitted_top1(model, split)}
        rows.append(r)
        if on_fold is not None:
            on_fold(step, r, float(np.mean([x["top1_accuracy"] for x in rows])))
    return pd.DataFrame(rows)


def make_objective(folds: list[SplitData], seed: int) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        def on_fold(step: int, r: dict, running: float) -> None:
            for k in ("top1_accuracy", "observation_log_loss", "train_top1_accuracy", "fit_seconds"):
                trial.set_user_attr(f"{r['fold']}_{k}", r[k])
            trial.report(running, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        res = cv_evaluate(params, folds, seed, on_fold)
        for k in ("top1_accuracy", "observation_log_loss", "train_top1_accuracy"):
            trial.set_user_attr(f"cv_{k}_mean", float(res[k].mean()))
            trial.set_user_attr(f"cv_{k}_std", float(res[k].std()))
        trial.set_user_attr("fit_seconds_total", float(res.fit_seconds.sum()))
        return float(res.top1_accuracy.mean())

    return objective


def create_study(seed: int, n_startup_trials: int = 10, storage: str | None = None,
                 study_name: str = "lambdarank_top1", load_if_exists: bool = False) -> optuna.Study:
    """TPE sampler (multivariate, seeded) with a median pruner over the per-fold running mean.

    The pruner only compares a trial with others at the same fold, and never acts
    on the first fold or during the random start-up trials.
    """
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=1)
    return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, storage=storage,
                               study_name=study_name, load_if_exists=load_if_exists)


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """One row per trial: state, objective, parameters and every recorded per-fold value."""
    rows = []
    for t in study.trials:
        # Optuna stores a pruned trial's last running mean as its value; keep that apart
        # so the CV column only holds full 5-fold means.
        running = list(t.intermediate_values.values())
        row = {"trial": t.number, "state": t.state.name,
               "cv_top1_accuracy_mean": t.value if t.state == optuna.trial.TrialState.COMPLETE else None,
               "last_running_top1": running[-1] if running else None,
               "duration_seconds": t.duration.total_seconds() if t.duration else None,
               "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
               "folds_completed": len(t.intermediate_values)}
        row.update({f"param_{k}": v for k, v in t.params.items()})
        row.update(t.user_attrs)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        done = df.state == "COMPLETE"
        df["best_so_far"] = df.cv_top1_accuracy_mean.where(done).cummax()
    return df
