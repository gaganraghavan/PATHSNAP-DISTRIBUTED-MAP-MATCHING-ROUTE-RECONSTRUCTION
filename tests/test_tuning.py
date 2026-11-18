import numpy as np
import optuna
import pandas as pd
import pytest

from pathsnap.evaluate import Partition
from pathsnap.features import MODEL_FEATURES
from pathsnap.models import SoftmaxRanker
from pathsnap.tuning import (
    SEARCH_SPACE,
    UNTUNED_PARAMS,
    create_study,
    cv_evaluate,
    make_objective,
    prepare_cv_folds,
    trials_table,
)


def synthetic_partition(n_trips=6, n_obs=40, n_cand=5, seed=0) -> Partition:
    """Candidates where the reference arc is usually, not always, the closest one."""
    rng = np.random.default_rng(seed)
    rows, labels, ends = [], [], {}
    for t in range(n_trips):
        trip = f"t{t}"
        for o in range(n_obs):
            dist = rng.gamma(2.0, 10.0, n_cand)
            true = int(rng.integers(n_cand))
            dist[true] *= 0.4
            order = np.argsort(dist)
            for rank, c in enumerate(order):
                rows.append({"trip": trip, "obs": o, "arc": int(c), "candidate_rank": rank,
                             "is_label": int(c == true), "label_reliable_50m": 1, "dist": dist[c]})
            labels.append({"trip": trip, "obs": o, "label_arc": true, "label_dist": 5.0})
        ends[trip] = (np.arange(n_cand), np.arange(1, n_cand + 1))
    cand = pd.DataFrame(rows)
    for i, f in enumerate(MODEL_FEATURES):
        cand[f] = cand["dist"] + rng.normal(0, 1 + i, len(cand)) if i < 3 else rng.normal(size=len(cand))
    cand["candidate_rank"] = cand.groupby(["trip", "obs"]).cumcount()
    cand["trip"] = cand["trip"].astype("category")
    return Partition(cand.drop(columns="dist"), pd.DataFrame(labels), ends)


SMALL = {**UNTUNED_PARAMS, "n_estimators": 20, "num_leaves": 7, "min_child_samples": 5}


def test_untuned_params_match_step04_ranker():
    assert UNTUNED_PARAMS == {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
                              "min_child_samples": 20, "truncation_level": 5, "reg_lambda": 0.0,
                              "colsample_bytree": 1.0}
    assert set(SEARCH_SPACE) <= set(UNTUNED_PARAMS)


def test_cv_folds_are_trip_disjoint_and_learn_something():
    part = synthetic_partition()
    folds = prepare_cv_folds(part, 3, max_obs=10_000, rng=np.random.default_rng(0), seed=0)
    assert len(folds) == 3
    seen = []
    for f in folds:
        assert not set(f.train_trips) & set(f.heldout_trips)
        seen += f.heldout_trips
    assert sorted(seen) == sorted(map(str, part.trips))
    res = cv_evaluate(SMALL, folds, seed=0)
    assert list(res.fold) == ["fold1", "fold2", "fold3"]
    assert res.top1_accuracy.mean() > 1 / 5 + 0.1  # clearly above chance over 5 candidates


def test_study_records_trials_and_per_fold_values():
    part = synthetic_partition()
    folds = prepare_cv_folds(part, 3, max_obs=10_000, rng=np.random.default_rng(0), seed=0)
    study = create_study(seed=0, n_startup_trials=2)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.enqueue_trial({"n_estimators": 100, "num_leaves": 7})
    study.optimize(make_objective(folds, seed=0), n_trials=3)
    t = trials_table(study)
    assert len(t) == 3 and set(t.state) <= {"COMPLETE", "PRUNED"}
    done = t[t.state == "COMPLETE"]
    assert {"fold1_top1_accuracy", "fold3_top1_accuracy", "cv_observation_log_loss_mean"} <= set(t.columns)
    assert done.best_so_far.is_monotonic_increasing
    folds_mean = done[[f"fold{k}_top1_accuracy" for k in (1, 2, 3)]].mean(axis=1)
    assert np.allclose(folds_mean, done.cv_top1_accuracy_mean)


def test_ranker_save_load_roundtrip(tmp_path):
    part = synthetic_partition(n_trips=2)
    c = part.candidates
    X = c[list(MODEL_FEATURES)].astype(np.float32)
    q = c.groupby(["trip", "obs"], observed=True, sort=False).ngroup().to_numpy()
    model = SoftmaxRanker(**SMALL).fit(X, c.is_label.to_numpy(), query=q)
    model.save(tmp_path / "m.txt")
    loaded = SoftmaxRanker.load(tmp_path / "m.txt", **SMALL)
    p0, p1 = model.predict_proba(X, q)[:, 1], loaded.predict_proba(X, q)[:, 1]
    assert np.allclose(p0, p1)
    assert pd.Series(p1).groupby(q).sum().to_numpy() == pytest.approx(1.0)
