"""Candidate-ranking models compared in step 04.

Every model is a binary classifier over (observation, candidate arc) rows that
outputs P(candidate is the reference arc). Scaling lives inside each pipeline,
so it is fitted on the training rows of the current fold only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    family: str
    config: str          # short human-readable key configuration
    estimator: BaseEstimator
    has_history: bool = False
    params: dict = field(default_factory=dict)


class EpochMLP(BaseEstimator, ClassifierMixin):
    """Standardised MLP trained epoch by epoch with a trip-held-out validation set.

    sklearn's own early stopping holds out random rows, which would put
    candidates of the same observation on both sides. Here the caller passes
    ``groups`` (trip ids) and a share of trips is held out instead. The weights
    from the epoch with the lowest validation log loss are kept, and the
    per-epoch train/validation log loss is stored in ``history_``.
    """

    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
                 batch_size=1024, max_epochs=100, patience=8, validation_share=0.15, random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_share = validation_share
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        if groups is None:
            groups = np.zeros(len(y))
        trips = np.unique(groups)
        n_val = max(1, int(round(self.validation_share * len(trips)))) if len(trips) > 1 else 0
        val_trips = rng.choice(trips, size=n_val, replace=False) if n_val else np.array([])
        is_val = np.isin(groups, val_trips)
        if not is_val.any() or is_val.all():
            is_val = rng.random(len(y)) < self.validation_share
        self.scaler_ = StandardScaler().fit(X[~is_val])
        Xt, yt = self.scaler_.transform(X[~is_val]), y[~is_val]
        Xv, yv = self.scaler_.transform(X[is_val]), y[is_val]
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                  learning_rate_init=self.learning_rate_init, batch_size=self.batch_size,
                                  max_iter=1, shuffle=True, random_state=self.random_state)
        best, best_state, wait, rows = np.inf, None, 0, []
        for epoch in range(1, self.max_epochs + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                self.mlp_.partial_fit(Xt, yt, classes=np.array([0, 1]))
            tr = log_loss(yt, self.mlp_.predict_proba(Xt)[:, 1], labels=[0, 1])
            va = log_loss(yv, self.mlp_.predict_proba(Xv)[:, 1], labels=[0, 1])
            rows.append({"epoch": epoch, "train_log_loss": tr, "validation_log_loss": va})
            if va < best - 1e-5:
                best, wait = va, 0
                best_state = ([w.copy() for w in self.mlp_.coefs_], [b.copy() for b in self.mlp_.intercepts_])
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.mlp_.coefs_, self.mlp_.intercepts_ = best_state
        self.history_ = pd.DataFrame(rows)
        self.best_epoch_ = int(self.history_.validation_log_loss.idxmin()) + 1
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(self.scaler_.transform(np.asarray(X, dtype=np.float32)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SoftmaxRanker(BaseEstimator):
    """LightGBM LambdaRank over each observation's candidates, softmax-normalised.

    Rows of one observation form a query group (``query`` ids, contiguous).
    LambdaRank optimises NDCG of the reference arc's position in each group, so
    raw outputs are only comparable within an observation; a softmax over the
    observation's candidates turns them into P(candidate is the reference arc).
    """

    # The defaults are the untuned Phase 1 configuration (LightGBM's own
    # defaults for the regularisation and sampling parameters).
    def __init__(self, n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                 truncation_level=5, colsample_bytree=1.0, subsample=1.0, reg_alpha=0.0,
                 reg_lambda=0.0, min_split_gain=0.0, random_state=0, n_jobs=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.truncation_level = truncation_level
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, query):
        import lightgbm as lgb

        query = np.asarray(query)
        starts = np.flatnonzero(np.r_[True, query[1:] != query[:-1]])
        if len(starts) != len(np.unique(query)):
            raise ValueError("rows of each query group must be contiguous")
        sizes = np.diff(np.r_[starts, len(query)])
        self.model_ = lgb.LGBMRanker(objective="lambdarank", n_estimators=self.n_estimators,
                                     learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                                     min_child_samples=self.min_child_samples,
                                     lambdarank_truncation_level=self.truncation_level,
                                     colsample_bytree=self.colsample_bytree,
                                     # Row bagging needs a bagging frequency; 0 disables it.
                                     subsample=self.subsample, subsample_freq=1 if self.subsample < 1 else 0,
                                     reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
                                     min_split_gain=self.min_split_gain,
                                     random_state=self.random_state, verbose=-1,
                                     **({} if self.n_jobs is None else {"n_jobs": self.n_jobs}))
        self.model_.fit(X, np.asarray(y), group=sizes)
        return self

    def predict_proba(self, X, query):
        raw = self.model_.predict(X)
        q = pd.Series(np.asarray(query))
        e = np.exp(raw - pd.Series(raw).groupby(q).transform("max").to_numpy())
        p = e / pd.Series(e).groupby(q).transform("sum").to_numpy()
        return np.column_stack([1 - p, p])


def model_specs(seed: int) -> list[ModelSpec]:
    """The untuned baselines: six classifier families plus a learning-to-rank model."""
    return [
        ModelSpec("Logistic Regression", "Linear", "standardised, L2, C=1.0",
                  make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
        ModelSpec("Gaussian Naive Bayes", "Probabilistic", "per-class Gaussian likelihoods, var_smoothing=1e-9",
                  GaussianNB()),
        ModelSpec("k-Nearest Neighbours", "Instance-based", "standardised, k=50, uniform weights",
                  make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50, n_jobs=-1))),
        ModelSpec("Random Forest", "Bagging ensemble", "200 trees, min_samples_leaf=5, sqrt features",
                  RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed)),
        ModelSpec("Histogram Gradient Boosting", "Boosting",
                  "lr=0.08, 31 leaves, up to 400 iters, early stop on 10% rows",
                  HistGradientBoostingClassifier(learning_rate=0.08, max_leaf_nodes=31, max_iter=400,
                                                 l2_regularization=1.0, early_stopping=True,
                                                 scoring="loss", validation_fraction=0.1,
                                                 n_iter_no_change=20, random_state=seed),
                  has_history=True),
        ModelSpec("Multilayer Perceptron", "Neural network",
                  "standardised, 64-32 ReLU, Adam lr=1e-3, early stop on held-out trips",
                  EpochMLP(hidden_layer_sizes=(64, 32), random_state=seed), has_history=True),
        ModelSpec("LightGBM LambdaRank", "Learning to rank",
                  "lambdarank per observation, 400 trees, lr=0.05, 31 leaves, softmax per observation",
                  SoftmaxRanker(random_state=seed)),
    ]


def fit(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, trips: np.ndarray, query: np.ndarray) -> BaseEstimator:
    """Fit a fresh copy. ``trips`` groups rows by trip, ``query`` by observation."""
    model = clone(spec.estimator)
    if isinstance(model, EpochMLP):
        model.fit(X, y, groups=trips)
    elif isinstance(model, SoftmaxRanker):
        model.fit(X, y, query=query)
    else:
        model.fit(X, y)
    return model


def history(model: BaseEstimator) -> pd.DataFrame | None:
    """Per-iteration train/validation log loss, for models that record one."""
    if isinstance(model, EpochMLP):
        return model.history_
    if isinstance(model, HistGradientBoostingClassifier) and len(getattr(model, "validation_score_", [])):
        # With scoring="loss" sklearn stores negative log loss; index 0 is before the first tree.
        return pd.DataFrame({"epoch": np.arange(len(model.train_score_)),
                             "train_log_loss": -model.train_score_,
                             "validation_log_loss": -model.validation_score_})
    return None


def select_features(X: pd.DataFrame, y: np.ndarray, seed: int) -> list[str]:
    """Extra Trees impurity importance, keeping features at or above the median."""
    et = ExtraTreesClassifier(n_estimators=150, max_depth=16, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    et.fit(X, y)
    keep = SelectFromModel(et, threshold="median", prefit=True).get_support()
    return [c for c, k in zip(X.columns, keep) if k] or list(X.columns)


def positive_proba(model: BaseEstimator, X: pd.DataFrame, query: np.ndarray) -> np.ndarray:
    """P(candidate is the reference arc); ``query`` (observation ids) is used by rankers only."""
    if isinstance(model, SoftmaxRanker):
        return model.predict_proba(X, query)[:, 1]
    return model.predict_proba(X)[:, 1]

