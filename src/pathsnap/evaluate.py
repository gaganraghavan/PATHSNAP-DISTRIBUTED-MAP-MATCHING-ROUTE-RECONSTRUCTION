"""Loading candidate tables and scoring candidate rankers per observation.

Evaluation set: every GPS observation whose aligned reference arc is within
50 m (``label_dist <= 50``). Its true class is that arc. The prediction is the
highest-scoring candidate (ties broken by candidate distance rank); an
observation with no candidate, or whose reference arc is not among its
candidates, still counts and can only be wrong.

Route continuity uses every observation that has a prediction, labelled or not:
consecutive predictions are continuous when they are the same arc or the first
arc ends at the node where the second starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import REPO_ROOT
from .io import list_trips, load_trip

FEATURE_DIR = REPO_ROOT / "artifacts" / "features"
LABEL_DIR = REPO_ROOT / "artifacts" / "labels"
RELIABLE_LABEL_M = 50.0
PROB_EPS = 1e-6


@dataclass
class Partition:
    """Candidate rows plus per-observation labels and arc endpoints for some trips."""

    candidates: pd.DataFrame          # one row per (trip, obs, arc); sorted by trip, obs, rank
    labels: pd.DataFrame              # one row per (trip, obs): label_arc, label_dist
    arc_ends: dict[str, tuple[np.ndarray, np.ndarray]]  # trip -> (u, v) per arc id

    @property
    def trips(self) -> np.ndarray:
        return np.sort(self.labels.trip.unique())

    def subset(self, trips) -> "Partition":
        keep = set(trips)
        return Partition(
            candidates=self.candidates[self.candidates.trip.isin(keep)].reset_index(drop=True),
            labels=self.labels[self.labels.trip.isin(keep)].reset_index(drop=True),
            arc_ends={t: e for t, e in self.arc_ends.items() if t in keep},
        )


def load_partition(partition: str) -> Partition:
    """Read the step-03 feature tables and step-01 labels for one partition."""
    cands, labels, ends = [], [], {}
    for trip_id in list_trips(partition):
        c = pd.read_parquet(FEATURE_DIR / partition / f"{trip_id}.parquet")
        c.insert(0, "trip", trip_id)
        cands.append(c)
        lab = pd.read_parquet(LABEL_DIR / f"{trip_id}.parquet", columns=["label_arc", "label_dist"])
        lab.insert(0, "obs", np.arange(len(lab)))
        lab.insert(0, "trip", trip_id)
        labels.append(lab)
        arcs = load_trip(trip_id, partition, with_route=False).arcs
        ends[trip_id] = (arcs.u.to_numpy(), arcs.v.to_numpy())
    cand = pd.concat(cands, ignore_index=True)
    cand["trip"] = cand["trip"].astype("category")
    return Partition(cand, pd.concat(labels, ignore_index=True), ends)


def training_rows(part: Partition, max_obs: int, rng: np.random.Generator) -> pd.DataFrame:
    """Candidates of sampled reliable observations whose reference arc is a candidate.

    At most ``max_obs`` observations are sampled; every candidate of a sampled
    observation is kept, so the class balance matches prediction time.
    """
    c = part.candidates
    c = c[c.label_reliable_50m == 1]
    hit = c.loc[c.is_label == 1, ["trip", "obs"]].drop_duplicates()
    if len(hit) > max_obs:
        hit = hit.iloc[np.sort(rng.choice(len(hit), size=max_obs, replace=False))]
    return c.merge(hit, on=["trip", "obs"], how="inner")


def top1(candidates: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Best-scoring candidate per observation: columns trip, obs, arc, score, prob."""
    df = candidates[["trip", "obs", "arc", "candidate_rank"]].copy()
    df["score"] = scores
    # Share of the observation's total score, used as a per-observation probability.
    total = df.groupby(["trip", "obs"], observed=True)["score"].transform("sum")
    df["prob"] = np.where(total > 0, df["score"] / total.where(total > 0, 1.0), 0.0)
    df = df.sort_values(["trip", "obs", "score", "candidate_rank"],
                        ascending=[True, True, False, True], kind="stable")
    return df.drop_duplicates(["trip", "obs"], keep="first").reset_index(drop=True)


def route_continuity(pred: pd.DataFrame, arc_ends: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """Share of consecutive predicted observations that are the same or directly linked arcs."""
    linked = total = 0
    for trip, g in pred.groupby("trip", observed=True, sort=False):
        a = g.sort_values("obs").arc.to_numpy()
        if len(a) < 2:
            continue
        u, v = arc_ends[trip]
        ok = (a[1:] == a[:-1]) | (v[a[:-1]] == u[a[1:]])
        linked += int(ok.sum())
        total += len(ok)
    return linked / total if total else float("nan")


def observation_table(part: Partition, scores: np.ndarray) -> pd.DataFrame:
    """Reliable observations with true arc, predicted arc and the true arc's probability."""
    best = top1(part.candidates, scores)
    lab = part.labels[part.labels.label_dist <= RELIABLE_LABEL_M]
    out = lab.merge(best[["trip", "obs", "arc", "prob"]].rename(columns={"arc": "pred_arc", "prob": "pred_prob"}),
                    on=["trip", "obs"], how="left")
    # Probability the ranker gives the true arc (0 when it is not a candidate).
    c = part.candidates[["trip", "obs", "arc"]].copy()
    c["score"] = scores
    total = c.groupby(["trip", "obs"], observed=True)["score"].transform("sum")
    c["true_prob"] = np.where(total > 0, c["score"] / total.where(total > 0, 1.0), 0.0)
    out = out.merge(c[["trip", "obs", "arc", "true_prob"]].rename(columns={"arc": "label_arc"}),
                    on=["trip", "obs", "label_arc"], how="left")
    out["true_prob"] = out["true_prob"].fillna(0.0)
    out["correct"] = out.pred_arc.to_numpy() == out.label_arc.to_numpy()
    return out


def evaluate(part: Partition, scores: np.ndarray) -> dict[str, float]:
    """Observation-level and candidate-level metrics for one set of candidate scores.

    ``scores`` are the model's positive-class probabilities, aligned with
    ``part.candidates``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    obs = observation_table(part, scores)
    y_true = obs.trip.astype(str) + ":" + obs.label_arc.astype(str)
    y_pred = np.where(obs.pred_arc.isna(), "none",
                      obs.trip.astype(str) + ":" + obs.pred_arc.fillna(-1).astype(np.int64).astype(str))
    # Classes are (trip, reference arc); macro averages over every class seen in truth or prediction.
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    # Candidate-level binary view on reliable observations: is this candidate the reference arc?
    cand_mask = part.candidates.label_reliable_50m.to_numpy() == 1
    y_c = part.candidates.is_label.to_numpy()[cand_mask]
    s_c = np.clip(scores[cand_mask], PROB_EPS, 1 - PROB_EPS)
    pb, rb, fb, _ = precision_recall_fscore_support(y_c, s_c >= 0.5, average="binary", zero_division=0)

    pred = top1(part.candidates, scores)
    return {
        "observations": len(obs),
        "top1_accuracy": float(obs.correct.mean()),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "observation_log_loss": float(-np.log(np.clip(obs.true_prob, PROB_EPS, 1.0)).mean()),
        "route_continuity": route_continuity(pred, part.arc_ends),
        "candidate_log_loss": float(log_loss(y_c, s_c, labels=[0, 1])),
        "candidate_roc_auc": float(roc_auc_score(y_c, s_c)),
        "candidate_average_precision": float(average_precision_score(y_c, s_c)),
        "candidate_precision": float(pb),
        "candidate_recall": float(rb),
        "candidate_f1": float(fb),
    }


def candidate_confusion(part: Partition, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """2x2 confusion matrix of candidate rows (reliable observations) at ``threshold``."""
    mask = part.candidates.label_reliable_50m.to_numpy() == 1
    return confusion_matrix(part.candidates.is_label.to_numpy()[mask],
                            np.asarray(scores)[mask] >= threshold, labels=[0, 1])


def reference_continuity(part: Partition) -> float:
    """Continuity of the aligned reference labels themselves (an upper reference point)."""
    pred = part.labels.rename(columns={"label_arc": "arc"})[["trip", "obs", "arc"]]
    return route_continuity(pred, part.arc_ends)
