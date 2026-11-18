"""Server-side scoring of a prediction against the trip's reference route.

Runs after the pipeline has produced its prediction. Uses the same
definitions as the offline experiments (steps 04 and 06):

- per-observation labels from the monotonic alignment of the GPS points to the
  reference route (``pathsnap.labels``); observations more than 50 m from their
  label are excluded as unreliable;
- observation-level top-1 accuracy and macro precision / recall / F1
  (``pathsnap.evaluate.arc_prediction_metrics``);
- observation log loss of the classifier (the probability it gives the
  reference arc; floored like the offline evaluation);
- candidate miss rate: reliable observations whose reference arc is not among
  their candidates;
- route continuity of the per-observation predictions;
- length-weighted route precision / recall / F1 and route mismatch after route
  reconstruction (``pathsnap.route``).

Only these aggregate numbers leave the server; the reference route and the
per-observation labels do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.evaluate import PROB_EPS, RELIABLE_LABEL_M, arc_prediction_metrics
from pathsnap.inference import Prediction
from pathsnap.labels import observation_labels
from pathsnap.route import route_metrics


def continuity(pred: pd.DataFrame, u: np.ndarray, v: np.ndarray) -> float | None:
    a = pred.sort_values("obs").arc.to_numpy()
    if len(a) < 2:
        return None
    ok = (a[1:] == a[:-1]) | (v[a[:-1]] == u[a[1:]])
    return float(ok.mean())


def _clean(d: dict) -> dict:
    """JSON-safe floats (NaN -> None)."""
    out = {}
    for k, x in d.items():
        if isinstance(x, (float, np.floating)):
            out[k] = None if not np.isfinite(x) else float(x)
        elif isinstance(x, np.integer):
            out[k] = int(x)
        else:
            out[k] = x
    return out


def evaluate_prediction(pred: Prediction, reference: np.ndarray) -> dict:
    net = pred.net
    labels = observation_labels(pred.xy, net, reference)
    lab = pd.DataFrame({"obs": np.arange(pred.n_obs), "label_arc": labels["label_arc"],
                        "label_dist": labels["label_dist"]})
    rel = lab[lab.label_dist <= RELIABLE_LABEL_M].copy()
    rel["trip"] = pred.trip_id

    cands = pred.candidates
    hit = rel.merge(cands[["obs", "arc", "prob"]].rename(columns={"arc": "label_arc"}),
                    on=["obs", "label_arc"], how="left")
    true_prob = hit.prob.fillna(0.0).to_numpy()
    miss_rate = float(hit.prob.isna().mean()) if len(hit) else float("nan")

    methods = {}
    for name, p in (("classifier", pred.classifier), ("hmm", pred.hmm)):
        o = rel.merge(p.rename(columns={"arc": "pred_arc"}), on="obs", how="left")
        m = arc_prediction_metrics(o) if len(o) else {"observations": 0, "top1_accuracy": float("nan"),
                                                      "macro_precision": float("nan"),
                                                      "macro_recall": float("nan"), "macro_f1": float("nan")}
        m["route_continuity"] = continuity(p, net.u, net.v)
        rm = route_metrics(pred.routes[name], reference, net.length)
        rm.pop("reference_length_km")          # a property of the reference; keep it server-side
        m.update(rm)
        m["route_breaks"] = pred.route_breaks[name]
        if name == "classifier":
            m["observation_log_loss"] = (float(-np.log(np.clip(true_prob, PROB_EPS, 1.0)).mean())
                                         if len(true_prob) else None)
        else:
            m["decoder_breaks"] = pred.decoder_breaks
        methods[name] = _clean(m)

    return {"reference_available": True,
            "evaluated_observations": int(len(rel)),
            "unreliable_observations": int(pred.n_obs - len(rel)),
            "reliable_label_threshold_m": RELIABLE_LABEL_M,
            "candidate_miss_rate": None if not np.isfinite(miss_rate) else miss_rate,
            "methods": methods}
