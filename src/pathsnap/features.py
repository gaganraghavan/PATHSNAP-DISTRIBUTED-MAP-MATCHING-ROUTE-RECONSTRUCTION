"""Feature engineering for the existing PathSnap candidate pipeline.

Each feature row describes one (GPS observation, candidate arc) pair. Reference
route labels are attached only as target/quality columns after model features
have been calculated; they are never included in MODEL_FEATURES.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import DEFAULT_RADIUS_M, Candidates
from .io import Trip
from .network import Network


# Features that compare a candidate with the other candidates of the same
# observation, or with the same arc at neighbouring observations.
CONTEXT_WINDOW = 3
CONTEXT_FEATURES = (
    "dist_minus_min",
    "cos_minus_max",
    "n_candidates",
    "window_mean_dist_m",
    "window_mean_cos",
    "window_presence",
)

MODEL_FEATURES = (
    "point_to_arc_m",
    "arc_length_m",
    "candidate_rank",
    "position_along_arc",
    "heading_delta_sin",
    "heading_delta_cos",
    "gps_step_m",
    "gps_gap_s",
    "gps_speed_mps",
    "start_node_degree",
    "end_node_degree",
    *CONTEXT_FEATURES,
)

TARGET_COLUMNS = ("is_label", "label_distance_m", "label_reliable_50m")
IDENTIFIER_COLUMNS = ("obs", "arc")


def candidate_feature_frame(
    trip: Trip,
    net: Network,
    xy: np.ndarray,
    candidates: Candidates,
    labels: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return one feature/target row per generated candidate arc.

    ``xy`` must be the projected GPS trace used by ``candidates.generate``.
    ``labels`` must be produced by ``labels.observation_labels``. Target and
    label-distance columns are metadata and must not be passed as model input.
    """
    frame = candidate_model_features(trip, net, xy, candidates)
    if len(frame) == 0:
        return pd.DataFrame(columns=[*IDENTIFIER_COLUMNS, *MODEL_FEATURES, *TARGET_COLUMNS])
    obs, arc = frame.obs.to_numpy(), frame.arc.to_numpy()
    label_arc = np.asarray(labels["label_arc"], dtype=np.int64)
    label_dist = np.asarray(labels["label_dist"], dtype=np.float64)
    frame["is_label"] = (arc == label_arc[obs]).astype(np.uint8)
    frame["label_distance_m"] = label_dist[obs].astype(np.float32)
    frame["label_reliable_50m"] = (label_dist[obs] <= 50.0).astype(np.uint8)
    return frame[[*IDENTIFIER_COLUMNS, *MODEL_FEATURES, *TARGET_COLUMNS]]


def candidate_model_features(
    trip: Trip,
    net: Network,
    xy: np.ndarray,
    candidates: Candidates,
) -> pd.DataFrame:
    """Identifier and model-feature columns per candidate, computed without any labels.

    This is the inference path (API): it needs only the GPS track and the road
    network. :func:`candidate_feature_frame` adds the training targets on top.
    """
    n_obs = len(xy)
    if len(candidates.obs) == 0:
        return pd.DataFrame(columns=[*IDENTIFIER_COLUMNS, *MODEL_FEATURES])

    step = np.zeros(n_obs, dtype=np.float64)
    gap = np.zeros(n_obs, dtype=np.float64)
    forward = np.zeros_like(xy, dtype=np.float64)
    if n_obs > 1:
        delta = np.diff(xy, axis=0)
        forward[:-1] = delta
        forward[-1] = delta[-1]
        step[1:] = np.linalg.norm(delta, axis=1)
        dt = np.diff(trip.track.t.to_numpy(dtype=np.float64))
        gap[1:] = dt

    # For near-stationary fixes use the previous non-zero movement direction;
    # zero-displacement observations receive a neutral alignment feature.
    magnitude = np.linalg.norm(forward, axis=1)
    backward = np.zeros_like(xy, dtype=np.float64)
    if n_obs > 1:
        backward[1:] = np.diff(xy, axis=0)
    movement = np.where((magnitude > 1.0)[:, None], forward, backward)
    gps_heading = np.arctan2(movement[:, 0], movement[:, 1])

    obs = candidates.obs.astype(np.int64, copy=False)
    arc = candidates.arc.astype(np.int64, copy=False)
    arc_heading = net.heading[arc]
    angle = np.arctan2(np.sin(arc_heading - gps_heading[obs]),
                       np.cos(arc_heading - gps_heading[obs]))
    indegree = np.bincount(net.v, minlength=net.n_nodes)
    outdegree = np.bincount(net.u, minlength=net.n_nodes)
    start, end = net.u[arc], net.v[arc]

    frame = pd.DataFrame({
        "obs": obs,
        "arc": arc,
        "point_to_arc_m": candidates.dist.astype(np.float32, copy=False),
        "arc_length_m": net.length[arc].astype(np.float32),
        "candidate_rank": candidates.rank.astype(np.int16, copy=False),
        "position_along_arc": candidates.t.astype(np.float32, copy=False),
        "heading_delta_sin": np.sin(angle).astype(np.float32),
        "heading_delta_cos": np.cos(angle).astype(np.float32),
        "gps_step_m": step[obs].astype(np.float32),
        "gps_gap_s": gap[obs].astype(np.float32),
        "gps_speed_mps": (step[obs] / np.maximum(gap[obs], 1.0)).astype(np.float32),
        "start_node_degree": (indegree[start] + outdegree[start]).astype(np.int16),
        "end_node_degree": (indegree[end] + outdegree[end]).astype(np.int16),
    })
    add_context_features(frame)
    return frame[[*IDENTIFIER_COLUMNS, *MODEL_FEATURES]]


def add_context_features(frame: pd.DataFrame, window: int = CONTEXT_WINDOW,
                         radius_m: float = DEFAULT_RADIUS_M) -> None:
    """Add CONTEXT_FEATURES in place to one trip's candidate rows.

    Uses only candidate geometry, never labels:
    - dist_minus_min: distance minus the nearest candidate's distance at this observation
    - cos_minus_max: heading agreement minus the best agreement at this observation
    - n_candidates: number of candidates at this observation
    - window_mean_dist_m: mean distance of the same arc over the ``window``
      observations before and after (``radius_m`` where the arc is not a candidate)
    - window_mean_cos: mean heading agreement of the same arc where it is a candidate there
    - window_presence: how many of those 2 * ``window`` observations have the arc as a candidate
    """
    g = frame.groupby("obs", sort=False)
    frame["dist_minus_min"] = (frame.point_to_arc_m - g.point_to_arc_m.transform("min")).astype(np.float32)
    frame["cos_minus_max"] = (frame.heading_delta_cos - g.heading_delta_cos.transform("max")).astype(np.float32)
    frame["n_candidates"] = g.arc.transform("size").astype(np.int16)

    keys = frame[["obs", "arc"]]
    base = frame[["obs", "arc", "point_to_arc_m", "heading_delta_cos"]]
    dist_sum = np.zeros(len(frame))
    cos_sum = np.zeros(len(frame))
    present = np.zeros(len(frame))
    offsets = [o for o in range(-window, window + 1) if o != 0]
    for off in offsets:
        # The neighbour row at obs + off, keyed back onto obs.
        nb = keys.merge(base.assign(obs=base.obs - off), on=["obs", "arc"], how="left")
        has = nb.point_to_arc_m.notna().to_numpy()
        dist_sum += np.where(has, nb.point_to_arc_m.to_numpy(), radius_m)
        cos_sum += np.where(has, nb.heading_delta_cos.to_numpy(), 0.0)
        present += has
    frame["window_mean_dist_m"] = (dist_sum / len(offsets)).astype(np.float32)
    frame["window_mean_cos"] = (cos_sum / np.maximum(present, 1)).astype(np.float32)
    frame["window_presence"] = present.astype(np.int16)


def observation_candidate_misses(n_obs: int, candidates: Candidates,
                                 labels: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return per-observation candidate misses and the <=50 m reliability mask."""
    hit = np.zeros(n_obs, dtype=bool)
    if len(candidates.obs):
        np.logical_or.at(hit, candidates.obs,
                         candidates.arc == np.asarray(labels["label_arc"])[candidates.obs])
    miss = ~hit
    reliable = np.asarray(labels["label_dist"]) <= 50.0
    return miss, reliable
