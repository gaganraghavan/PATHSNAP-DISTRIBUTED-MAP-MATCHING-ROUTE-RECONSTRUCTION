import numpy as np
import pandas as pd
import pytest

from pathsnap.evaluate import Partition, evaluate, observation_table, route_continuity, top1


def _toy() -> Partition:
    # Arcs: 0: n0->n1, 1: n1->n2, 2: n1->n0 (reverse of 0), 3: n5->n6 (disconnected)
    u, v = np.array([0, 1, 1, 5]), np.array([1, 2, 0, 6])
    cand = pd.DataFrame({
        "trip": "t", "obs": [0, 0, 1, 1, 2, 2, 3],
        "arc": [0, 2, 1, 3, 1, 3, 3],
        "candidate_rank": [0, 0, 0, 1, 0, 1, 0],
        "is_label": [1, 0, 1, 0, 0, 0, 0],
        "label_reliable_50m": [1, 1, 1, 1, 1, 1, 0],
    })
    # obs 2: reference arc 0 is not a candidate (miss); obs 3: unreliable label.
    labels = pd.DataFrame({"trip": "t", "obs": [0, 1, 2, 3], "label_arc": [0, 1, 0, 3],
                           "label_dist": [1.0, 2.0, 3.0, 80.0]})
    return Partition(cand, labels, {"t": (u, v)})


def test_top1_breaks_ties_by_candidate_rank():
    p = _toy()
    best = top1(p.candidates, np.array([0.5, 0.5, 0.9, 0.1, 0.4, 0.4, 0.3]))
    assert best.arc.tolist() == [0, 1, 1, 3]


def test_route_continuity_counts_linked_and_repeated_arcs():
    pred = pd.DataFrame({"trip": "t", "obs": [0, 1, 2, 3], "arc": [0, 1, 1, 3]})
    # 0->1 linked, 1->1 same, 1->3 disconnected
    assert route_continuity(pred, _toy().arc_ends) == pytest.approx(2 / 3)


def test_candidate_miss_is_counted_wrong_and_unreliable_excluded():
    p = _toy()
    scores = np.array([0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9])
    obs = observation_table(p, scores)
    assert obs.obs.tolist() == [0, 1, 2]
    assert obs.correct.tolist() == [True, True, False]
    assert obs.true_prob.tolist() == pytest.approx([0.8, 0.7, 0.0])
    m = evaluate(p, scores)
    assert m["observations"] == 3
    assert m["top1_accuracy"] == pytest.approx(2 / 3)


def test_context_features_use_neighbouring_observations():
    from pathsnap.features import add_context_features

    f = pd.DataFrame({
        "obs": [0, 0, 1, 2],
        "arc": [7, 8, 7, 9],
        "point_to_arc_m": np.array([2.0, 5.0, 4.0, 1.0], dtype=np.float32),
        "heading_delta_cos": np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32),
    })
    add_context_features(f, window=1, radius_m=100.0)
    assert f.dist_minus_min.tolist() == [0.0, 3.0, 0.0, 0.0]
    assert f.cos_minus_max.tolist() == [0.0, -2.0, 0.0, 0.0]
    assert f.n_candidates.tolist() == [2, 2, 1, 1]
    # arc 7 at obs 0: neighbour obs 1 has it (4 m), obs -1 does not exist (100 m)
    assert f.window_mean_dist_m[0] == pytest.approx(52.0)
    assert f.window_mean_cos[0] == pytest.approx(0.5)
    assert f.window_presence.tolist() == [1, 0, 1, 0]
