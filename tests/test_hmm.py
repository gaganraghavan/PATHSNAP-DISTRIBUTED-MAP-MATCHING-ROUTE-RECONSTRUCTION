import numpy as np
import pandas as pd

from pathsnap.hmm import HMMParams, RoadGraph, build_lattice, classifier_only, viterbi
from pathsnap.network import Network
from pathsnap.route import reconstruct, route_metrics


def straight_road():
    """Nodes 0-1-2-3 on a line 100 m apart; arcs 0..2 eastbound, 3..5 their westbound twins."""
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])
    u = np.array([0, 1, 2, 1, 2, 3])
    v = np.array([1, 2, 3, 0, 1, 2])
    return Network(epsg=32600, node_xy=xy, u=u, v=v)


def eastbound_track():
    gps = np.array([[20.0, 3.0], [60.0, 3.0], [120.0, 3.0], [170.0, 3.0], [240.0, 3.0]])
    t = np.arange(len(gps), dtype=float) * 3.0
    return gps, t


def candidates(net, gps, flip_obs):
    """Eastbound and westbound twin as candidates; the classifier prefers the wrong twin at ``flip_obs``."""
    rows = []
    for i, p in enumerate(gps):
        seg = int(p[0] // 100)
        east, west = seg, seg + 3
        te = (p[0] - 100 * seg) / 100
        pe = 0.3 if i == flip_obs else 0.7
        rows += [(i, east, te, 0, pe), (i, west, 1 - te, 0, 1 - pe)]
    return pd.DataFrame(rows, columns=["obs", "arc", "position_along_arc", "candidate_rank", "prob"])


def test_viterbi_fixes_a_wrong_direction_flip():
    net = straight_road()
    gps, t = eastbound_track()
    cands = candidates(net, gps, flip_obs=2)
    graph = RoadGraph(net, gps)
    lat = build_lattice(graph, gps, t, cands)
    pred, breaks = viterbi(lat, HMMParams(beta_m=5.0))
    assert classifier_only(cands).arc.tolist() == [0, 0, 4, 1, 2]
    assert pred.arc.tolist() == [0, 0, 1, 1, 2]
    assert breaks == 0
    route, rbreaks = reconstruct(graph, pred, t)
    assert route == [0, 1, 2] and rbreaks == 0
    m = route_metrics(route, [0, 1, 2], net.length)
    assert m["route_f1"] == 1.0 and m["route_mismatch"] == 0.0


def test_disconnected_candidates_do_not_kill_the_chain():
    # Two separate road pieces with no link between them.
    net = Network(epsg=32600, node_xy=np.array([[0.0, 0], [100, 0], [5000, 0], [5100, 0]]),
                  u=np.array([0, 2]), v=np.array([1, 3]))
    gps = np.array([[50.0, 2], [5050.0, 2]])
    cands = pd.DataFrame({"obs": [0, 1], "arc": [0, 1], "position_along_arc": [0.5, 0.5],
                          "candidate_rank": [0, 0], "prob": [1.0, 1.0]})
    graph = RoadGraph(net, gps)
    pred, breaks = viterbi(build_lattice(graph, gps, np.array([0.0, 1.0]), cands))
    assert pred.arc.tolist() == [0, 1] and breaks == 1
    route, rbreaks = reconstruct(graph, pred, np.array([0.0, 1.0]))
    assert route == [0, 1] and rbreaks == 1


def test_pair_distances_respect_direction_and_limit():
    net = straight_road()
    graph = RoadGraph(net, net.node_xy)
    d = graph.pair_distances(np.array([0, 3, 0]), np.array([3, 0, 3]), np.array([1000.0, 1000.0, 150.0]))
    assert np.allclose(d[:2], [300.0, 300.0]) and np.isinf(d[2])
