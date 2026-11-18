import numpy as np

from pathsnap.labels import _running_argmin, align_to_route
from pathsnap.network import Network, point_segment_distance


def test_point_segment_distance():
    d, t = point_segment_distance(np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]]), np.array([[2.0, 0.0]]))
    assert np.allclose(d, 1.0) and np.allclose(t, 0.5)
    d, t = point_segment_distance(np.array([[-3.0, 4.0]]), np.array([[0.0, 0.0]]), np.array([[2.0, 0.0]]))
    assert np.allclose(d, 5.0) and np.allclose(t, 0.0)


def test_running_argmin():
    x = np.array([3.0, 1.0, 2.0, 1.0, 0.5])
    assert _running_argmin(x).tolist() == [0, 1, 1, 1, 4]


def test_alignment_is_monotonic_and_follows_route():
    # straight road along x: nodes every 10 m, arcs 0->1->2->3
    nodes = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float)
    net = Network(epsg=0, node_xy=nodes, u=np.array([0, 1, 2]), v=np.array([1, 2, 3]))
    route = np.array([0, 1, 2])
    xy = np.array([[1, 2], [6, -1], [12, 1], [18, 0.5], [25, -2], [29, 1]], dtype=float)
    pos, dist = align_to_route(xy, net, route)
    assert pos.tolist() == [0, 0, 1, 1, 2, 2]
    assert np.all(np.diff(pos) >= 0) and dist.max() <= 2.0
