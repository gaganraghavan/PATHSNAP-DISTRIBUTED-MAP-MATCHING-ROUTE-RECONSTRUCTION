import json

import pytest
from fastapi.testclient import TestClient

from synthetic import write_trip

TRIPS = ["00000101", "00000102", "00000103"]


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("eval")
    for k, tid in enumerate(TRIPS):
        write_trip(root, tid, seed=k)
    write_trip(root, "00000199", seed=9, with_route=False)    # served, but no reference on the server
    monkeypatch_module.setenv("PATHSNAP_EVAL_DIR", str(root))
    from api.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_trips_listed(client):
    r = client.get("/trips")
    assert r.status_code == 200
    ids = [t["trip_id"] for t in r.json()["trips"]]
    assert ids == TRIPS + ["00000199"]
    assert all(t["n_obs"] > 0 for t in r.json()["trips"])


def test_predict_returns_route_and_metrics_but_no_reference(client):
    r = client.post("/predict", json={"trip_id": TRIPS[0]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_obs"] == len(body["gps"]) and body["n_matched"] == body["n_obs"]
    assert body["routes"]["hmm"]["lines"] and body["routes"]["classifier"]["lines"]
    m = body["metrics"]
    assert m["reference_available"] and m["evaluated_observations"] > 0
    for method in ("hmm", "classifier"):
        for key in ("top1_accuracy", "macro_f1", "route_f1", "route_continuity", "route_mismatch"):
            v = m["methods"][method][key]
            assert v is not None and v >= 0
    # Clean synthetic grid: the decoded route should be close to the reference.
    assert m["methods"]["hmm"]["route_f1"] > 0.8
    # The reference route and per-point labels never leave the server.
    text = json.dumps(body)
    for leak in ("label_arc", "reference_route", "\"route\":", "labels"):
        assert leak not in text


def test_predict_without_reference(client):
    r = client.post("/predict", json={"trip_id": "00000199"})
    assert r.status_code == 200
    assert r.json()["metrics"] == {"reference_available": False, "methods": {}}


def test_unknown_trip_is_404(client):
    assert client.post("/predict", json={"trip_id": "../../etc"}).status_code == 404
    assert client.post("/predict", json={"trip_id": "nope"}).status_code == 404


def test_logs_record_requests(client):
    client.post("/predict", json={"trip_id": TRIPS[1]})
    lines = [x["message"] for x in client.get("/logs").json()["lines"]]
    assert any(f"PREDICT trip={TRIPS[1]}" in x for x in lines)
    assert any(f"EVAL trip={TRIPS[1]}" in x for x in lines)


def test_predictor_refuses_reference(tmp_path):
    from pathsnap.inference import Predictor
    from pathsnap.io import load_trip

    write_trip(tmp_path, "00000001")
    trip = load_trip("00000001", ".", root=tmp_path)
    assert trip.route is not None
    with pytest.raises(ValueError):
        Predictor().predict(trip)


def test_inference_features_match_training_features(tmp_path):
    """The API's label-free features are exactly the model columns step 03 trains on."""
    import pandas as pd

    from pathsnap.candidates import generate
    from pathsnap.features import MODEL_FEATURES, candidate_feature_frame, candidate_model_features
    from pathsnap.io import load_trip
    from pathsnap.labels import observation_labels
    from pathsnap.network import build_network, track_xy

    write_trip(tmp_path, "00000002", seed=3)
    trip = load_trip("00000002", ".", root=tmp_path)
    net = build_network(trip)
    xy = track_xy(trip, net.epsg)
    cands = generate(xy, net)
    train = candidate_feature_frame(trip, net, xy, cands, observation_labels(xy, net, trip.route))
    infer = candidate_model_features(trip, net, xy, cands)
    cols = ["obs", "arc", *MODEL_FEATURES]
    pd.testing.assert_frame_equal(train[cols], infer[cols])
