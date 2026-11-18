"""End-to-end prediction for one trip, as served by the API.

GPS track + road network -> candidate arcs (step 02 settings) -> label-free
candidate features (step 03) -> tuned LambdaRank emission probabilities
(step 05) -> HMM/Viterbi decoding with the chosen decoder parameters (step 06)
-> route reconstruction (``pathsnap.route``).

The predictor never sees the reference route: :meth:`Predictor.predict`
refuses a trip that carries one. Scoring against the reference happens
elsewhere, after prediction, on the server only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .candidates import DEFAULT_K, DEFAULT_RADIUS_M, generate
from .config import REPO_ROOT
from .features import candidate_model_features
from .hmm import HMMParams, RoadGraph, build_lattice, classifier_only, viterbi
from .io import Trip
from .models import SoftmaxRanker
from .network import Network, build_network, track_xy
from .route import reconstruct

MODEL_DIR = REPO_ROOT / "models"
MODEL_FILE = MODEL_DIR / "lambdarank_tuned.joblib"
DECODER_FILE = MODEL_DIR / "hmm_decoder.json"


@dataclass
class Prediction:
    """Everything the pipeline produced for one trip (no reference data)."""

    trip_id: str
    net: Network
    xy: np.ndarray                      # projected GPS points (m)
    candidates: pd.DataFrame            # obs, arc, position_along_arc, candidate_rank, prob
    classifier: pd.DataFrame            # obs, arc: top-1 candidate per observation
    hmm: pd.DataFrame                   # obs, arc: Viterbi path
    routes: dict[str, list[int]]        # method -> reconstructed arc sequence
    route_breaks: dict[str, int]        # method -> jumps the reconstruction could not bridge
    decoder_breaks: int                 # unreachable transitions taken by Viterbi
    timings: dict[str, float] = field(default_factory=dict)   # stage -> seconds

    @property
    def n_obs(self) -> int:
        return len(self.xy)

    @property
    def n_matched(self) -> int:
        """Observations that had at least one candidate arc (and so a prediction)."""
        return int(self.hmm.obs.nunique())


class Predictor:
    """Holds the tuned ranker and the decoder configuration; predicts one trip at a time."""

    def __init__(self, model_file: Path = MODEL_FILE, decoder_file: Path = DECODER_FILE,
                 radius_m: float = DEFAULT_RADIUS_M, k: int = DEFAULT_K):
        bundle = joblib.load(model_file)
        self.model: SoftmaxRanker = bundle["model"]
        self.features: list[str] = list(bundle["features"])
        self.model_params: dict = dict(bundle.get("params", {}))
        self.params = HMMParams(**json.loads(Path(decoder_file).read_text()))
        self.radius_m = radius_m
        self.k = k
        self.model_file = Path(model_file)
        self.decoder_file = Path(decoder_file)

    def describe(self) -> dict:
        return {"model": "LightGBM LambdaRank (softmax per observation)",
                "model_file": self.model_file.name, "features": self.features,
                "model_params": self.model_params, "decoder": self.params.as_dict(),
                "candidates": {"radius_m": self.radius_m, "k": self.k}}

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        """Emission probability per candidate row (rows must be grouped by observation)."""
        if len(feats) == 0:
            return np.zeros(0)
        X = feats[self.features].astype(np.float32)
        query = feats.obs.to_numpy()
        return self.model.predict_proba(X, query)[:, 1]

    def predict(self, trip: Trip) -> Prediction:
        if trip.route is not None:
            raise ValueError("the predictor must not receive the reference route; "
                             "load the trip with with_route=False")
        timings: dict[str, float] = {}
        tick = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal tick
            now = time.perf_counter()
            timings[name] = now - tick
            tick = now

        net = build_network(trip)
        xy = track_xy(trip, net.epsg)
        t = trip.track.t.to_numpy(dtype=np.float64)
        lap("network_s")

        cands = generate(xy, net, self.radius_m, self.k)
        feats = candidate_model_features(trip, net, xy, cands)
        lap("features_s")

        prob = self.score(feats)
        scored = feats[["obs", "arc", "position_along_arc", "candidate_rank"]].copy()
        scored["prob"] = prob
        lap("classifier_s")

        graph = RoadGraph(net, xy)
        if len(scored):
            lattice = build_lattice(graph, xy, t, scored, self.params)
            hmm_pred, decoder_breaks = viterbi(lattice, self.params)
            clf_pred = classifier_only(scored)
        else:
            empty = pd.DataFrame({"obs": np.array([], dtype=np.int64), "arc": np.array([], dtype=np.int64)})
            hmm_pred, clf_pred, decoder_breaks = empty, empty.copy(), 0
        lap("hmm_s")

        routes, breaks = {}, {}
        for name, pred in (("classifier", clf_pred), ("hmm", hmm_pred)):
            routes[name], breaks[name] = reconstruct(graph, pred, t, self.params)
        lap("route_s")
        timings["total_s"] = sum(timings.values())

        return Prediction(trip_id=trip.trip_id, net=net, xy=xy, candidates=scored,
                          classifier=clf_pred, hmm=hmm_pred, routes=routes, route_breaks=breaks,
                          decoder_breaks=decoder_breaks, timings=timings)
