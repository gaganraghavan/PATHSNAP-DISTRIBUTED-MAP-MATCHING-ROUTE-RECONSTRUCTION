"""PathSnap FastAPI backend (Laptop 1).

Holds the tuned classifier, the HMM decoder configuration, the trips' road
networks and the server-side reference routes. The UI (Laptop 2) asks for the
trip list, sends one trip id to ``POST /predict``, and receives the predicted
route geometry plus accuracy computed here against the reference route.

What the client receives: GPS points, the nearby road network, the predicted
routes (classifier-only and classifier + HMM) and aggregate metrics. What it
never receives: the reference route or per-observation labels.

Configuration (environment variables):
- ``PATHSNAP_EVAL_DIR``  folder with one sub-folder per served trip
  (default: ``<PATHSNAP_DATA>/student_test``). Point it at the evaluator folder
  for the final demo.
- ``PATHSNAP_CORS_ORIGINS``  comma-separated allowed UI origins (default ``*``).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from itertools import count
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pathsnap.config import DATA_ROOT
from pathsnap.inference import Prediction, Predictor
from pathsnap.maps import nearby_arcs

from .evaluation import evaluate_prediction
from .trips import TripStore

MAX_GPS_POINTS = 4000          # GPS points sent for display (subsampled evenly beyond this)
NETWORK_BUFFER_M = 120.0       # road network shown within this distance of the track
MAX_NETWORK_ARCS = 25000

# ---------------------------------------------------------------- logging

LOG = logging.getLogger("pathsnap.api")


class _RingHandler(logging.Handler):
    """Keeps recent log lines so the UI can show them next to the console."""

    def __init__(self, size: int = 500):
        super().__init__()
        self.lines: deque[dict] = deque(maxlen=size)
        self.seq = count(1)

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append({"id": next(self.seq), "time": record.created,
                           "level": record.levelname, "message": record.getMessage()})


RING = _RingHandler()


def _setup_logging() -> None:
    if getattr(LOG, "_pathsnap_ready", False):
        return
    LOG.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    LOG.addHandler(console)
    LOG.addHandler(RING)
    LOG.propagate = False
    LOG._pathsnap_ready = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------- state

class State:
    predictor: Predictor
    store: TripStore
    lock = threading.Lock()   # one trip through the pipeline at a time (CPU bound)
    started = time.time()
    requests_served = 0


STATE = State()


def eval_dir() -> Path:
    return Path(os.environ.get("PATHSNAP_EVAL_DIR", DATA_ROOT / "student_test"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    t0 = time.perf_counter()
    STATE.predictor = Predictor()
    STATE.store = TripStore(eval_dir())
    d = STATE.predictor.describe()
    LOG.info("model loaded: %s (%d features) from models/%s", d["model"], len(d["features"]), d["model_file"])
    LOG.info("HMM decoder: %s", ", ".join(f"{k}={v:g}" for k, v in d["decoder"].items()))
    LOG.info("serving %d trips from %s: %s", len(STATE.store.trip_ids), STATE.store.root,
             " ".join(STATE.store.trip_ids))
    LOG.info("ready in %.2fs", time.perf_counter() - t0)
    yield


app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan,
              description="Map matching of noisy GPS tracks: candidate ranker + HMM/Viterbi decoder.")

_origins = [o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.method != "OPTIONS" and request.url.path != "/logs":
        client = request.client.host if request.client else "?"
        LOG.info("%s %s %s -> %d (%.0f ms)", client, request.method, request.url.path,
                 response.status_code, 1000 * (time.perf_counter() - t0))
    return response


# ---------------------------------------------------------------- schemas

class PredictRequest(BaseModel):
    trip_id: str = Field(..., examples=["00000007"])


# ---------------------------------------------------------------- helpers

def _latlon(lat: np.ndarray, lon: np.ndarray) -> list[list[float]]:
    return np.round(np.column_stack([lat, lon]), 6).tolist()


def _route_lines(pred: Prediction, nodes_lat, nodes_lon, route: list[int]) -> list[list[list[float]]]:
    """Route as polylines of node coordinates; a new line starts wherever the route is broken."""
    u, v = pred.net.u, pred.net.v
    lines: list[list[int]] = []
    for a in route:
        if not lines or lines[-1][-1] != u[a]:
            lines.append([int(u[a])])
        if lines[-1][-1] != v[a]:
            lines[-1].append(int(v[a]))
    return [_latlon(nodes_lat[ln], nodes_lon[ln]) for ln in lines]


def _payload(trip, pred: Prediction) -> dict:
    tr = trip.track
    lat, lon, t = tr.lat.to_numpy(), tr.lon.to_numpy(), tr.t.to_numpy()
    step = max(1, int(np.ceil(len(tr) / MAX_GPS_POINTS)))
    idx = np.unique(np.r_[np.arange(0, len(tr), step), len(tr) - 1]) if len(tr) else np.array([], int)
    gps = np.column_stack([np.round(lat[idx], 6), np.round(lon[idx], 6), t[idx]]).tolist()

    nlat, nlon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
    near = nearby_arcs(trip, pred.net, NETWORK_BUFFER_M)[:MAX_NETWORK_ARCS]
    # One line per undirected road: drop the reverse twin of two-way arcs.
    pairs = np.sort(np.column_stack([pred.net.u[near], pred.net.v[near]]), axis=1)
    _, keep = np.unique(pairs, axis=0, return_index=True)
    pairs = pairs[np.sort(keep)]
    network = [[[round(float(nlat[a]), 6), round(float(nlon[a]), 6)],
                [round(float(nlat[b]), 6), round(float(nlon[b]), 6)]] for a, b in pairs]

    routes = {}
    for name in ("hmm", "classifier"):
        r = pred.routes[name]
        routes[name] = {"lines": _route_lines(pred, nlat, nlon, r), "n_arcs": len(r),
                        "length_km": float(pred.net.length[r].sum() / 1000) if r else 0.0,
                        "breaks": pred.route_breaks[name]}
    return {
        "trip_id": pred.trip_id,
        "n_obs": pred.n_obs,
        "n_matched": pred.n_matched,
        "n_candidates": int(len(pred.candidates)),
        "duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
        "bounds": [[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]],
        "gps": gps,
        "gps_every": step,
        "network": network,
        "routes": routes,
    }


# ---------------------------------------------------------------- endpoints

@app.get("/")
def root():
    return {"service": "PathSnap API", "docs": "/docs",
            "endpoints": ["GET /health", "GET /trips", "POST /predict", "GET /model", "GET /logs"]}


@app.get("/health")
def health():
    return {"status": "ok", "trips": len(STATE.store.trip_ids), "data_dir": STATE.store.root.name,
            "uptime_s": round(time.time() - STATE.started, 1), "requests_served": STATE.requests_served}


@app.get("/model")
def model():
    return STATE.predictor.describe()


@app.get("/trips")
def trips():
    out = []
    for tid in STATE.store.trip_ids:
        i = STATE.store.info(tid)
        out.append({"trip_id": i.trip_id, "n_obs": i.n_obs, "duration_s": i.duration_s,
                    "evaluable": i.has_reference})
    return {"count": len(out), "trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in STATE.store:
        LOG.warning("PREDICT unknown trip %r", tid)
        raise HTTPException(status_code=404, detail=f"unknown trip id {tid!r}")
    with STATE.lock:
        LOG.info("PREDICT trip=%s: request received", tid)
        t0 = time.perf_counter()
        trip = STATE.store.load_for_prediction(tid)          # no .route read here
        load_s = time.perf_counter() - t0
        pred = STATE.predictor.predict(trip)
        tm = pred.timings
        LOG.info("PREDICT trip=%s: %d GPS points, %d candidates, %d/%d points matched",
                 tid, pred.n_obs, len(pred.candidates), pred.n_matched, pred.n_obs)
        LOG.info("PREDICT trip=%s: network %.2fs | features %.2fs | classifier %.2fs | HMM %.2fs | "
                 "route %.2fs", tid, tm["network_s"], tm["features_s"], tm["classifier_s"], tm["hmm_s"],
                 tm["route_s"])

        # Scoring: the reference route is read only now, after prediction, and stays on the server.
        t1 = time.perf_counter()
        reference = STATE.store.load_reference(tid)
        if reference is None:
            metrics = {"reference_available": False, "methods": {}}
            LOG.warning("EVAL trip=%s: no reference route on the server, accuracy not computed", tid)
        else:
            metrics = evaluate_prediction(pred, reference)
            h, c = metrics["methods"]["hmm"], metrics["methods"]["classifier"]

            def f(x):
                return "n/a" if x is None else f"{x:.4f}"

            LOG.info("EVAL trip=%s: top-1 %s (HMM) vs %s (classifier) | macro F1 %s | route F1 %s | "
                     "continuity %s | %d evaluated points", tid, f(h["top1_accuracy"]), f(c["top1_accuracy"]),
                     f(h["macro_f1"]), f(h["route_f1"]), f(h["route_continuity"]),
                     metrics["evaluated_observations"])
        eval_s = time.perf_counter() - t1

        body = _payload(trip, pred)
        body["metrics"] = metrics
        body["timings"] = {"load_s": load_s, **tm, "evaluation_s": eval_s,
                           "server_total_s": time.perf_counter() - t0}
        STATE.requests_served += 1
        LOG.info("PREDICT trip=%s: done in %.2fs (pipeline %.2fs)", tid, body["timings"]["server_total_s"],
                 tm["total_s"])
        return body


@app.get("/logs")
def logs(after: int = 0, limit: int = 200):
    lines = [x for x in RING.lines if x["id"] > after][-limit:]
    return {"lines": lines}
