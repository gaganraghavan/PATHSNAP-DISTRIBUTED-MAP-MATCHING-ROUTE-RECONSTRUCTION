# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- `uv sync`: install (Python 3.12, managed by uv)
- `uv run pytest`: tests
- `uv run python scripts/01_inventory.py`: inventory + per-observation labels (`artifacts/labels/`, needed by later steps)
- `uv run python scripts/02_candidate_sweep.py`: candidate miss-rate sweep
- `uv run python scripts/03_feature_engineering.py`: per-candidate features (`artifacts/features/`, needed by 04)
- `uv run python scripts/04_model_comparison.py`: 7-model comparison, 5-fold trip CV + student-test (~8 min); `--feature-set pair` runs the no-context ablation; `scripts/plot_model_comparison.py` redraws its figures
- `uv run python scripts/05_bayesian_optimisation.py`: Optuna TPE search for LambdaRank (60 trials, ~1 h; resumable from `artifacts/tuning/optuna.db`), untuned vs tuned, writes `models/lambdarank_tuned.*` and emission scores to `artifacts/scores/` (needed by 06)
- `uv run python scripts/06_hmm_decoding.py`: decoder grid on train (OOF emissions), classifier-only vs HMM on train + student-test (`results/hmm/`, `models/hmm_decoder.json`, predictions in `artifacts/hmm/`, needed by 07)
- `uv run python scripts/07_route_maps.py`: Folium + static route maps for student-test (`results/maps/route_*.{html,png}`)
- `uv run python -m api --data-dir <trips folder>`: FastAPI backend on :8000 (default folder `<PATHSNAP_DATA>/student_test`)
- `python ui/serve.py --api http://<api-host>:8000`: evaluator UI on :5173 (stdlib only)
- EDA notebook: edit `scripts/make_eda_notebook.py`, run it from the repo root, then `cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb`
- Brief PDFs: `pdftotext -layout "docs/<file>.pdf" -`

## Layout

Library code in `src/pathsnap/` (io, geo, network, labels, candidates, features, models, evaluate, validate, maps, hmm, route, inference); FastAPI backend in `api/` (trips, evaluation, app); static UI in `ui/` (Leaflet vendored in `ui/vendor/`); numbered pipeline scripts in `scripts/`; committed outputs in `results/`; committed model weights and decoder config in `models/`; regenerable intermediates in `artifacts/` (git-ignored); dataset in `dataset/Student folder/{train,student_test}/<trip>/` (git-ignored, override with `PATHSNAP_DATA`).

## Established data facts

- Each trip has its own road network; arc/node ids are trip-local. Trips span 25 UTM zones; project per trip (`build_network` picks the zone from the GPS centroid).
- ~92% of arcs have a reverse twin with identical geometry, so direction must come from heading features.
- Per-observation labels come from `labels.align_to_route` (monotonic DP). Trips 33 and 54 have most points > 50 m from their label (off-network driving); treat those labels as unreliable.
- Default candidates: radius 100 m, k = 32 nearest (ties kept).
- HMM: states = candidate arcs, emissions = tuned ranker softmax, transitions = Newson-Krumm |route distance - GPS step| scaled by beta * dt, unreachable moves get a finite `disconnect_logp`. Decoder params are chosen on train OOF emissions only.

## Project: PathSnap

PathSnap is a PES University ML hackathon (UE24CS352A): a map-matching system that reconstructs a vehicle's route on a road network from noisy GPS points. Planned pipeline, in order:

1. Parse and explore raw data, then generate road-segment candidates for each GPS observation and engineer features.
2. Train a classifier that scores candidate segments per observation (at least 5 models from different algorithm families, compared on evidence).
3. Bayesian hyperparameter optimisation on the chosen model, using group-aware CV over the 80 training trips (group = trip).
4. HMM + Viterbi decoder. Hidden states are road segments, emissions come from the classifier output, and transitions come from road-network connectivity. It must handle missing candidates, GPS gaps and disconnected segments, and be compared against classifier-only predictions.
5. FastAPI backend (Laptop 1) holding the model, decoder, road network and server-side ground truth. A separate UI (Laptop 2) lists the 10 evaluator trips, sends a trip ID to the API, and shows the predicted route and accuracy.
6. Folium maps (raw GPS, road network, reference route, predicted route, with legend and layer control), plus a technical report PDF.

## Dataset facts (from `docs/Dataset Exploration.pdf`)

Data is organised per trip, with headerless whitespace-separated files:
- `.track`: longitude, latitude, timestamp (s). Nominally 1 Hz (~97% of steps are 1 s apart).
- `.nodes`: longitude, latitude. The 0-based row index is the node id.
- `.arcs`: from_node_id, to_node_id. The 0-based row index is the `arc_id`.
- `.route`: one column, the chronological ground-truth sequence of `arc_id`s.

Gotchas:
- Coordinate order is lon, lat. Use a projected CRS or geodesic distance, never Euclidean distance in degrees.
- A GPS row does not map to a node or arc row. Matching has to be built.
- The reference route is reference data, not perfect truth, and GPS points may lie off the network.
- Verify file naming, row counts and dtypes per trip. Do not assume trips are identical.

Partitions are split by trip: 80 training, 10 student-testing (development and model comparison), 10 evaluator-testing (final evaluation only).

## Hard rules (grading and integrity)

- Evaluator-test data and labels must never be used for training, feature selection, model/hyperparameter/threshold choice or decoder tuning. Fit learned preprocessing on training data only. Decoder parameters are tuned on training and student-test data only.
- Ground-truth labels stay server-side and never reach the prediction pipeline or UI. No hardcoded predictions or per-trip special-casing (repos are inspected for this). The API must work for all 10 evaluator trips, and API logs are shown during the demo.
- Everything reported (metrics, plots, tables) must come from real experiment runs and be reproducible from the repo. Never fabricate results or training curves. Don't plot loss curves for models without epoch-wise histories.
- Keep experiment logs, seeds, configs and the Bayesian search trial record.
- Commit all code used for processing, training, evaluation, HMM, API, UI and Folium maps, plus model weights/configs (within size limits) and a README with install/run/deploy steps. No secrets.
- Teams are compared for similarity across code, configs, weights and results, so the work should be independent.

## Metrics required

Observation-level Top-1 accuracy, macro F1, precision/recall, log loss (where applicable), candidate miss rate (reference arc absent from the candidate set), route continuity, sequence-level performance, and training/inference time. State the primary model-selection metric and why. Compare tuned vs untuned and classifier-only vs classifier+HMM on the same partitions.
