# Phase 2: Bayesian tuning, HMM/Viterbi decoding and route maps

Every number below comes from `scripts/05_bayesian_optimisation.py`, `scripts/06_hmm_decoding.py` and `scripts/07_route_maps.py` (seed 42). The results are in `results/tuning/`, `results/hmm/` and `results/maps/`.

## 1. Bayesian hyperparameter optimisation (LightGBM LambdaRank)

### Setup

- **Model.** The Phase 1 winner: LightGBM LambdaRank with a per-observation softmax (`pathsnap.models.SoftmaxRanker`).
- **Folds and data.** The same 5 `GroupKFold` trip folds over the 80 training trips as Phase 1, with the same 20,000-observation training sample per fit and the same per-fold feature selection (the same 9 features in every fold). Student-test is not used during the search.
- **Objective.** Maximise the mean observation-level top-1 accuracy over the 5 held-out folds. This is the Phase 1 primary metric, because the task is judged per observation and top-1 is what the decoder starts from.
- **Optimiser.** Optuna `TPESampler(multivariate=True, n_startup_trials=15, seed=42)`.
- **Pruning.** `MedianPruner(n_startup_trials=10, n_warmup_steps=2)` on the running fold mean.
- **Storage.** The study is stored in SQLite, so it can be resumed.
- **Trials.** 30 finished: 24 completed and 6 pruned. One more trial was started and stopped when the search was capped at 30 for time. It is recorded as `FAIL` in `trials.csv` and was not used.

**Search space** (`results/tuning/search_space.json`):

| Hyperparameter | Range | Scale | Untuned | Best (trial 9) |
|---|---|---|---:|---:|
| n_estimators | 100 – 1500 | log | 400 | 540 |
| learning_rate | 0.01 – 0.2 | log | 0.05 | 0.0269 |
| num_leaves | 7 – 255 | log | 31 | 8 |
| min_child_samples | 5 – 500 | log | 20 | 19 |
| lambdarank_truncation_level | 1 – 32 | linear | 5 | 11 |
| colsample_bytree | 0.4 – 1.0 | linear | 1.0 | 0.838 |
| subsample (bagging, freq 1) | 0.4 – 1.0 | linear | 1.0 | 0.783 |
| reg_alpha | 1e-4 – 10 | log | 0 | 2.73 |
| reg_lambda | 1e-4 – 30 | log | 0 | 0.0386 |

**fANOVA importance** over the completed trials (`param_importance.csv`): num_leaves 0.34, subsample 0.23, learning_rate 0.20, n_estimators 0.10, colsample_bytree 0.09; the rest are each below 0.02. The search moved towards much smaller trees (8 leaves instead of 31) with row and column subsampling and L1 regularisation. That addresses the Phase 1 concern: top-1 on fitted observations was 0.956 against 0.881 held out.

### Untuned vs tuned (same folds, same partitions; `untuned_vs_tuned.csv`)

| Metric | Untuned | Tuned |
|---|---:|---:|
| CV top-1 (mean ± std over 5 folds) | 0.8815 ± 0.0280 | **0.8851 ± 0.0268** |
| CV macro F1 | 0.8376 ± 0.0535 | **0.8435 ± 0.0528** |
| CV observation log loss | 0.3516 | **0.3464** |
| Student-test top-1 | 0.8809 | **0.8861** |
| Student-test macro F1 | 0.8421 | **0.8465** |
| Student-test observation log loss | **0.2955** | 0.2993 |
| Student-test candidate log loss | **0.0256** | 0.0259 |
| Student-test route continuity (classifier only) | 0.9499 | **0.9517** |
| Fit time (all 80 trips' sample) | 18.7 s | 21.4 s |
| Inference, ms per 1,000 observations | **260** | 594 |

- **Accuracy.** The tuned model is better on the objective and on both student-test classification metrics. The gain is small (+0.5 points of top-1), which is consistent with the history plot: the best was reached at trial 9, and later trials stayed within 0.001 of it.
- **Log loss.** Student-test observation log loss got slightly worse (+0.004), although CV log loss improved. The objective was top-1, not calibration.
- **Inference time.** The tuned model is slower at inference: 540 trees instead of 400, with deeper boosting rounds evaluated per row. Both times were measured in the same run on the same machine. They are not comparable with the Phase 1 timings, which came from a different (20-core) machine.
- **Untuned baseline.** It scores 0.8815 CV top-1 here against 0.8810 in Phase 1. The folds, samples and features are identical; the 0.0005 difference comes from LightGBM's multithreaded histogram construction on a different core count.

Figures (`results/tuning/figures/`):
- `optimisation_history.png`: CV top-1 per trial and the best so far, against the untuned line.
- `param_importance.png`
- `untuned_vs_tuned.png`

Model files:
- `models/lambdarank_tuned.joblib`: the fitted `SoftmaxRanker` and its feature list.
- `models/lambdarank_tuned.txt`: the LightGBM booster as text.
- `models/lambdarank_tuned.json`: parameters, features and student-test scores.

## 2. HMM and Viterbi route reconstruction

Implementation: `src/pathsnap/hmm.py` (decoder) and `src/pathsnap/route.py` (route reconstruction and route metrics).

1. **Hidden states.** The states at each GPS observation are its candidate arcs (100 m radius, 32 nearest). Each state carries the arc id and the position `t` (0..1) of the point's projection along the arc. Arcs are directed, so the two directions of a road are different states.
2. **Emissions.** `emission_weight * log P(arc | observation)`, where P is the tuned ranker's per-observation softmax, floored at 1e-9.
3. **Transitions.** Newson and Krumm (2009): `log p = -|route_m - gps_m| / (beta_m * max(1, dt))`.
   - `gps_m` is the straight-line distance between the two GPS points.
   - `route_m` is the directed network distance from the first projected point to the second: the rest of arc a, then the Dijkstra shortest path from a's end node to b's start node, then the start of arc b.
   - On the same arc, a forward move, or a backward move of up to 15 m (GPS jitter), costs only the distance along the arc.
   - A U-turn onto the reverse twin is therefore charged its real extra distance. This is what lets the decoder fix the direction flips the classifier makes on two-way roads, which account for 92% of arcs.
   - Dijkstra runs on the trip's arcs inside the track's bounding box plus 2 km.
4. **Missing candidates and large GPS gaps.**
   - Observations with no candidate get no state. The chain skips them, and the next transition spans the real time gap.
   - The transition scale grows linearly with `dt`, so after a long gap the decoder relies on the emissions instead of insisting on a matching path length.
   - The reachability bound `max_speed_mps * dt + slack_m` (60 m/s, 150 m) also grows with the gap.
5. **Disconnected segments.**
   - A transition is unreachable when no directed path exists within the bound. Unreachable transitions get a finite log-probability, `disconnect_logp`, instead of -inf, so Viterbi never runs out of paths. The decoder can cross network breaks, off-network driving or missing arcs, and the best emissions decide.
   - Each such step is counted as a decoder break.
   - In route reconstruction, a jump with no connecting path is kept as a route break, not invented.
6. **Decoder tuning.**
   - A grid over `beta_m` ∈ {2, 5, 10, 25, 60}, `emission_weight` ∈ {0.5, 1, 2} and `disconnect_logp` ∈ {-10, -30} was decoded on the 80 training trips (`grid.csv`, `figures/decoder_grid.png`).
   - Training-trip emissions are the tuned ranker's out-of-fold probabilities: each trip was scored by the fold model that did not train on it, so they are as noisy as on unseen trips.
   - Best train top-1: `beta_m = 5`, `emission_weight = 1`, `disconnect_logp = -30` (`models/hmm_decoder.json`). Student-test was not used for this choice.

### Did the HMM improve the route? (`results/hmm/summary.csv`)

Same tuned emissions, same observations. The classifier-only route is reconstructed with the same procedure as the HMM route.

| Metric | Train (OOF), classifier only | Train (OOF), + HMM | Student-test, classifier only | Student-test, + HMM |
|---|---:|---:|---:|---:|
| Observation top-1 | 0.8891 | **0.9250** | 0.8861 | **0.9097** |
| Macro precision | 0.8671 | **0.9376** | 0.8638 | **0.9391** |
| Macro recall | 0.8483 | **0.9317** | 0.8435 | **0.9343** |
| Macro F1 | 0.8525 | **0.9316** | 0.8465 | **0.9330** |
| Route continuity | 0.9462 | **0.9836** | 0.9517 | **0.9833** |
| Route precision (length-weighted, mean over trips) | 0.8523 | **0.9643** | 0.8739 | **0.9735** |
| Route recall | 0.9859 | **0.9902** | **0.9942** | 0.9933 |
| Route F1 | 0.9099 | **0.9764** | 0.9278 | **0.9832** |
| Route mismatch (median over trips; lower is better) | 0.1462 | **0.0229** | 0.1396 | **0.0268** |
| Trips with route F1 ≥ 0.9 | 54 / 80 | **76 / 80** | 7 / 10 | **10 / 10** |
| Unbridgeable route breaks | 4,779 | **402** | 276 | **2** |
| Decode time, ms per 1,000 observations | – | 435 | – | 297 |

- **Overall effect.** Decoding adds 2.4 points of top-1 on student-test (3.6 on train) and 8.7 points of macro F1. Macro F1 rises more than top-1 because the classifier's errors concentrate on short arcs and direction twins, and the transition model fixes exactly those.
- **Route quality.** The reconstructed route improves most. Classifier-only predictions flicker between parallel roads and the two directions of a road. Joining them produces detours and U-turns: route precision is 0.87, and 276 jumps could not be bridged at all. After decoding, route precision is 0.97, the median route mismatch drops from 14% to 2.7% of the reference length, and every student-test trip has route F1 ≥ 0.9.
- **Route recall.** It barely changes (0.994 to 0.993): the classifier-only route already covered the reference. The HMM's gain is in removing wrong arcs, not in finding missed ones.

### Where the HMM did not help

- **Student-test trip 00000007.** Top-1 fell from 0.922 to 0.885, while route F1 is unchanged (0.987).
  - Of the 75 observations the HMM got wrong and the classifier got right, 74 are matched to the reverse twin of the reference arc.
  - Their reference labels are 24 m away on median, against a typical 4–5 m, so this is a noisy stretch.
  - The route uses both directions of that road, and the decoder places the turnaround at a different point. The set of arcs in the route is unchanged, but the per-point direction is wrong for about 30 seconds of driving.
- **Training trips.** 4 of 80 got slightly worse: 00000005 by −0.8 points, 00000097 by −0.7, 00000045 by −0.3 and 00000016 by −0.1. On trip 00000005 all 8 lost observations sit on the arc next to the reference arc, so they are boundary-timing differences at junctions, not wrong roads. The other 75 training trips improved.
- **Remaining breaks.** The decoder still crosses 534 unreachable transitions on train and 8 on student-test. They are concentrated in trips with off-network driving (33, 54) and long GPS gaps. There, no directed path within the speed bound exists, and the decoder follows the emissions.

Figures (`results/hmm/figures/`):
- `classifier_vs_hmm.png`
- `per_trip_top1.png`: per-trip scatter of classifier-only vs HMM top-1.
- `decoder_grid.png`

## 3. Route maps (Folium)

`scripts/07_route_maps.py` writes one interactive map per student-test trip to `results/maps/route_<trip>.html`, with an index at `results/maps/index.html`. Each map has these layers, with layer control, a legend, start and end markers, and a fit-to-track extent:
- raw GPS (red points with timestamps);
- the road network within 150 m of the track (grey);
- the reference route (wide translucent green);
- the classifier + HMM route (solid blue);
- the classifier-only route (dashed orange).

An info box gives that trip's top-1, route F1, mismatch and breaks for both methods. A static view of each map for the report is in `results/maps/route_<trip>.png`.

## Reproduce

```bash
uv run python scripts/01_inventory.py
uv run python scripts/03_feature_engineering.py
uv run python scripts/05_bayesian_optimisation.py --trials 30   # ~1.5 h from scratch on a laptop
uv run python scripts/06_hmm_decoding.py                         # ~6 min
uv run python scripts/07_route_maps.py                           # ~1 min
```
