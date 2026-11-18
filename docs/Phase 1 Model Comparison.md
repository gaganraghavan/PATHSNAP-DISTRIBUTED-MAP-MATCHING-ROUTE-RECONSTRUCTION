# Phase 1 model comparison

This experiment ranks road-arc candidates for each GPS observation. It builds on the existing EDA and monotonic route-label alignment in the repository. Labels are used only as training targets; the held-out score is reported only for observations with a reliable route label (within 50 m) represented in the generated candidate set.

## Feature selection

An Extra Trees model selected features by importance using the median-importance threshold. The selected inputs were point-to-arc distance, arc length, candidate rank, position along the arc, and sine/cosine of the difference between GPS movement heading and road heading. GPS speed/gap and endpoint degree were not retained.

## Validation setup

A fixed 80/20 split was made by trip (seed 42): 64 trips for training and 16 held out for validation. This avoids putting observations from one trip on both sides of the split. The model fitting sample was capped at 300,000 candidate rows, retaining positive labels and sampling negative candidates. The held-out set contained 38,646 reliable observations.

The primary metric is top-1 accuracy: the fraction of held-out observations where the model's highest-scoring candidate matches the aligned route arc. Top-3 accuracy and mean reciprocal rank (MRR) are included to show whether the correct arc appears near the top.

| Model | Top-1 | Top-3 | MRR | Candidate ROC-AUC |
|---|---:|---:|---:|---:|
| Histogram Gradient Boosting | 0.7822 | 0.9602 | 0.8739 | 0.9846 |
| Random Forest | 0.7805 | 0.9605 | 0.8728 | 0.9839 |
| Logistic Regression | 0.7757 | 0.9486 | 0.8646 | 0.9747 |

Histogram Gradient Boosting is the current best by top-1 accuracy and MRR; Random Forest is marginally higher on top-3. The small top-1 gap means this is a provisional best from one fixed holdout, not evidence of a large or statistically established advantage.

## Reproduce

1. Generate candidate features with `python scripts/03_feature_engineering.py`.
2. Run `python scripts/04_model_comparison.py --max-train-rows 300000`.

The script writes aggregate comparison results to `results/models/model_comparison.csv`, selected inputs and split counts to `results/models/feature_selection.json`, and a local `artifacts` model file. Raw data and per-trip outputs are not committed.
