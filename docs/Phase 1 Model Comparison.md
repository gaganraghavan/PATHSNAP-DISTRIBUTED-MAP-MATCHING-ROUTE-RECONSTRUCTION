# Phase 1: candidate model comparison

Seven untuned models score road-arc candidates for each GPS observation: six classifier families and one learning-to-rank model. They are compared with 5-fold trip-grouped cross-validation on the 80 training trips and then on the 10 student-test trips. Every number below comes from `scripts/04_model_comparison.py` (seed 42, about 8 minutes on 20 cores), and the results are in `results/models/`. The pair-feature ablation is in `results/models/ablation_pair_features/`.

## Task framing

Each (observation, candidate arc) row is an example: is this candidate the reference arc? A model outputs P(reference) per candidate. The predicted arc for an observation is its highest-scoring candidate, with ties going to the nearer candidate. Candidates come from step 02 (100 m radius, 32 nearest). The 17 features come from step 03 (`results/features/feature_schema.csv`):

- **Per pair (11):** point-to-arc distance, arc length, distance rank, position along the arc, sin/cos of the heading difference, GPS step/gap/speed, and endpoint degrees.
- **Context (6):** computed from candidate geometry only, never from labels:
  - compared with the other candidates of the same observation: distance minus the nearest candidate's distance, heading agreement minus the best agreement, and the number of candidates;
  - the same arc over the 3 observations before and after: its mean distance, its mean heading agreement, and how many of those 6 observations have it as a candidate.

### Feature ablation

The same comparison was rerun with the 11 per-pair features only (`--feature-set pair`, results in `results/models/ablation_pair_features/`):

| Model | CV top-1, pair only | CV top-1, pair + context | Student-test top-1, pair only | Student-test top-1, pair + context |
|---|---:|---:|---:|---:|
| LightGBM LambdaRank | 0.7922 | **0.8810** | 0.8171 | **0.8821** |
| Histogram Gradient Boosting | **0.7937** | 0.8740 | **0.8194** | 0.8760 |
| Random Forest | 0.7917 | 0.8710 | 0.8149 | 0.8726 |
| Multilayer Perceptron | 0.7895 | 0.8696 | 0.8147 | 0.8715 |
| Logistic Regression | 0.7884 | 0.8689 | 0.8125 | 0.8674 |
| k-Nearest Neighbours | 0.7853 | 0.8648 | 0.8109 | 0.8677 |
| Gaussian Naive Bayes | 0.7601 | 0.8277 | 0.7599 | 0.8459 |

The context features add 7 to 9 points of CV top-1 for every model. A single point often cannot tell parallel roads or the two directions of a road apart, but its neighbours usually can. Classifier-only route continuity also rises from about 0.89 to 0.92 to 0.95. Without context features all models are within 0.009 of each other, and the ranker is not the best. Its advantage shows only once the features describe how a candidate compares with its competitors.

## Protocol

- **Folds.** `GroupKFold(5, shuffle=True)` over the 80 training trips, grouped by trip, so no trip is on both sides of a split. The fold assignment is in `cv_folds.csv` (`validation_trips`).
- **Training rows.** Training uses observations whose aligned label lies within 50 m and appears among the candidates. Each fit samples 20,000 such observations, about 400k candidate rows. Every candidate of a sampled observation is kept, so the 1-in-20 class balance matches prediction time. The cap keeps k-NN and the MLP affordable, and every model sees the same sample.
- **Feature selection** happens inside every fold, on that fold's training rows only. It uses Extra Trees impurity importance with a median threshold. All five folds and the final fit kept the same 9 features: `point_to_arc_m`, `candidate_rank`, `position_along_arc`, `heading_delta_cos`, `dist_minus_min`, `cos_minus_max`, `n_candidates`, `window_mean_dist_m` and `window_mean_cos`.
- **Preprocessing.** Standardisation sits inside the Logistic Regression, k-NN and MLP pipelines, so it is fitted on training rows only.
- **Scoring.** Each fold scores every candidate of its held-out trips. The final models are refitted on all 80 training trips (same sample size) and scored on student-test.

## Metrics

All metrics are computed by `pathsnap.evaluate`.

- **Evaluation set.** All observations whose aligned label lies within 50 m. If the label arc is not among an observation's candidates, the observation stays in the set and counts as wrong. Student-test has 19,133 such observations and no candidate misses.
- **Top-1 accuracy (primary).** The share of observations whose top-scored candidate is the reference arc. It is the brief's observation-level metric and the quantity the route-matching task is judged on.
- **Macro precision/recall/F1.** A class is a (trip, reference arc) pair, so short segments count as much as long ones.
- **Observation log loss.** −log of the probability given to the reference arc, after normalising scores within the observation. This is the emission quality the HMM will consume. A candidate miss gets probability 1e-6.
- **Candidate log loss, ROC-AUC, P/R/F1 at 0.5.** The binary view over candidate rows. For the ranker these use its per-observation softmax probabilities.
- **Route continuity.** The share of consecutive observations whose predicted arcs are the same arc or directly linked. The aligned reference labels score 0.989 on train and 0.985 on student-test.

## Results

The CV columns are mean ± std over 5 folds. All other columns are student-test. The log-loss column gives observation-level then candidate-level. Inference is in ms per 1,000 observations.

| Model | Family | Key configuration | CV macro F1 | CV top-1 | Top-1 | Route continuity | Macro F1 | Log loss (obs / cand) | Fit | Inference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM LambdaRank | Learning to rank | lambdarank per observation, 400 trees, lr=0.05, 31 leaves, softmax per observation | **0.8365 ± 0.0537** | **0.8810 ± 0.0276** | **0.8821** | **0.9487** | **0.8393** | **0.2938** / **0.0255** | 1.64 s | 20.3 |
| Histogram Gradient Boosting | Boosting | lr=0.08, 31 leaves, up to 400 iters, early stop on 10% rows | 0.8064 ± 0.0587 | 0.8740 ± 0.0294 | 0.8760 | 0.9345 | 0.8036 | 0.3185 / 0.0310 | 1.12 s | 14.4 |
| Random Forest | Bagging ensemble | 200 trees, min_samples_leaf=5, sqrt features | 0.7954 ± 0.0617 | 0.8710 ± 0.0311 | 0.8726 | 0.9308 | 0.7883 | 0.3355 / 0.0317 | 11.76 s | 14.8 |
| Multilayer Perceptron | Neural network | standardised, 64-32 ReLU, Adam lr=1e-3, early stop on held-out trips | 0.7889 ± 0.0607 | 0.8696 ± 0.0316 | 0.8715 | 0.9302 | 0.7759 | 0.3368 / 0.0325 | 18.50 s | 3.4 |
| Logistic Regression | Linear | standardised, L2, C=1.0 | 0.7821 ± 0.0600 | 0.8689 ± 0.0296 | 0.8674 | 0.9220 | 0.7738 | 0.4725 / 0.0433 | 0.30 s | 0.6 |
| k-Nearest Neighbours | Instance-based | standardised, k=50, uniform weights | 0.7805 ± 0.0609 | 0.8648 ± 0.0313 | 0.8677 | 0.9292 | 0.7754 | 0.4412 / 0.0406 | 0.34 s | 877.3 |
| Gaussian Naive Bayes | Probabilistic | per-class Gaussian likelihoods | 0.7148 ± 0.0570 | 0.8277 ± 0.0291 | 0.8459 | 0.9149 | 0.7234 | 0.7646 / 0.2878 | 0.02 s | 0.5 |

### Stability and overfitting

| Model | Top-1 on fitted obs | CV top-1 | Gap | Paired CV gap to LambdaRank | Folds won | Macro P / R (test) | Cand. P / R at 0.5 (test) | Cand. ROC-AUC (test) | CV obs log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM LambdaRank | 0.9563 | 0.8810 | +0.075 | – | 5 / 5 | 0.858 / 0.836 | 0.916 / 0.858 | 0.9975 | 0.354 |
| Histogram Gradient Boosting | 0.9065 | 0.8740 | +0.033 | −0.0070 ± 0.0030 | 0 | 0.828 / 0.797 | 0.878 / 0.861 | 0.9964 | 0.373 |
| Random Forest | 0.9464 | 0.8710 | +0.075 | −0.0099 ± 0.0044 | 0 | 0.813 / 0.781 | 0.895 / 0.844 | 0.9961 | 0.409 |
| Multilayer Perceptron | 0.8835 | 0.8696 | +0.014 | −0.0113 ± 0.0052 | 0 | 0.802 / 0.769 | 0.896 / 0.836 | 0.9960 | 0.399 |
| Logistic Regression | 0.8752 | 0.8689 | +0.006 | −0.0121 ± 0.0044 | 0 | 0.800 / 0.766 | 0.839 / 0.816 | 0.9932 | 0.563 |
| k-Nearest Neighbours | 0.8830 | 0.8648 | +0.018 | −0.0162 ± 0.0066 | 0 | 0.801 / 0.768 | 0.887 / 0.818 | 0.9918 | 0.580 |
| Gaussian Naive Bayes | 0.8314 | 0.8277 | +0.004 | −0.0533 ± 0.0116 | 0 | 0.753 / 0.716 | 0.448 / 0.966 | 0.9887 | 0.912 |

The ±0.03 CV standard deviation comes mostly from fold difficulty, which every model shares. Fold 2 is easy (about 0.92) and fold 5 is hard (about 0.84). Comparing models within each fold is therefore more informative. LambdaRank has the best top-1 in all five folds. Its paired margin over HGB is 0.007 ± 0.003.

## Selection

**LightGBM LambdaRank** goes forward to Bayesian tuning and the HMM. The choice rests on the primary metric, CV mean top-1, and the other evidence agrees:

- It has the best top-1 in every CV fold and on student-test (0.882).
- It has the lowest observation log loss in CV (0.354) and on student-test (0.294). The HMM uses these probabilities as emissions.
- It has the highest macro F1 (0.836 CV, 0.839 test), so the gain is not limited to long, frequently observed segments.
- It has the best classifier-only route continuity on student-test (0.949).
- Inference takes about 20 ms per 1,000 observations, cheap enough for the API.

The main concern is overfitting: 0.956 on fitted observations against 0.881 held out. It still generalises best, and `num_leaves`, `min_child_samples`, `n_estimators` and the learning rate are the obvious dimensions for the Bayesian search.

Why ranking helps: each observation has exactly one correct candidate, and the useful question is which candidate beats the others. LambdaRank optimises that ordering within each observation directly. The classifiers instead score every row independently against the global 1-in-20 base rate.

## Per-model notes

- **LightGBM LambdaRank.** Gradient-boosted trees with the LambdaRank objective. Each observation is a query group, and the reference arc is the one relevant item. Raw scores are comparable only within an observation, so a softmax over the observation's candidates turns them into probabilities. That makes the candidate-level metrics comparable but not identical in meaning to the classifiers'. There is no per-iteration loss history: the ranking objective is not a log loss, so the model gets no loss curve.
- **Histogram Gradient Boosting.** Second on every metric. Overfitting is moderate (a 0.033 gap). Validation log loss reaches its minimum at iteration 201, and early stopping ended at 221 (`histories/histogram_gradient_boosting.csv`). Its early-stopping split is random rows, not trips (sklearn's built-in mechanism), so that internal curve is slightly optimistic. The CV and student-test scores are unaffected.
- **Random Forest.** Third. It has a 0.075 gap, the same as LambdaRank's, but a lower held-out score. With leaves of 5 rows it memorises the training observations.
- **Multilayer Perceptron.** Trained epoch by epoch with 15% of the training trips held out. The weights from the best validation epoch (52 of 60) are kept. Its probabilities are close to HGB's, and it has a small gap (0.014). Of the nonlinear models it has the fastest inference and the slowest fit.
- **Logistic Regression.** The context features make the linear model competitive on top-1 (0.869, 0.012 behind the leader). Its probabilities are clearly worse (observation log loss 0.47 against 0.29), because it cannot express interactions such as "heading agreement matters only when the point is close". The gap of 0.006 rules out overfitting.
- **k-Nearest Neighbours.** It ranks competitively but is the slowest at inference by far (877 ms per 1,000 observations), because every query searches 400k training rows in 9 dimensions. Its probabilities are coarse (multiples of 1/50).
- **Gaussian Naive Bayes.** Last on every metric. The features are strongly dependent (several are differences of others), which breaks the independence assumption and pushes probabilities to the extremes. At 0.5 it reaches 0.97 candidate recall at 0.45 precision. The fit and held-out scores match, so it underfits from model bias.

## Figures

All figures are in `results/models/figures/` and are drawn from the CSVs by `scripts/plot_model_comparison.py`:

- `cv_scores.png`: CV macro F1 and top-1, mean ± std with the individual folds.
- `student_test_scores.png`: student-test top-1 and macro F1.
- `log_loss.png`: observation-level and candidate-level log loss.
- `confusion_matrices.png`: candidate-level confusion at P ≥ 0.5 for each model.
- `loss_curves.png`: train and validation log loss for HGB and the MLP, the two models with log-loss iteration histories.
- `fit_vs_heldout.png`: top-1 on fitted observations against held-out top-1.
- `timing.png`: fit and inference time.

## Reproduce

```bash
uv run python scripts/01_inventory.py
uv run python scripts/03_feature_engineering.py
uv run python scripts/04_model_comparison.py      # also redraws the figures
uv run python scripts/04_model_comparison.py --feature-set pair   # ablation without context features
uv run python scripts/plot_model_comparison.py    # figures only, from saved CSVs
```

Fitted models are written to `artifacts/models/` (git-ignored). The tuned final model will be committed after the Bayesian optimisation step.
