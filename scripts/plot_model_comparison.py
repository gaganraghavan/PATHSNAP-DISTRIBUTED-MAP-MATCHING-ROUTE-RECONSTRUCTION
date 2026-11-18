"""Figures for the step-04 model comparison, drawn only from its saved CSV results.

Run on its own to redraw without retraining:
    uv run python scripts/plot_model_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap.config import RESULTS_DIR  # noqa: E402

# Fixed colour per model (categorical slots in a fixed order, never by rank).
MODEL_ORDER = ["Logistic Regression", "Gaussian Naive Bayes", "k-Nearest Neighbours",
               "Random Forest", "Histogram Gradient Boosting", "Multilayer Perceptron", "LightGBM LambdaRank"]
MODEL_COLOR = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                     "#6250d6"]))
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES_A, SERIES_B = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _order(models) -> list[str]:
    return [m for m in MODEL_ORDER if m in set(models)]


def _hbar(ax, models, values, fmt, err=None, xlabel="", lower_is_better=False, label_x=None):
    y = np.arange(len(models))
    ax.barh(y, values, height=0.6, color=[MODEL_COLOR[m] for m in models],
            xerr=err, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3}, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, models)
    if not ax.yaxis_inverted():  # shared y axes would otherwise flip back
        ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel + (" (lower is better)" if lower_is_better else ""))
    ends = np.asarray(values) + (np.asarray(err) if err is not None else 0)
    if label_x is not None:
        ends = np.maximum(ends, label_x)
    top = np.nanmax(ends)
    for yi, v, end in zip(y, values, ends):
        ax.text(end + top * 0.015, yi, fmt.format(v), va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, top * 1.15)


def plot_cv(cv: pd.DataFrame, out: Path) -> None:
    models = _order(cv.model)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, metric, title in [(axes[0], "macro_f1", "CV macro F1"), (axes[1], "top1_accuracy", "CV top-1 accuracy")]:
        g = cv.groupby("model")[metric]
        mean, std = g.mean().loc[models].to_numpy(), g.std().loc[models].to_numpy()
        fold_max = g.max().loc[models].to_numpy()
        _hbar(ax, models, mean, "{:.3f}", err=std, xlabel=f"{title}: mean ± std over folds", label_x=fold_max)
        for i, m in enumerate(models):  # individual folds
            vals = cv.loc[cv.model == m, metric].to_numpy()
            ax.scatter(vals, np.full(len(vals), i), s=14, color=INK, zorder=3, linewidths=0)
        ax.set_title(title)
    fig.suptitle(f"{cv.fold.nunique()}-fold trip-grouped cross-validation on the 80 training trips "
                 "(dots = individual folds)", x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out / "cv_scores.png", dpi=150)
    plt.close(fig)


def plot_student_test(summary: pd.DataFrame, out: Path) -> None:
    s = summary.set_index("model").loc[_order(summary.model)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    _hbar(axes[0], s.index, s.test_top1_accuracy.to_numpy(), "{:.3f}", xlabel="Top-1 accuracy")
    axes[0].set_title("Student-test top-1 accuracy")
    _hbar(axes[1], s.index, s.test_macro_f1.to_numpy(), "{:.3f}", xlabel="Macro F1 (classes = trip × reference arc)")
    axes[1].set_title("Student-test macro F1")
    fig.tight_layout()
    fig.savefig(out / "student_test_scores.png", dpi=150)
    plt.close(fig)


def plot_log_loss(summary: pd.DataFrame, out: Path) -> None:
    s = summary.set_index("model").loc[_order(summary.model)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    _hbar(axes[0], s.index, s.test_observation_log_loss.to_numpy(), "{:.3f}",
          xlabel="−log P(reference arc), scores normalised per observation", lower_is_better=True)
    axes[0].set_title("Student-test observation-level log loss")
    _hbar(axes[1], s.index, s.test_candidate_log_loss.to_numpy(), "{:.3f}",
          xlabel="Binary log loss over candidate rows", lower_is_better=True)
    axes[1].set_title("Student-test candidate-level log loss")
    fig.tight_layout()
    fig.savefig(out / "log_loss.png", dpi=150)
    plt.close(fig)


def plot_confusion(conf: pd.DataFrame, out: Path) -> None:
    models = _order(conf.model)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6))
    for ax, m in zip(axes.flat, models):
        r = conf.set_index("model").loc[m]
        mat = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
        share = mat / mat.sum(axis=1, keepdims=True)  # row-normalised: recall per true class
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{int(v):,}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["other", "reference"])
        ax.set_yticks([0, 1], ["other", "reference"], rotation=90, va="center")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(m, fontsize=10)
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    fig.suptitle("Student-test candidate-level confusion at P ≥ 0.5 (row-normalised shading)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_histories(results: Path, out: Path) -> None:
    files = sorted((results / "histories").glob("*.csv"))
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(6 * len(files), 3.8), squeeze=False)
    for ax, f in zip(axes[0], files):
        h = pd.read_csv(f)
        ax.plot(h.epoch, h.train_log_loss, color=SERIES_A, lw=2, label="train")
        ax.plot(h.epoch, h.validation_log_loss, color=SERIES_B, lw=2, label="validation")
        best = h.validation_log_loss.idxmin()
        ax.scatter([h.epoch[best]], [h.validation_log_loss[best]], s=40, color=SERIES_B,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"best {h.validation_log_loss[best]:.4f}", (h.epoch[best], h.validation_log_loss[best]),
                    textcoords="offset points", xytext=(6, 8), fontsize=9, color=INK_2)
        name = f.stem.replace("_", " ").title()
        unit = "boosting iteration" if "Boosting" in name else "epoch"
        ax.set_title(f"{name}: final fit on 80 training trips")
        ax.set_xlabel(unit)
        ax.set_ylabel("log loss")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_fit_gap(summary: pd.DataFrame, out: Path) -> None:
    s = summary.set_index("model").loc[_order(summary.model)]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hlines(y, s.cv_top1_accuracy_mean, s.cv_train_top1_accuracy_mean, color=GRID, lw=3)
    ax.scatter(s.cv_train_top1_accuracy_mean, y, s=60, color=SERIES_A, label="fitted observations",
               edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.scatter(s.cv_top1_accuracy_mean, y, s=60, color=SERIES_B, label="held-out fold trips",
               edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(y, s.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Top-1 accuracy, mean over CV folds")
    ax.set_title("Fit vs held-out top-1 accuracy (gap = overfitting)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fit_vs_heldout.png", dpi=150)
    plt.close(fig)


def plot_timing(summary: pd.DataFrame, out: Path) -> None:
    s = summary.set_index("model").loc[_order(summary.model)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, col, title, xl in [
        (axes[0], "test_fit_seconds", "Fit time (final fit)", "seconds, log scale"),
        (axes[1], "test_predict_ms_per_1k_obs", "Inference time on student-test", "ms per 1,000 observations, log scale"),
    ]:
        v = s[col].to_numpy()
        yy = np.arange(len(s))
        ax.barh(yy, v, height=0.6, color=[MODEL_COLOR[m] for m in s.index], edgecolor=SURFACE, linewidth=2)
        ax.set_xscale("log")
        ax.set_yticks(yy, s.index)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        for yi, vi in zip(yy, v):
            label = f"{vi:,.0f}" if vi >= 10 else f"{vi:.1f}" if vi >= 1 else f"{vi:.2f}"
            ax.text(vi * 1.1, yi, label, va="center", fontsize=9, color=INK_2)
        ax.set_xlim(v.min() / 2, v.max() * 4)
        ax.set_title(title)
        ax.set_xlabel(xl)
    fig.tight_layout()
    fig.savefig(out / "timing.png", dpi=150)
    plt.close(fig)


def plot_all(results: Path = RESULTS_DIR / "models") -> None:
    figs = results / "figures"
    figs.mkdir(exist_ok=True)
    cv = pd.read_csv(results / "cv_folds.csv")
    summary = pd.read_csv(results / "model_comparison.csv")
    conf = pd.read_csv(results / "confusion_matrices.csv")
    plot_cv(cv, figs)
    plot_student_test(summary, figs)
    plot_log_loss(summary, figs)
    plot_confusion(conf, figs)
    plot_fit_gap(summary, figs)
    plot_timing(summary, figs)
    plot_histories(results, figs)


if __name__ == "__main__":
    plot_all()
