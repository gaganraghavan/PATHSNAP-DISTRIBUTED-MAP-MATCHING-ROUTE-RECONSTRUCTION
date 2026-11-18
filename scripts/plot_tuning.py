"""Figures for the step-05 Bayesian tuning, drawn only from its saved CSV results.

Run on its own to redraw without re-tuning:
    uv run python scripts/plot_tuning.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.config import RESULTS_DIR
from plot_model_comparison import GRID, INK, INK_2, SERIES_A, SERIES_B, SURFACE, plt  # shared style

UNTUNED, TUNED = SERIES_B, SERIES_A


def plot_history(trials: pd.DataFrame, baseline: float, out: Path) -> None:
    done = trials[trials.state == "COMPLETE"]
    n_pruned = int((trials.state == "PRUNED").sum())
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.scatter(done.trial, done.cv_top1_accuracy_mean, s=22, color=INK_2, alpha=0.6, linewidths=0,
               label="completed trial")
    ax.step(done.trial, done.best_so_far, where="post", color=TUNED, lw=2, label="best so far")
    ax.axhline(baseline, color=UNTUNED, lw=1.5, ls="--", label=f"untuned ({baseline:.4f})")
    best = done.loc[done.cv_top1_accuracy_mean.idxmax()]
    ax.annotate(f"best {best.cv_top1_accuracy_mean:.4f} (trial {int(best.trial)})",
                (best.trial, best.cv_top1_accuracy_mean), textcoords="offset points", xytext=(-6, 8),
                ha="right", fontsize=9, color=INK_2)
    lo = min(baseline, done.cv_top1_accuracy_mean.min())
    span = done.cv_top1_accuracy_mean.max() - lo
    ax.set_ylim(lo - 0.05 * span - 0.002, done.cv_top1_accuracy_mean.max() + 0.12 * span + 0.002)
    ax.set_xlabel("trial")
    ax.set_ylabel("CV mean top-1 accuracy")
    ax.set_title(f"Bayesian search: {len(done)} completed trials, {n_pruned} pruned (not shown)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "optimisation_history.png", dpi=150)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, out: Path) -> None:
    imp = imp.sort_values("importance", ascending=False)
    y = np.arange(len(imp))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(y, imp.importance, height=0.6, color=TUNED, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, imp.param)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, imp.importance):
        ax.text(v + imp.importance.max() * 0.015, yi, f"{v:.2f}", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, imp.importance.max() * 1.15)
    ax.set_xlabel("fANOVA share of CV top-1 variance across completed trials")
    ax.set_title("Hyperparameter importance")
    fig.tight_layout()
    fig.savefig(out / "param_importance.png", dpi=150)
    plt.close(fig)


def plot_comparison(cv: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})

    # (a) Paired CV folds: same trips for both configurations.
    ax = axes[0]
    piv = cv.pivot(index="fold", columns="configuration", values="top1_accuracy")
    x = np.arange(len(piv))
    ax.vlines(x, piv.untuned, piv.tuned, color=GRID, lw=3)
    ax.scatter(x, piv.untuned, s=60, color=UNTUNED, edgecolor=SURFACE, linewidth=2, zorder=3, label="untuned")
    ax.scatter(x, piv.tuned, s=60, color=TUNED, edgecolor=SURFACE, linewidth=2, zorder=3, label="tuned")
    ax.set_xticks(x, piv.index)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("held-out top-1 accuracy")
    ax.set_title("CV folds (paired)")

    # (b) Student-test observation metrics.
    ax = axes[1]
    s = summary.set_index("configuration").loc[["untuned", "tuned"]]
    metrics = [("test_top1_accuracy", "top-1"), ("test_macro_f1", "macro F1"),
               ("test_route_continuity", "continuity")]
    x = np.arange(len(metrics))
    w = 0.38
    for i, (cfg, color) in enumerate([("untuned", UNTUNED), ("tuned", TUNED)]):
        vals = [s.loc[cfg, m] for m, _ in metrics]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=color, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=INK_2)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.grid(axis="x", visible=False)
    ax.set_title("Student-test (10 trips)")

    # (c) Observation-level log loss, CV mean and student-test.
    ax = axes[2]
    groups = [("cv_observation_log_loss_mean", "CV mean"), ("test_observation_log_loss", "student-test")]
    x = np.arange(len(groups))
    for i, (cfg, color) in enumerate([("untuned", UNTUNED), ("tuned", TUNED)]):
        vals = [s.loc[cfg, c] for c, _ in groups]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=color, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.01 + 0.003, f"{v:.3f}", ha="center", fontsize=8,
                    color=INK_2)
    ax.set_xticks(x, [label for _, label in groups])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, s[[c for c, _ in groups]].max().max() * 1.2)
    ax.set_title("Observation log loss (lower is better)")

    fig.suptitle("Tuned vs untuned LightGBM LambdaRank on identical partitions", x=0.01, ha="left",
                 fontsize=10, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (UNTUNED, TUNED)]
    fig.legend(handles, ["untuned", "tuned"], loc="upper right", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "tuned_vs_untuned.png", dpi=150)
    plt.close(fig)


def plot_all(results: Path = RESULTS_DIR / "tuning") -> None:
    figs = results / "figures"
    figs.mkdir(exist_ok=True)
    trials = pd.read_csv(results / "trials.csv")
    cv = pd.read_csv(results / "cv_folds.csv")
    summary = pd.read_csv(results / "tuned_vs_untuned.csv")
    baseline = float(summary.set_index("configuration").loc["untuned", "cv_top1_accuracy_mean"])
    plot_history(trials, baseline, figs)
    if (results / "param_importance.csv").exists():
        plot_importance(pd.read_csv(results / "param_importance.csv"), figs)
    plot_comparison(cv, summary, figs)


if __name__ == "__main__":
    plot_all()
