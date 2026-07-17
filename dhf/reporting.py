"""Shared model-fitting and plotting helpers for per-scenario baseline notebooks.

Both pipeline/04_baseline_linear_regression.py and
pipeline/05_baseline_lightgbm.py follow the identical structure implemented
once here: fit per (target, horizon), score on validation, then plot
MAE-by-horizon, actual-vs-predicted (summer/winter), rolling validation
error, and feature importance. Keeping this in one place is what makes
"same structure, different model" a guarantee rather than something that
can drift between the two notebooks.
"""

from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dhf.features import build_supervised_frame
from dhf.splits import time_split
from dhf.viz import ACTUAL_COLOR, HORIZON_COLORS, TARGET_LABELS, style_axis

TARGETS = ["pv", "wind_onshore", "wind_offshore", "load"]
HORIZONS_DAYS = [1, 7, 14]

# 7-day rolling mean: smooths daily noise while still tracking seasonal
# swings in error over the full ~2-year validation period.
ROLLING_ERROR_WINDOW_HOURS = 24 * 7


def describe_feature_sets(panel: pd.DataFrame) -> dict[str, list[str]]:
    """Feature column names per target (stable across horizons — see dhf/features.py)."""
    return {
        target: build_supervised_frame(panel, target, 24, is_load=(target == "load"))[0].columns.tolist()
        for target in TARGETS
    }


def fit_and_evaluate(
    panel: pd.DataFrame,
    model_factory: Callable[[], object],
    importance_fn: Callable[[object, list[str]], pd.Series],
) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame], dict[tuple[str, int], pd.Series]]:
    """Fit one model per (target, horizon) cell; return (results, val_predictions, importances).

    `results`: one row per (target, horizon) with mae/rmse/n_train/n_val.
    `val_predictions`: keyed by (target, horizon_days) -> DataFrame(actual,
    predicted), indexed by the *target* timestamp (origin + horizon).
    `importances`: keyed by (target, horizon_days) -> feature importances
    from `importance_fn(fitted_model, feature_names)`, normalized to sum 1.
    """
    val_start, test_start = time_split(panel.index)

    rows = []
    val_predictions: dict[tuple[str, int], pd.DataFrame] = {}
    importances: dict[tuple[str, int], pd.Series] = {}

    for target in TARGETS:
        for horizon_days in HORIZONS_DAYS:
            horizon_hours = horizon_days * 24
            X, y = build_supervised_frame(panel, target, horizon_hours, is_load=(target == "load"))

            train_mask = X.index < val_start
            val_mask = (X.index >= val_start) & (X.index < test_start)
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            cell = pd.DataFrame({"actual": y_val, "predicted": y_pred})
            cell.index = X_val.index + pd.Timedelta(hours=horizon_hours)
            val_predictions[(target, horizon_days)] = cell

            importance = importance_fn(model, list(X_train.columns))
            importances[(target, horizon_days)] = importance / importance.sum()

            rows.append(
                {
                    "target": target,
                    "horizon_days": horizon_days,
                    "mae": mean_absolute_error(y_val, y_pred),
                    "rmse": mean_squared_error(y_val, y_pred) ** 0.5,
                    "n_train": len(X_train),
                    "n_val": len(X_val),
                }
            )

    return pd.DataFrame(rows), val_predictions, importances


def plot_mae_by_horizon(results: pd.DataFrame, color: str) -> plt.Figure:
    """2x2 grid, one target per panel: this model's validation MAE vs. horizon."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, target in zip(axes.flat, TARGETS):
        target_results = results[results["target"] == target].sort_values("horizon_days")
        ax.plot(target_results["horizon_days"], target_results["mae"], color=color, marker="o", linewidth=2, zorder=3)
        ax.set_xticks(HORIZONS_DAYS)
        ax.set_xticklabels([f"{h}d" for h in HORIZONS_DAYS])
        ax.set_title(TARGET_LABELS[target])
        ax.set_ylabel("Validation MAE (MW)")
        ax.set_ylim(bottom=0)
        style_axis(ax)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    val_predictions: dict[tuple[str, int], pd.DataFrame],
    window: tuple[pd.Timestamp, pd.Timestamp],
    color: str,
    model_label: str,
) -> plt.Figure:
    """Target x horizon grid of actual (black) vs. predicted (`color`) over one fixed window."""
    window_start, window_end = window
    fig, axes = plt.subplots(len(TARGETS), len(HORIZONS_DAYS), figsize=(15, 14), sharex=True)

    for row, target in enumerate(TARGETS):
        for col, horizon_days in enumerate(HORIZONS_DAYS):
            ax = axes[row, col]
            cell = val_predictions[(target, horizon_days)].loc[window_start:window_end]
            ax.plot(cell.index, cell["actual"], color=ACTUAL_COLOR, linewidth=1.2, label="Actual")
            ax.plot(cell.index, cell["predicted"], color=color, linewidth=1.0, label=model_label)
            if row == 0:
                ax.set_title(f"{horizon_days}d horizon")
            if col == 0:
                ax.set_ylabel(f"{TARGET_LABELS[target]}\nMW")
            style_axis(ax)
            ax.tick_params(axis="x", rotation=30)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout()
    return fig


def plot_rolling_error(val_predictions: dict[tuple[str, int], pd.DataFrame]) -> plt.Figure:
    """Full-validation-period rolling-mean absolute error, one panel per target, one line per horizon."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, target in zip(axes.flat, TARGETS):
        for horizon_days in HORIZONS_DAYS:
            cell = val_predictions[(target, horizon_days)].sort_index()
            abs_err = (cell["actual"] - cell["predicted"]).abs()
            rolling = abs_err.rolling(ROLLING_ERROR_WINDOW_HOURS, min_periods=ROLLING_ERROR_WINDOW_HOURS // 2).mean()
            ax.plot(rolling.index, rolling, color=HORIZON_COLORS[horizon_days], linewidth=1.2, label=f"{horizon_days}d", zorder=3)
        ax.set_title(TARGET_LABELS[target])
        ax.set_ylabel("7d rolling MAE (MW)")
        style_axis(ax)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout()
    return fig


def plot_importance_heatmaps(importances: dict[tuple[str, int], pd.Series]) -> plt.Figure:
    """2x2 grid of feature x horizon importance-share heatmaps (each column sums to 1)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap = plt.get_cmap("Blues")
    vmax = max(s.max() for s in importances.values())

    im = None
    for ax, target in zip(axes.flat, TARGETS):
        table = pd.DataFrame({h: importances[(target, h)] for h in HORIZONS_DAYS})
        table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
        im = ax.imshow(table.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(HORIZONS_DAYS)))
        ax.set_xticklabels([f"{h}d" for h in HORIZONS_DAYS])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index, fontsize=8)
        ax.set_title(TARGET_LABELS[target])

    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    fig.colorbar(im, cax=cbar_ax, label="Share of total importance")
    return fig
