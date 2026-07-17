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


def describe_feature_sets(
    panel: pd.DataFrame, extra_features_by_target: dict[str, pd.DataFrame] | None = None
) -> dict[str, list[str]]:
    """Feature column names per target (stable across horizons — see dhf/features.py)."""
    extra_features_by_target = extra_features_by_target or {}
    return {
        target: build_supervised_frame(
            panel, target, 24, is_load=(target == "load"), extra_target_time_features=extra_features_by_target.get(target)
        )[0].columns.tolist()
        for target in TARGETS
    }


def fit_and_evaluate(
    panel: pd.DataFrame,
    model_factory: Callable[[], object],
    importance_fn: Callable[[object, list[str]], pd.Series],
    extra_features_by_target: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame], dict[tuple[str, int], pd.Series]]:
    """Fit one model per (target, horizon) cell; return (results, val_predictions, importances).

    `results`: one row per (target, horizon) with mae/rmse/n_train/n_val.
    `val_predictions`: keyed by (target, horizon_days) -> DataFrame(actual,
    predicted), indexed by the *target* timestamp (origin + horizon).
    `importances`: keyed by (target, horizon_days) -> feature importances
    from `importance_fn(fitted_model, feature_names)`, normalized to sum 1.
    `extra_features_by_target`, if given, adds a scenario's extra
    target-time features (e.g. perfect-foresight PECD capacity factor) for
    whichever targets have an entry — targets without one get the plain
    baseline feature set unchanged (e.g. `load`, which has no weather
    feature yet — see docs/modeling_plan.md).
    """
    val_start, test_start = time_split(panel.index)
    extra_features_by_target = extra_features_by_target or {}

    rows = []
    val_predictions: dict[tuple[str, int], pd.DataFrame] = {}
    importances: dict[tuple[str, int], pd.Series] = {}

    for target in TARGETS:
        for horizon_days in HORIZONS_DAYS:
            horizon_hours = horizon_days * 24
            X, y = build_supervised_frame(
                panel,
                target,
                horizon_hours,
                is_load=(target == "load"),
                extra_target_time_features=extra_features_by_target.get(target),
            )

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


def periodic_windows(val_start: pd.Timestamp, test_start: pd.Timestamp, n_windows: int = 6) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [val_start, test_start) into `n_windows` equal-length chunks.

    `n_windows=6` gives ~4-month windows over this project's default
    730-day (~2-year) validation period (see `dhf.splits.VAL_DAYS`).
    Deliberately *not* snapped to calendar quarter starts (Jan/Apr/Jul/Oct
    1): since `val_start` rarely lands exactly on one, that approach leaves
    an extra short stub window and produces n_windows+1 chunks instead of
    n_windows.
    """
    boundaries = pd.date_range(val_start, test_start, periods=n_windows + 1)
    return list(zip(boundaries[:-1], boundaries[1:]))


def fit_and_evaluate_walkforward(
    panel: pd.DataFrame,
    model_factory: Callable[[], object],
    importance_fn: Callable[[object, list[str]], pd.Series],
    extra_features_by_target: dict[str, pd.DataFrame] | None = None,
    n_windows: int = 6,
) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame], dict[tuple[str, int], pd.Series]]:
    """Same contract and return shape as `fit_and_evaluate`, but refits each
    (target, horizon) model every `n_windows`-th of the validation period on
    an expanding window (all data strictly before that period) instead of
    fitting once on the pre-validation training set.

    Every period's own model only predicts its own period; those
    predictions are stitched into one continuous validation-period series
    per (target, horizon), so `results`' MAE/RMSE stay directly comparable
    to a single-fit scenario's (same "error over the full validation
    period" semantics) — only the refit schedule differs, everything else
    (features, model, hyperparameters) held constant, per this project's
    "one delta at a time" convention (see docs/modeling_plan.md). Feature
    importances are averaged across periods rather than taken from a
    single fit.
    """
    val_start, test_start = time_split(panel.index)
    extra_features_by_target = extra_features_by_target or {}
    windows = periodic_windows(val_start, test_start, n_windows=n_windows)

    rows = []
    val_predictions: dict[tuple[str, int], pd.DataFrame] = {}
    importances: dict[tuple[str, int], pd.Series] = {}

    for target in TARGETS:
        for horizon_days in HORIZONS_DAYS:
            horizon_hours = horizon_days * 24
            X, y = build_supervised_frame(
                panel,
                target,
                horizon_hours,
                is_load=(target == "load"),
                extra_target_time_features=extra_features_by_target.get(target),
            )

            period_cells = []
            period_importances = []
            train_sizes = []
            for window_start, window_end in windows:
                train_mask = X.index < window_start
                window_mask = (X.index >= window_start) & (X.index < window_end)
                X_train, y_train = X[train_mask], y[train_mask]
                X_window, y_window = X[window_mask], y[window_mask]
                if X_window.empty:
                    continue
                train_sizes.append(len(X_train))

                model = model_factory()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_window)

                cell = pd.DataFrame({"actual": y_window, "predicted": y_pred})
                cell.index = X_window.index + pd.Timedelta(hours=horizon_hours)
                period_cells.append(cell)

                importance = importance_fn(model, list(X_train.columns))
                period_importances.append(importance / importance.sum())

            cell = pd.concat(period_cells).sort_index()
            val_predictions[(target, horizon_days)] = cell
            importances[(target, horizon_days)] = pd.concat(period_importances, axis=1).mean(axis=1)

            rows.append(
                {
                    "target": target,
                    "horizon_days": horizon_days,
                    "mae": mean_absolute_error(cell["actual"], cell["predicted"]),
                    "rmse": mean_squared_error(cell["actual"], cell["predicted"]) ** 0.5,
                    "n_train": train_sizes[-1],
                    "n_val": len(cell),
                    "n_refits": len(period_cells),
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


def plot_scatter_actual_vs_predicted(val_predictions: dict[tuple[str, int], pd.DataFrame], color: str) -> plt.Figure:
    """Target x horizon grid of hourly actual-vs-predicted scatter, full validation period.

    The black 1:1 line marks perfect agreement; a cloud that consistently
    sits above/below it (rather than scattered evenly around it) reveals
    systematic bias, not just random error — e.g. points below the line at
    high actual values mean peaks are being under-predicted.
    """
    fig, axes = plt.subplots(len(TARGETS), len(HORIZONS_DAYS), figsize=(15, 18))

    for row, target in enumerate(TARGETS):
        for col, horizon_days in enumerate(HORIZONS_DAYS):
            ax = axes[row, col]
            cell = val_predictions[(target, horizon_days)]
            ax.scatter(cell["actual"], cell["predicted"], s=2, alpha=0.15, color=color, linewidths=0)
            max_val = max(cell["actual"].max(), cell["predicted"].max())
            ax.plot([0, max_val], [0, max_val], color=ACTUAL_COLOR, linewidth=1, zorder=3)
            corr = cell["actual"].corr(cell["predicted"])
            ax.text(0.05, 0.95, f"corr={corr:.3f}", transform=ax.transAxes, va="top", fontsize=9)
            if row == 0:
                ax.set_title(f"{horizon_days}d horizon")
            if col == 0:
                ax.set_ylabel(f"{TARGET_LABELS[target]}\nPredicted (MW)")
            if row == len(TARGETS) - 1:
                ax.set_xlabel("Actual (MW)")
            style_axis(ax)

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
