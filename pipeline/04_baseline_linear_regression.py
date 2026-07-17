# ---
# jupytext:
#   text_representation:
#     format_name: percent
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Baseline scenario: Linear Regression
#
# `scenario_id = baseline_linear_regression`
#
# Plain OLS (on standardized features) per (target, horizon) cell, on the
# baseline feature set — seasonality, lags, and (load only) weekday
# features. No weather or capacity information yet. See
# `docs/modeling_plan.md` for the full plan and `dhf/features.py` for the
# feature definitions.
#
# This notebook and `05_baseline_lightgbm.ipynb` follow an identical
# structure (features used -> MAE/RMSE -> actual-vs-predicted -> error
# analysis -> feature importance) so the two models are easy to compare
# section by section; the shared computation/plotting code lives in
# `dhf/reporting.py`.

# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd

from dhf.paths import ProjPaths
from dhf.reporting import (
    describe_feature_sets,
    fit_and_evaluate,
    plot_actual_vs_predicted,
    plot_importance_heatmaps,
    plot_mae_by_horizon,
    plot_rolling_error,
    plot_scatter_actual_vs_predicted,
)
from dhf.scenarios import save_scenario_metrics
from dhf.splits import EXAMPLE_WINDOWS
from dhf.viz import KNOWN_SCENARIO_COLORS, MODEL_LABELS

paths = ProjPaths()
paths.ensure_directories()

SCENARIO_ID = "baseline_linear_regression"
COLOR = KNOWN_SCENARIO_COLORS[SCENARIO_ID]
MODEL_LABEL = MODEL_LABELS["linear_regression"]

panel = pd.read_parquet(paths.target_panel_file)

# %% [markdown]
# ## Features used
#
# Feature columns are identical across all three horizons for a given
# target (only their *values* change — see `dhf/features.py`), so it's
# enough to list them once per target.

# %%
for target, features in describe_feature_sets(panel).items():
    print(f"{target:15s} ({len(features)} features): {', '.join(features)}")

# %% [markdown]
# ## Fit and evaluate
#
# One `LinearRegression` (on standardized features, via a
# `StandardScaler` pipeline — needed so coefficient magnitude is a
# meaningful importance signal later, without changing the predictions
# themselves) per (target, horizon) cell, trained on data before the
# validation cutoff and scored on the validation period only (test is
# reserved — see `docs/modeling_plan.md`).

# %%
def make_model():
    return make_pipeline(StandardScaler(), LinearRegression())


def importance_fn(model, feature_names):
    coefs = model.named_steps["linearregression"].coef_
    return pd.Series(np.abs(coefs), index=feature_names)


results, val_predictions, importances = fit_and_evaluate(panel, make_model, importance_fn)

save_scenario_metrics(
    paths,
    scenario_id=SCENARIO_ID,
    title=MODEL_LABEL,
    description=(
        "Linear regression (standardized features) on seasonality (daily/annual) + "
        "short lags (1h, 2h, 24h, 48h) + load-only weekday features. No weather/capacity data."
    ),
    notebook="04_baseline_linear_regression",
    valid=True,
    results=results,
)
print(f"Saved -> {paths.scenario_results_file(SCENARIO_ID)}")

# %% [markdown]
# ## Results overview
#
# MAE is the primary metric; RMSE is reported alongside for context.

# %%
overview = results.set_index(["target", "horizon_days"])[["mae", "rmse", "n_train", "n_val"]].round(1)
overview

# %% [markdown]
# ## Validation MAE by horizon
#
# ```{figure} ../../output/images/04_lr_mae_by_horizon.png
# :name: fig-04-lr-mae-by-horizon
# Linear regression validation MAE by forecast horizon, one panel per target.
# ```

# %%
fig = plot_mae_by_horizon(results, color=COLOR)
fig.suptitle(f"{MODEL_LABEL} — validation MAE by forecast horizon", y=1.04)
fig.savefig(paths.images_path / "04_lr_mae_by_horizon.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — summer window
#
# ```{figure} ../../output/images/04_lr_actual_vs_predicted_summer.png
# :name: fig-04-lr-actual-vs-predicted-summer
# Actual (black) vs. linear regression (blue), 2024-07-01 .. 2024-07-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["summer"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} — actual vs. predicted, summer window", y=1.04)
fig.savefig(paths.images_path / "04_lr_actual_vs_predicted_summer.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — winter window
#
# ```{figure} ../../output/images/04_lr_actual_vs_predicted_winter.png
# :name: fig-04-lr-actual-vs-predicted-winter
# Actual (black) vs. linear regression (blue), 2025-01-01 .. 2025-01-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["winter"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} — actual vs. predicted, winter window", y=1.04)
fig.savefig(paths.images_path / "04_lr_actual_vs_predicted_winter.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — scatter, full validation period
#
# The 1:1 line marks perfect agreement; a cloud that consistently sits off
# to one side (rather than scattered evenly around it) reveals systematic
# bias — e.g. points below the line at high actual values mean peaks are
# under-predicted.
#
# ```{figure} ../../output/images/04_lr_scatter_actual_vs_predicted.png
# :name: fig-04-lr-scatter-actual-vs-predicted
# Hourly actual vs. predicted, full validation period, by target and horizon.
# ```

# %%
fig = plot_scatter_actual_vs_predicted(val_predictions, color=COLOR)
fig.suptitle(f"{MODEL_LABEL} — actual vs. predicted scatter, full validation period", y=1.01)
fig.savefig(paths.images_path / "04_lr_scatter_actual_vs_predicted.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Error analysis — full validation period
#
# 7-day rolling mean absolute error over the full ~2-year validation
# period, one line per horizon. Reveals whether error is roughly constant
# over time (homoskedastic) or clusters seasonally / grows with some
# regime — e.g. PV error should track the summer/winter irradiance swing.
#
# ```{figure} ../../output/images/04_lr_rolling_error.png
# :name: fig-04-lr-rolling-error
# 7-day rolling mean absolute error over the full validation period, by
# target and horizon.
# ```

# %%
fig = plot_rolling_error(val_predictions)
fig.suptitle(f"{MODEL_LABEL} — 7-day rolling validation MAE", y=1.06)
fig.savefig(paths.images_path / "04_lr_rolling_error.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Feature importance
#
# Absolute standardized coefficient, normalized to a share of the total
# per (target, horizon) fit, so columns are directly comparable.
#
# ```{figure} ../../output/images/04_lr_feature_importance.png
# :name: fig-04-lr-feature-importance
# Feature importance share (|standardized coefficient|) by target and
# horizon.
# ```

# %%
fig = plot_importance_heatmaps(importances)
fig.suptitle(f"{MODEL_LABEL} — feature importance", y=1.02)
fig.savefig(paths.images_path / "04_lr_feature_importance.png", dpi=150, bbox_inches="tight")
