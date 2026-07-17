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
# # Baseline scenario: LightGBM
#
# `scenario_id = baseline_lightgbm`
#
# LightGBM (300 trees) per (target, horizon) cell, on the *identical*
# baseline feature set as `04_baseline_linear_regression.ipynb` —
# seasonality, lags, and (load only) weekday features. No weather or
# capacity information yet. See `docs/modeling_plan.md` for the full plan
# and `dhf/features.py` for the feature definitions.
#
# This notebook follows the same structure as the linear regression one
# (features used -> MAE/RMSE -> actual-vs-predicted -> error analysis ->
# feature importance) so the two are easy to compare section by section;
# the shared computation/plotting code lives in `dhf/reporting.py`.

# %%
import pandas as pd
from lightgbm import LGBMRegressor

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

SCENARIO_ID = "baseline_lightgbm"
COLOR = KNOWN_SCENARIO_COLORS[SCENARIO_ID]
MODEL_LABEL = MODEL_LABELS["lightgbm"]

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
# One `LGBMRegressor` per (target, horizon) cell, trained on data before
# the validation cutoff and scored on the validation period only (test is
# reserved — see `docs/modeling_plan.md`).

# %%
def make_model():
    return LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        importance_type="gain",
    )


def importance_fn(model, feature_names):
    return pd.Series(model.feature_importances_, index=feature_names)


results, val_predictions, importances = fit_and_evaluate(panel, make_model, importance_fn)

save_scenario_metrics(
    paths,
    scenario_id=SCENARIO_ID,
    title=MODEL_LABEL,
    description=(
        "LightGBM (300 trees, max_depth=6) on the identical feature set as the linear "
        "regression baseline. No weather/capacity data."
    ),
    notebook="05_baseline_lightgbm",
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
# ```{figure} ../../output/images/05_lgbm_mae_by_horizon.png
# :name: fig-05-lgbm-mae-by-horizon
# LightGBM validation MAE by forecast horizon, one panel per target.
# ```

# %%
fig = plot_mae_by_horizon(results, color=COLOR)
fig.suptitle(f"{MODEL_LABEL} — validation MAE by forecast horizon", y=1.04)
fig.savefig(paths.images_path / "05_lgbm_mae_by_horizon.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — summer window
#
# ```{figure} ../../output/images/05_lgbm_actual_vs_predicted_summer.png
# :name: fig-05-lgbm-actual-vs-predicted-summer
# Actual (black) vs. LightGBM (green), 2024-07-01 .. 2024-07-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["summer"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} — actual vs. predicted, summer window", y=1.04)
fig.savefig(paths.images_path / "05_lgbm_actual_vs_predicted_summer.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — winter window
#
# ```{figure} ../../output/images/05_lgbm_actual_vs_predicted_winter.png
# :name: fig-05-lgbm-actual-vs-predicted-winter
# Actual (black) vs. LightGBM (green), 2025-01-01 .. 2025-01-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["winter"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} — actual vs. predicted, winter window", y=1.04)
fig.savefig(paths.images_path / "05_lgbm_actual_vs_predicted_winter.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — scatter, full validation period
#
# The 1:1 line marks perfect agreement; a cloud that consistently sits off
# to one side (rather than scattered evenly around it) reveals systematic
# bias — e.g. points below the line at high actual values mean peaks are
# under-predicted.
#
# ```{figure} ../../output/images/05_lgbm_scatter_actual_vs_predicted.png
# :name: fig-05-lgbm-scatter-actual-vs-predicted
# Hourly actual vs. predicted, full validation period, by target and horizon.
# ```

# %%
fig = plot_scatter_actual_vs_predicted(val_predictions, color=COLOR)
fig.suptitle(f"{MODEL_LABEL} — actual vs. predicted scatter, full validation period", y=1.01)
fig.savefig(paths.images_path / "05_lgbm_scatter_actual_vs_predicted.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Error analysis — full validation period
#
# 7-day rolling mean absolute error over the full ~2-year validation
# period, one line per horizon. Reveals whether error is roughly constant
# over time (homoskedastic) or clusters seasonally / grows with some
# regime — e.g. PV error should track the summer/winter irradiance swing.
#
# ```{figure} ../../output/images/05_lgbm_rolling_error.png
# :name: fig-05-lgbm-rolling-error
# 7-day rolling mean absolute error over the full validation period, by
# target and horizon.
# ```

# %%
fig = plot_rolling_error(val_predictions)
fig.suptitle(f"{MODEL_LABEL} — 7-day rolling validation MAE", y=1.06)
fig.savefig(paths.images_path / "05_lgbm_rolling_error.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Feature importance
#
# LightGBM gain-based importance, normalized to a share of the total per
# (target, horizon) fit, so columns are directly comparable.
#
# ```{figure} ../../output/images/05_lgbm_feature_importance.png
# :name: fig-05-lgbm-feature-importance
# Feature importance share (gain) by target and horizon.
# ```

# %%
fig = plot_importance_heatmaps(importances)
fig.suptitle(f"{MODEL_LABEL} — feature importance", y=1.02)
fig.savefig(paths.images_path / "05_lgbm_feature_importance.png", dpi=150, bbox_inches="tight")
