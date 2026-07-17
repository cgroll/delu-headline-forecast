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
# # Phase 2 scenario: perfect-foresight weather, periodic refit (CatBoost)
#
# `scenario_id = perfect_foresight_weather_catboost_quarterly_refit`
#
# Identical features and periodic-refit schedule (6 equal-length windows,
# ~4 months each) as
# `16_perfect_foresight_weather_lightgbm_quarterly_refit.ipynb` — the only
# thing that changes here is the model: CatBoost instead of LightGBM, with
# matching hyperparameters (300 trees, max_depth=6, learning_rate=0.05)
# where the two APIs line up, so any MAE difference is attributable to the
# algorithm alone, not the refit schedule or features.
#
# Note this is **not** expected to fix offshore wind's extrapolation
# ceiling (see `13_perfect_foresight_weather_lightgbm.ipynb`) — CatBoost is
# also a tree ensemble and shares the same inability to predict beyond the
# range of target values seen during training. It's here as a general
# algorithm comparison, not a targeted fix for that specific problem (see
# docs/modeling_plan.md's target-rescaling note for the actual fix).

# %%
import pandas as pd
from catboost import CatBoostRegressor

from dhf.paths import ProjPaths
from dhf.pecd import average_across_technologies, build_temperature_feature, build_weather_capacity_features
from dhf.reporting import (
    describe_feature_sets,
    fit_and_evaluate_walkforward,
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

SCENARIO_ID = "perfect_foresight_weather_catboost_quarterly_refit"
COLOR = KNOWN_SCENARIO_COLORS[SCENARIO_ID]
MODEL_LABEL = MODEL_LABELS["catboost"]

panel = pd.read_parquet(paths.target_panel_file)

# %% [markdown]
# ## Build the weather + capacity features
#
# Identical to `13_perfect_foresight_weather_lightgbm.ipynb` — see there for
# the per-technology / load rationale.

# %%
solar_cf = average_across_technologies(pd.read_parquet(paths.pecd_solar_capacity_factors_file))
nuts2_capacity = pd.read_parquet(paths.mastr_capacity_by_nuts2_month_file)
solar_capacity = (
    nuts2_capacity[nuts2_capacity["series"].str.startswith("solar_")]
    .groupby(["region_code", "month"], as_index=False)["capacity_mw"]
    .sum()
)

wind_onshore_cf = pd.read_parquet(paths.pecd_wind_onshore_capacity_factors_file)
wind_onshore_capacity = pd.read_parquet(paths.mastr_capacity_by_peon_month_file)

wind_offshore_cf = pd.read_parquet(paths.pecd_wind_offshore_capacity_factors_file)
wind_offshore_capacity = pd.read_parquet(paths.mastr_capacity_by_peof_month_file)

pv_features = build_weather_capacity_features(solar_cf, solar_capacity, prefix="pv")
wind_onshore_features = build_weather_capacity_features(wind_onshore_cf, wind_onshore_capacity, prefix="wind_onshore")
wind_offshore_features = build_weather_capacity_features(wind_offshore_cf, wind_offshore_capacity, prefix="wind_offshore")
temperature_features = build_temperature_feature(pd.read_parquet(paths.pecd_temperature_file))

extra_features_by_target = {
    "pv": pv_features,
    "wind_onshore": wind_onshore_features,
    "wind_offshore": wind_offshore_features,
    "load": pd.concat([pv_features, wind_onshore_features, wind_offshore_features, temperature_features], axis=1),
}

# %% [markdown]
# ## Features used

# %%
for target, features in describe_feature_sets(panel, extra_features_by_target).items():
    print(f"{target:15s} ({len(features)} features): {', '.join(features)}")

# %% [markdown]
# ## Fit and evaluate — periodic refit
#
# `n_refits` in the results table below shows how many windows each
# (target, horizon) cell was actually split into (6). `verbose=False` since
# CatBoost's per-iteration training log would otherwise print for every one
# of the ~72 fits in this scenario.

# %%
def make_model():
    return CatBoostRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbose=False,
    )


def importance_fn(model, feature_names):
    return pd.Series(model.feature_importances_, index=feature_names)


results, val_predictions, importances = fit_and_evaluate_walkforward(panel, make_model, importance_fn, extra_features_by_target)

save_scenario_metrics(
    paths,
    scenario_id=SCENARIO_ID,
    title=f"{MODEL_LABEL} + perfect-foresight weather (periodic refit)",
    description=(
        "Same features and periodic-refit schedule (6 windows) as "
        "perfect_foresight_weather_lightgbm_quarterly_refit, CatBoost instead of LightGBM "
        "(matching hyperparameters) — general algorithm comparison, not a fix for offshore's "
        "extrapolation ceiling (CatBoost is also tree-based)."
    ),
    notebook="18_perfect_foresight_weather_catboost_quarterly_refit",
    valid=False,
    results=results,
)
print(f"Saved -> {paths.scenario_results_file(SCENARIO_ID)}")

# %% [markdown]
# ## Results overview
#
# MAE is the primary metric; RMSE is reported alongside for context.
# `n_refits` is the number of periodic windows fit for that (target,
# horizon) cell.

# %%
overview = results.set_index(["target", "horizon_days"])[["mae", "rmse", "n_train", "n_val", "n_refits"]].round(1)
overview

# %% [markdown]
# ## Validation MAE by horizon
#
# ```{figure} ../../output/images/18_cb_mae_by_horizon.png
# :name: fig-18-cb-mae-by-horizon
# Periodic-refit CatBoost validation MAE by forecast horizon, one panel per
# target.
# ```

# %%
fig = plot_mae_by_horizon(results, color=COLOR)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (periodic refit) — validation MAE by forecast horizon", y=1.04)
fig.savefig(paths.images_path / "18_cb_mae_by_horizon.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — summer window
#
# ```{figure} ../../output/images/18_cb_actual_vs_predicted_summer.png
# :name: fig-18-cb-actual-vs-predicted-summer
# Actual (black) vs. periodic-refit CatBoost (orange), 2024-07-01 .. 2024-07-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["summer"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (periodic refit) — actual vs. predicted, summer window", y=1.04)
fig.savefig(paths.images_path / "18_cb_actual_vs_predicted_summer.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — winter window
#
# ```{figure} ../../output/images/18_cb_actual_vs_predicted_winter.png
# :name: fig-18-cb-actual-vs-predicted-winter
# Actual (black) vs. periodic-refit CatBoost (orange), 2025-01-01 .. 2025-01-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["winter"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (periodic refit) — actual vs. predicted, winter window", y=1.04)
fig.savefig(paths.images_path / "18_cb_actual_vs_predicted_winter.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — scatter, full validation period
#
# Compare the offshore wind panel against
# `16_perfect_foresight_weather_lightgbm_quarterly_refit.ipynb`'s equivalent
# chart — expect a similar ceiling, since CatBoost has the same
# extrapolation limitation as LightGBM.
#
# ```{figure} ../../output/images/18_cb_scatter_actual_vs_predicted.png
# :name: fig-18-cb-scatter-actual-vs-predicted
# Hourly actual vs. predicted, full validation period, by target and horizon.
# ```

# %%
fig = plot_scatter_actual_vs_predicted(val_predictions, color=COLOR)
fig.suptitle(
    f"{MODEL_LABEL} + perfect-foresight weather (periodic refit) — actual vs. predicted scatter, full validation period",
    y=1.01,
)
fig.savefig(paths.images_path / "18_cb_scatter_actual_vs_predicted.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Error analysis — full validation period
#
# ```{figure} ../../output/images/18_cb_rolling_error.png
# :name: fig-18-cb-rolling-error
# 7-day rolling mean absolute error over the full validation period, by
# target and horizon.
# ```

# %%
fig = plot_rolling_error(val_predictions)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (periodic refit) — 7-day rolling validation MAE", y=1.06)
fig.savefig(paths.images_path / "18_cb_rolling_error.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Feature importance
#
# Averaged across each (target, horizon) cell's 6 periodic fits.
#
# ```{figure} ../../output/images/18_cb_feature_importance.png
# :name: fig-18-cb-feature-importance
# Feature importance share, averaged across periodic refits, by target and
# horizon.
# ```

# %%
fig = plot_importance_heatmaps(importances)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (periodic refit) — feature importance", y=1.02)
fig.savefig(paths.images_path / "18_cb_feature_importance.png", dpi=150, bbox_inches="tight")
