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
# # Phase 2 scenario: perfect-foresight weather, periodic refit (LightGBM)
#
# `scenario_id = perfect_foresight_weather_lightgbm_quarterly_refit`
# (name kept for continuity with earlier iterations of this scenario — the
# actual refit cadence below is 6 equal-length periods, not literal
# calendar quarters; see `dhf.reporting.periodic_windows`.)
#
# Identical features, model, and hyperparameters as
# `13_perfect_foresight_weather_lightgbm.ipynb` — the only thing that
# changes is the refit schedule: instead of one model fit on the
# pre-validation training set and scored across the whole ~2-year
# validation period, the model is **refit 6 times** (~4-month windows) on
# an expanding window (all data strictly before that window), and only used
# to predict that one window (see `dhf.reporting.fit_and_evaluate_walkforward`).
#
# Motivation: `13_perfect_foresight_weather_lightgbm.ipynb`'s scatter plot
# showed a hard ceiling in offshore wind predictions — LightGBM (a
# tree-based model) can't predict beyond the range of target values it saw
# during training, and offshore capacity nearly tripled over the data's
# history, so a model trained once on a stale multi-year window
# systematically under-predicts once the fleet grows past what training
# ever saw. Periodic refitting keeps the training window's upper end closer
# to "now," which should shrink (not necessarily eliminate — see
# docs/modeling_plan.md's note on the target-rescaling alternative, not
# applied here) that gap.
#
# This is a **refit-schedule-only** change relative to
# `13_perfect_foresight_weather_lightgbm.ipynb` — the target is still
# absolute MW, not a capacity-factor ratio (that's a separate, not-yet-tried
# lever, deliberately not combined with this one so the two can't be
# confused — see docs/modeling_plan.md's "one delta at a time" convention).

# %%
import pandas as pd
from lightgbm import LGBMRegressor

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

SCENARIO_ID = "perfect_foresight_weather_lightgbm_quarterly_refit"
COLOR = KNOWN_SCENARIO_COLORS[SCENARIO_ID]
MODEL_LABEL = MODEL_LABELS["lightgbm"]

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
# Identical `LGBMRegressor` config as `13_perfect_foresight_weather_lightgbm.ipynb`;
# `n_refits` in the results table below shows how many windows each
# (target, horizon) cell was actually split into (6), so any MAE difference
# against the single-fit scenario is attributable to the refit schedule
# alone.

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


results, val_predictions, importances = fit_and_evaluate_walkforward(panel, make_model, importance_fn, extra_features_by_target)

save_scenario_metrics(
    paths,
    scenario_id=SCENARIO_ID,
    title=f"{MODEL_LABEL} + perfect-foresight weather (periodic refit)",
    description=(
        "Identical to perfect_foresight_weather_lightgbm (same features, model, hyperparameters), "
        "but refit 6 times (~4-month windows) on an expanding window instead of once — tests whether "
        "that alone narrows offshore wind's extrapolation ceiling (target is still absolute MW, not rescaled)."
    ),
    notebook="16_perfect_foresight_weather_lightgbm_quarterly_refit",
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
# ```{figure} ../../output/images/16_qr_mae_by_horizon.png
# :name: fig-16-qr-mae-by-horizon
# Quarterly-refit LightGBM validation MAE by forecast horizon, one panel per
# target.
# ```

# %%
fig = plot_mae_by_horizon(results, color=COLOR)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (quarterly refit) — validation MAE by forecast horizon", y=1.04)
fig.savefig(paths.images_path / "16_qr_mae_by_horizon.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — summer window
#
# ```{figure} ../../output/images/16_qr_actual_vs_predicted_summer.png
# :name: fig-16-qr-actual-vs-predicted-summer
# Actual (black) vs. quarterly-refit LightGBM (yellow), 2024-07-01 .. 2024-07-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["summer"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (quarterly refit) — actual vs. predicted, summer window", y=1.04)
fig.savefig(paths.images_path / "16_qr_actual_vs_predicted_summer.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — winter window
#
# ```{figure} ../../output/images/16_qr_actual_vs_predicted_winter.png
# :name: fig-16-qr-actual-vs-predicted-winter
# Actual (black) vs. quarterly-refit LightGBM (yellow), 2025-01-01 .. 2025-01-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["winter"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (quarterly refit) — actual vs. predicted, winter window", y=1.04)
fig.savefig(paths.images_path / "16_qr_actual_vs_predicted_winter.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — scatter, full validation period
#
# The offshore wind panel here is the key comparison — check whether the
# hard ceiling from `13_perfect_foresight_weather_lightgbm.ipynb`'s
# equivalent chart narrows or persists.
#
# ```{figure} ../../output/images/16_qr_scatter_actual_vs_predicted.png
# :name: fig-16-qr-scatter-actual-vs-predicted
# Hourly actual vs. predicted, full validation period, by target and horizon.
# ```

# %%
fig = plot_scatter_actual_vs_predicted(val_predictions, color=COLOR)
fig.suptitle(
    f"{MODEL_LABEL} + perfect-foresight weather (quarterly refit) — actual vs. predicted scatter, full validation period",
    y=1.01,
)
fig.savefig(paths.images_path / "16_qr_scatter_actual_vs_predicted.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Error analysis — full validation period
#
# ```{figure} ../../output/images/16_qr_rolling_error.png
# :name: fig-16-qr-rolling-error
# 7-day rolling mean absolute error over the full validation period, by
# target and horizon.
# ```

# %%
fig = plot_rolling_error(val_predictions)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (quarterly refit) — 7-day rolling validation MAE", y=1.06)
fig.savefig(paths.images_path / "16_qr_rolling_error.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Feature importance
#
# Averaged across each (target, horizon) cell's 6 periodic fits.
#
# ```{figure} ../../output/images/16_qr_feature_importance.png
# :name: fig-16-qr-feature-importance
# Feature importance share (gain), averaged across quarterly refits, by
# target and horizon.
# ```

# %%
fig = plot_importance_heatmaps(importances)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather (quarterly refit) — feature importance", y=1.02)
fig.savefig(paths.images_path / "16_qr_feature_importance.png", dpi=150, bbox_inches="tight")
