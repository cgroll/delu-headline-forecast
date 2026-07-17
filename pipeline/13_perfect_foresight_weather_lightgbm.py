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
# # Phase 2 scenario: perfect-foresight weather (LightGBM)
#
# `scenario_id = perfect_foresight_weather_lightgbm`
#
# Identical model and baseline feature set as `05_baseline_lightgbm.ipynb`
# (same LightGBM hyperparameters, same lags/seasonality features), plus two
# extra features per technology — PECD v4.2's capacity-weighted national
# capacity factor and MaStR's total installed capacity, both assumed
# perfectly known at the forecast horizon (see
# `pipeline/12_pecd_mastr_comparison_eda.ipynb` for how closely that
# perfect-foresight assumption already tracks actual generation on its own,
# with no model at all). pv/wind_onshore/wind_offshore each get only their
# own technology's pair; `load` gets all three — it has no capacity factor
# of its own, but PV output in particular plausibly *suppresses*
# grid-visible load through behind-the-meter self-consumption (a household
# using its own solar power draws less from the grid), so it's a candidate
# feature worth letting the model test rather than assuming away.
#
# `load` also gets 2m temperature at country level (`temperature_de_c`) —
# the classic load driver, and the only variable here that isn't a
# capacity factor (see docs/data_sources.md for why population-weighted
# temperature isn't usable at Germany's specific level in v4.2).
#
# Since real weather/capacity aren't actually known this far in advance,
# this scenario is saved with `valid=False` (see `dhf/scenarios.py`) — it
# answers "does weather help *if* known perfectly", the phase-2 question
# from CLAUDE.md, not "would this work operationally" (that's phase 3).

# %%
import pandas as pd
from lightgbm import LGBMRegressor

from dhf.paths import ProjPaths
from dhf.pecd import average_across_technologies, build_temperature_feature, build_weather_capacity_features
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

SCENARIO_ID = "perfect_foresight_weather_lightgbm"
COLOR = KNOWN_SCENARIO_COLORS[SCENARIO_ID]
MODEL_LABEL = MODEL_LABELS["lightgbm"]

panel = pd.read_parquet(paths.target_panel_file)

# %% [markdown]
# ## Build the weather + capacity features
#
# One (`{tech}_effective_cf`, `{tech}_capacity_mw`) pair per technology —
# see `dhf/pecd.py`. Solar's capacity factor is first collapsed across its
# 4 PECD technology sub-types (no MaStR split by technology to weight them
# by yet, only by behind-the-meter category — that's the next, separate
# experiment per `docs/modeling_plan.md`).

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
#
# `load` gets all three technologies' pairs (6 extra columns) plus
# `temperature_de_c`, rather than just its own, since it has no capacity
# factor of its own but may respond to others' — see the PV
# self-consumption hypothesis above.

# %%
for target, features in describe_feature_sets(panel, extra_features_by_target).items():
    print(f"{target:15s} ({len(features)} features): {', '.join(features)}")

# %% [markdown]
# ## Fit and evaluate
#
# Identical `LGBMRegressor` config as `05_baseline_lightgbm.ipynb`, so any
# MAE difference is attributable to the extra features alone.

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


results, val_predictions, importances = fit_and_evaluate(panel, make_model, importance_fn, extra_features_by_target)

save_scenario_metrics(
    paths,
    scenario_id=SCENARIO_ID,
    title=f"{MODEL_LABEL} + perfect-foresight weather",
    description=(
        "LightGBM (300 trees, max_depth=6), identical to baseline_lightgbm plus PECD v4.2 "
        "capacity-weighted capacity factor + MaStR installed capacity per technology, both assumed "
        "perfectly known at the forecast horizon. pv/wind_onshore/wind_offshore get their own "
        "technology's pair; load gets all three (testing PV self-consumption suppressing grid load) "
        "plus PECD 2m temperature at country level."
    ),
    notebook="13_perfect_foresight_weather_lightgbm",
    valid=False,
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
# ```{figure} ../../output/images/13_pfw_mae_by_horizon.png
# :name: fig-13-pfw-mae-by-horizon
# Perfect-foresight-weather LightGBM validation MAE by forecast horizon, one
# panel per target.
# ```

# %%
fig = plot_mae_by_horizon(results, color=COLOR)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather — validation MAE by forecast horizon", y=1.04)
fig.savefig(paths.images_path / "13_pfw_mae_by_horizon.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — summer window
#
# ```{figure} ../../output/images/13_pfw_actual_vs_predicted_summer.png
# :name: fig-13-pfw-actual-vs-predicted-summer
# Actual (black) vs. perfect-foresight-weather LightGBM (magenta),
# 2024-07-01 .. 2024-07-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["summer"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather — actual vs. predicted, summer window", y=1.04)
fig.savefig(paths.images_path / "13_pfw_actual_vs_predicted_summer.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — winter window
#
# ```{figure} ../../output/images/13_pfw_actual_vs_predicted_winter.png
# :name: fig-13-pfw-actual-vs-predicted-winter
# Actual (black) vs. perfect-foresight-weather LightGBM (magenta),
# 2025-01-01 .. 2025-01-21.
# ```

# %%
fig = plot_actual_vs_predicted(val_predictions, EXAMPLE_WINDOWS["winter"], color=COLOR, model_label=MODEL_LABEL)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather — actual vs. predicted, winter window", y=1.04)
fig.savefig(paths.images_path / "13_pfw_actual_vs_predicted_winter.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Actual vs. predicted — scatter, full validation period
#
# The 1:1 line marks perfect agreement; a cloud that consistently sits off
# to one side (rather than scattered evenly around it) reveals systematic
# bias — e.g. points below the line at high actual values mean peaks are
# under-predicted. Compare against `05_baseline_lightgbm.ipynb`'s equivalent
# chart to see whether the weather features tighten the cloud or just shift
# it.
#
# ```{figure} ../../output/images/13_pfw_scatter_actual_vs_predicted.png
# :name: fig-13-pfw-scatter-actual-vs-predicted
# Hourly actual vs. predicted, full validation period, by target and horizon.
# ```

# %%
fig = plot_scatter_actual_vs_predicted(val_predictions, color=COLOR)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather — actual vs. predicted scatter, full validation period", y=1.01)
fig.savefig(paths.images_path / "13_pfw_scatter_actual_vs_predicted.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Error analysis — full validation period
#
# ```{figure} ../../output/images/13_pfw_rolling_error.png
# :name: fig-13-pfw-rolling-error
# 7-day rolling mean absolute error over the full validation period, by
# target and horizon.
# ```

# %%
fig = plot_rolling_error(val_predictions)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather — 7-day rolling validation MAE", y=1.06)
fig.savefig(paths.images_path / "13_pfw_rolling_error.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Feature importance
#
# If the weather/capacity features matter, expect `pecd_effective_cf` and
# `mastr_capacity_mw` to show up with meaningful (nonzero) importance share
# for pv/wind_onshore/wind_offshore, growing at longer horizons as the lag
# features' signal decays.
#
# ```{figure} ../../output/images/13_pfw_feature_importance.png
# :name: fig-13-pfw-feature-importance
# Feature importance share (gain) by target and horizon.
# ```

# %%
fig = plot_importance_heatmaps(importances)
fig.suptitle(f"{MODEL_LABEL} + perfect-foresight weather — feature importance", y=1.02)
fig.savefig(paths.images_path / "13_pfw_feature_importance.png", dpi=150, bbox_inches="tight")
