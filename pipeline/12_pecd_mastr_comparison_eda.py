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
# # EDA: PECD x MaStR capacity-weighted feature vs. actual generation
#
# First feature (see docs/modeling_plan.md): for each of pv, wind_onshore,
# wind_offshore, aggregate PECD's zone-level capacity factors to one
# DE-LU-wide, capacity-weighted time series — weights are MaStR's installed
# capacity per region/zone, updated monthly (NUTS2 for solar, PEON/PEOF for
# wind, since v4.2 doesn't offer NUTS2 for wind — see docs/data_sources.md).
# Multiplying by total installed capacity gives an *implied generation*
# figure (MW), directly comparable to SMARD's actual generation.
#
# This is a perfect-foresight sanity check with zero modeling involved: if
# PECD's weather-driven capacity factors and MaStR's capacity figures
# already track actual generation closely on their own, that's a strong
# early signal for CLAUDE.md's phase 2 (does weather have explanatory
# power at all, before any model tries to use it).

# %%
import matplotlib.pyplot as plt
import pandas as pd

from dhf.paths import ProjPaths
from dhf.pecd import average_across_technologies, national_capacity_mw, national_capacity_weighted_cf
from dhf.viz import TARGET_COLORS, TARGET_LABELS, style_axis

paths = ProjPaths()

TARGETS = ["pv", "wind_onshore", "wind_offshore"]

target_panel = pd.read_parquet(paths.target_panel_file)


# %% [markdown]
# ## Build the implied-generation series per target
#
# Each target's MaStR capacity panel is prepared slightly differently:
# solar sums all 4 behind-the-meter categories by NUTS2 region (no
# per-category CF split available yet, see `dhf/pecd.py`); onshore/offshore
# wind each have a single `series` value already, by PEON/PEOF zone.

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

CF_AND_CAPACITY = {
    "pv": (solar_cf, solar_capacity),
    "wind_onshore": (wind_onshore_cf, wind_onshore_capacity),
    "wind_offshore": (wind_offshore_cf, wind_offshore_capacity),
}

implied = {}
for target, (cf, capacity) in CF_AND_CAPACITY.items():
    cf_national = national_capacity_weighted_cf(cf, capacity)
    capacity_national = national_capacity_mw(capacity, cf.index)
    implied[target] = (cf_national * capacity_national).rename(target)
    print(f"{target}: mean installed capacity {capacity_national.mean():,.0f} MW over {cf.index.min()} .. {cf.index.max()}")

# %% [markdown]
# ## Compare against actual SMARD generation
#
# Inner-joined to SMARD's own availability window per target (offshore/load
# only start once the DE-LU market area does, 2018-10 — see
# docs/data_sources.md), so each target may compare over a different span.

# %%
comparisons = {}
metrics = []
for target in TARGETS:
    df = pd.concat([target_panel[target].rename("actual"), implied[target].rename("implied")], axis=1, sort=False).dropna()
    comparisons[target] = df
    error = df["implied"] - df["actual"]
    mean_actual = df["actual"].mean()
    metrics.append(
        {
            "target": target,
            "n_hours": len(df),
            "start": df.index.min(),
            "end": df.index.max(),
            "corr": df["actual"].corr(df["implied"]),
            "mean_actual_mw": mean_actual,
            "mean_implied_mw": df["implied"].mean(),
            "mae_mw": error.abs().mean(),
            "mae_pct_of_mean": error.abs().mean() / mean_actual * 100,
            "bias_mw": error.mean(),
            "bias_pct_of_mean": error.mean() / mean_actual * 100,
            "rmse_mw": (error**2).mean() ** 0.5,
        }
    )

metrics_df = pd.DataFrame(metrics).set_index("target")
print(metrics_df.round({c: 2 for c in metrics_df.select_dtypes("number").columns}).to_string())

# %% [markdown]
# ## Daily-mean time series: actual vs. implied
#
# ```{figure} ../../output/images/12_pecd_mastr_implied_vs_actual_timeseries.png
# :name: fig-12-pecd-mastr-implied-vs-actual-timeseries
# Daily-mean actual generation (solid) vs. the PECD x MaStR capacity-weighted
# implied generation (dashed), over each target's common history.
# ```

# %%
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(13, 9), sharex=False)

for ax, target in zip(axes, TARGETS):
    daily = comparisons[target].resample("D").mean()
    ax.plot(daily.index, daily["actual"], color=TARGET_COLORS[target], linewidth=0.8, label="Actual (SMARD)")
    ax.plot(daily.index, daily["implied"], color="#0b0b0b", linewidth=0.8, linestyle="--", label="Implied (PECD x MaStR)")
    ax.set_title(TARGET_LABELS[target])
    ax.set_ylabel("MW")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False)

fig.suptitle("Daily-mean actual vs. implied generation", y=1.02)
fig.tight_layout()
fig.savefig(paths.images_path / "12_pecd_mastr_implied_vs_actual_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Hourly scatter: actual vs. implied
#
# ```{figure} ../../output/images/12_pecd_mastr_implied_vs_actual_scatter.png
# :name: fig-12-pecd-mastr-implied-vs-actual-scatter
# Hourly actual vs. implied generation, with the 1:1 line (perfect agreement)
# in black. Points below/above the line mean the implied feature under-/
# over-states actual generation at that hour.
# ```

# %%
fig, axes = plt.subplots(1, len(TARGETS), figsize=(15, 5))

for ax, target in zip(axes, TARGETS):
    df = comparisons[target]
    ax.scatter(df["actual"], df["implied"], s=2, alpha=0.15, color=TARGET_COLORS[target], linewidths=0)
    max_val = max(df["actual"].max(), df["implied"].max())
    ax.plot([0, max_val], [0, max_val], color="#0b0b0b", linewidth=1, zorder=3)
    ax.set_title(f"{TARGET_LABELS[target]}\ncorr={metrics_df.loc[target, 'corr']:.3f}")
    ax.set_xlabel("Actual (MW)")
    ax.set_ylabel("Implied (MW)")
    style_axis(ax)

fig.tight_layout()
fig.savefig(paths.images_path / "12_pecd_mastr_implied_vs_actual_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
