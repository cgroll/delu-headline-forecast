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
# # EDA: data availability and headline variables
#
# Two questions before any modeling: how much history does each SMARD
# series actually have, and what do the four headline variables look like
# over time?

# %%
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from dhf.paths import ProjPaths
from dhf.splits import time_split
from dhf.viz import ACTUAL_COLOR, TARGET_COLORS, TARGET_LABELS, style_axis

paths = ProjPaths()

TARGETS = ["pv", "wind_onshore", "wind_offshore", "load"]

panel = pd.read_parquet(paths.target_panel_file)
val_start, test_start = time_split(panel.index)
panel_start, panel_end = panel.index.min(), panel.index.max()

# %% [markdown]
# ## Data availability and the train / validation / test split
#
# Each raw SMARD series' own coverage (dark bars), against the fixed
# chronological split used by every baseline model (see
# `docs/modeling_plan.md`): pv and wind_onshore have several years more
# history than wind_offshore and load, which only start once the DE-LU
# market area itself does (2018-10) — see `docs/data_sources.md`. The
# combined panel used for modeling is trimmed to the common overlap, so
# training only ever starts at the latest of the four start dates.
#
# ```{figure} ../../output/images/03_data_availability.png
# :name: fig-03-data-availability
# Raw per-series data availability (dark bars) against the train /
# validation / test split (background bands) used by every baseline model.
# ```

# %%
availability = pd.DataFrame(
    [
        {
            "target": name,
            "start": pd.read_parquet(paths.smard_raw_file(name)).index.min(),
            "end": pd.read_parquet(paths.smard_raw_file(name)).index.max(),
        }
        for name in TARGETS
    ]
)

fig, ax = plt.subplots(figsize=(11, 4))

ax.axvspan(panel_start, val_start, color="#86b6ef", alpha=0.35, zorder=1, label="Train")
ax.axvspan(val_start, test_start, color="#3987e5", alpha=0.35, zorder=1, label="Validation")
ax.axvspan(test_start, panel_end, color="#184f95", alpha=0.35, zorder=1, label="Test (reserved)")

y_positions = range(len(availability))
for y, row in zip(y_positions, availability.itertuples()):
    ax.barh(y, row.end - row.start, left=row.start, height=0.5, color="#0b0b0b", zorder=2, label="Data available" if y == 0 else None)

ax.set_yticks(list(y_positions))
ax.set_yticklabels([TARGET_LABELS[t] for t in availability["target"]])
ax.invert_yaxis()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
style_axis(ax)
ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
ax.set_title("Data availability vs. train / validation / test split", pad=45)
ax.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
fig.savefig(paths.images_path / "03_data_availability.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Headline variables over time
#
# Full history of the combined (inner-joined) panel — 2018-09-30 onward.
#
# ```{figure} ../../output/images/03_headline_variables_timeseries.png
# :name: fig-03-headline-variables-timeseries
# The four headline variables over their full common history. Note the
# differing y-axis scales — wind onshore capacity/output dwarfs offshore,
# and load sits on a much higher, narrower band than any single generation
# source.
# ```

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

for ax, target in zip(axes.flat, TARGETS):
    ax.plot(panel.index, panel[target], color=TARGET_COLORS[target], linewidth=0.4)
    ax.set_title(TARGET_LABELS[target])
    ax.set_ylabel("MW")
    style_axis(ax)

fig.suptitle("Headline variables — full history", y=1.02)
fig.tight_layout()
fig.savefig(paths.images_path / "03_headline_variables_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Rolling 30-day volatility
#
# 30-day rolling standard deviation of each headline variable — a way to
# see which *periods* are more or less volatile, not just what the level
# is. For pv/wind this mixes two effects: the diurnal on/off cycle's own
# amplitude (which itself varies seasonally — e.g. PV swings harder on long
# summer days than short winter ones) and genuine day-to-day weather
# variability, so a high value here doesn't by itself mean "stormier
# weather" — only that the series moved around more, for whichever reason.
#
# ```{figure} ../../output/images/03_rolling_volatility.png
# :name: fig-03-rolling-volatility
# 30-day rolling standard deviation of each headline variable, full history.
# ```

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

for ax, target in zip(axes.flat, TARGETS):
    # min_periods=15 days: without it, the window's first few hours produce
    # noisy std estimates from a handful of points, showing up as a sharp
    # spike right at the start of each series.
    rolling_std = panel[target].rolling("30D", min_periods=24 * 15).std()
    ax.plot(rolling_std.index, rolling_std, color=TARGET_COLORS[target], linewidth=0.8)
    ax.set_title(TARGET_LABELS[target])
    ax.set_ylabel("30d rolling std (MW)")
    style_axis(ax)

fig.suptitle("Headline variables — 30-day rolling volatility", y=1.02)
fig.tight_layout()
fig.savefig(paths.images_path / "03_rolling_volatility.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Seasonality: standard deviation by day-of-year
#
# For each calendar day-of-year (Jan 1 .. Dec 31), pool every hourly
# observation that falls on that day across all years in the panel and take
# their standard deviation — a climatological view of *which times of year
# are typically more volatile*. This complements the rolling-30-day chart
# above: that one shows the actual volatility trajectory over time (trend
# and season mixed together), this one averages out year-to-year
# differences and isolates the pure seasonal shape.
#
# ```{figure} ../../output/images/03_doy_std.png
# :name: fig-03-doy-std
# Standard deviation of each headline variable's hourly values, pooled by
# day-of-year across all years in the panel.
# ```

# %%
doy_std = panel.groupby(panel.index.dayofyear)[TARGETS].std()
# Map day-of-year (1..366) onto a placeholder non-leap year's dates so the
# x-axis can use real month labels — these aren't real observation dates.
doy_dates = pd.Timestamp("2001-01-01") + pd.to_timedelta(doy_std.index - 1, unit="D")

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

for ax, target in zip(axes.flat, TARGETS):
    ax.plot(doy_dates, doy_std[target], color=TARGET_COLORS[target], linewidth=1.2)
    ax.set_title(TARGET_LABELS[target])
    ax.set_ylabel("Std by day-of-year (MW)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    style_axis(ax)

fig.suptitle("Headline variables — standard deviation by day-of-year", y=1.02)
fig.tight_layout()
fig.savefig(paths.images_path / "03_doy_std.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Renewables vs. demand
#
# Stacked PV + wind onshore + wind offshore generation against load — the
# big-picture view of renewable buildout catching up to demand over the
# years. Daily means, since the full multi-year history at hourly
# resolution would just be visual noise — but that averaging also hides
# exactly the moments this chart alone can't show: at hourly resolution,
# renewables actually *exceed* load in 602 of the ~87,600 hours in this
# panel, none of which are visible here since a single high hour gets
# averaged into its day. See the next chart for that.
#
# ```{figure} ../../output/images/03_renewables_vs_demand.png
# :name: fig-03-renewables-vs-demand
# Daily-mean stacked renewable generation (PV, wind onshore, wind offshore)
# vs. daily-mean load, full history.
# ```

# %%
daily = panel[TARGETS].resample("D").mean()

fig, ax = plt.subplots(figsize=(13, 5))

ax.stackplot(
    daily.index,
    daily["pv"],
    daily["wind_onshore"],
    daily["wind_offshore"],
    colors=[TARGET_COLORS["pv"], TARGET_COLORS["wind_onshore"], TARGET_COLORS["wind_offshore"]],
    labels=[TARGET_LABELS["pv"], TARGET_LABELS["wind_onshore"], TARGET_LABELS["wind_offshore"]],
    alpha=0.85,
    zorder=2,
)
ax.plot(daily.index, daily["load"], color=ACTUAL_COLOR, linewidth=1.2, label=TARGET_LABELS["load"], zorder=3)

ax.set_ylabel("MW (daily mean)")
ax.set_title("Stacked renewable generation vs. load")
style_axis(ax)
ax.legend(loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.0, 1.14))
fig.tight_layout()
fig.savefig(paths.images_path / "03_renewables_vs_demand.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Renewable share of load — daily peak
#
# Daily *maximum* (not mean) of the hourly renewable/load ratio — preserves
# the within-day peak instead of averaging it away, so hours where
# renewables actually cover all of load (ratio >= 1, above the reference
# line) are visible as spikes. Even ratios comfortably below 1 are still
# curtailment-risk candidates in practice: Germany's inflexible ("must-run")
# baseload generation can't ramp down to zero, so once renewables cover
# most of *residual* demand, wind/solar is usually the cheapest thing left
# to curtail — not just literal renewables > load moments. Not a direct
# curtailment measurement either way (see the note above), but a much more
# honest "candidate periods" view than the daily-mean chart above.
#
# ```{figure} ../../output/images/03_renewable_share_of_load.png
# :name: fig-03-renewable-share-of-load
# Daily maximum of the hourly (PV + wind onshore + wind offshore) / load
# ratio, full history. The dashed line marks 100% (renewables alone
# covering all of that hour's load).
# ```

# %%
renewable_share = (panel["pv"] + panel["wind_onshore"] + panel["wind_offshore"]) / panel["load"]
daily_peak_share = renewable_share.resample("D").max()

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(daily_peak_share.index, daily_peak_share * 100, color=TARGET_COLORS["wind_onshore"], linewidth=0.8)
ax.axhline(100, color=ACTUAL_COLOR, linewidth=1, linestyle="--", zorder=3)
ax.set_ylabel("Daily peak renewable\nshare of load (%)")
ax.set_title("Renewable share of load — daily peak hour")
style_axis(ax)
fig.tight_layout()
fig.savefig(paths.images_path / "03_renewable_share_of_load.png", dpi=150, bbox_inches="tight")
plt.show()
