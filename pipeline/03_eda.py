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
from dhf.viz import TARGET_COLORS, TARGET_LABELS, style_axis

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
