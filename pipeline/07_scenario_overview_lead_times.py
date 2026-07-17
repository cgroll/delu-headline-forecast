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
# # Scenario overview: error by lead time
#
# How MAE and RMSE evolve across forecast horizons, one line per scenario,
# one panel per headline variable. Reads every scenario's metrics file
# from `data/processed/results/` (written by each scenario's own notebook
# via `dhf.scenarios.save_scenario_metrics`) — no numbers are recomputed
# here. See
# [06_scenario_overview_tables.ipynb](06_scenario_overview_tables.ipynb)
# for the per-horizon numbers in table form, alongside scenario
# descriptions and validity.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from dhf.paths import ProjPaths
from dhf.scenarios import load_all_scenario_metrics
from dhf.viz import TARGET_LABELS, assign_scenario_colors, style_axis

paths = ProjPaths()
paths.ensure_directories()

TARGETS = ["pv", "wind_onshore", "wind_offshore", "load"]

metrics = load_all_scenario_metrics(paths)
horizons_days = sorted(metrics["horizon_days"].unique())
scenario_titles = metrics.drop_duplicates("scenario_id").set_index("scenario_id")["title"]
scenario_colors = assign_scenario_colors(sorted(metrics["scenario_id"].unique()))

print(f"{len(scenario_colors)} scenarios: {list(scenario_colors)}")


# %%
def plot_metric_by_horizon(metric: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, target in zip(axes.flat, TARGETS):
        target_metrics = metrics[metrics["target"] == target]
        for scenario_id, group in target_metrics.groupby("scenario_id"):
            group = group.sort_values("horizon_days")
            ax.plot(
                group["horizon_days"],
                group[metric],
                color=scenario_colors[scenario_id],
                marker="o",
                linewidth=2,
                label=scenario_titles[scenario_id],
                zorder=3,
            )
        ax.set_xticks(horizons_days)
        ax.set_xticklabels([f"{h}d" for h in horizons_days])
        ax.set_title(TARGET_LABELS[target])
        ax.set_ylabel(f"Validation {metric.upper()} (MW)")
        ax.set_ylim(bottom=0)
        style_axis(ax)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(scenario_colors), bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout()
    return fig


# %% [markdown]
# ## MAE by lead time
#
# ```{figure} ../../output/images/07_lead_time_mae.png
# :name: fig-07-lead-time-mae
# Validation MAE vs. forecast horizon, one line per scenario.
# ```

# %%
fig = plot_metric_by_horizon("mae")
fig.suptitle("MAE by forecast horizon, all scenarios", y=1.1)
fig.savefig(paths.images_path / "07_lead_time_mae.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## RMSE by lead time
#
# ```{figure} ../../output/images/07_lead_time_rmse.png
# :name: fig-07-lead-time-rmse
# Validation RMSE vs. forecast horizon, one line per scenario.
# ```

# %%
fig = plot_metric_by_horizon("rmse")
fig.suptitle("RMSE by forecast horizon, all scenarios", y=1.1)
fig.savefig(paths.images_path / "07_lead_time_rmse.png", dpi=150, bbox_inches="tight")
