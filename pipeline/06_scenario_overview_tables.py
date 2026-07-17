"""Build the scenario-comparison overview page: one table per headline variable.

Pure script — writes book/notebooks/06_scenario_overview_tables.ipynb
directly via nbformat (markdown cells only, no code execution), mirroring
the pattern in ~/research/flexa-challenge and documented in
docs/modeling_plan.md § Bookkeeping convention. Re-run whenever a new
scenario's metrics file is added to data/processed/results/.
"""

import nbformat as nbf
import pandas as pd

from dhf.paths import ProjPaths
from dhf.scenarios import load_all_scenario_metrics
from dhf.viz import TARGET_LABELS

paths = ProjPaths()
paths.ensure_directories()

TARGETS = ["pv", "wind_onshore", "wind_offshore", "load"]

metrics = load_all_scenario_metrics(paths)
horizons_days = sorted(metrics["horizon_days"].unique())

intro = """# Scenario overview: comparison tables

One table per headline variable, one row per scenario. **Valid** marks
whether a scenario is a fair forecast — i.e. doesn't use information that
wouldn't really be available at forecast time. Every scenario so far uses
only lagged/calendar features (see `docs/modeling_plan.md`); this only
turns to ⚠️ once a scenario uses perfect-foresight weather (phase 2) or an
unvalidated real forecast (phase 3).

Rows are sorted by MAE at the longest horizon ({max_horizon}d, best first) —
the project's main interest is the longer lead times (see
`docs/modeling_plan.md`). See
[07_scenario_overview_lead_times.ipynb](07_scenario_overview_lead_times.ipynb)
for how MAE/RMSE evolve with forecast horizon.
""".format(max_horizon=max(horizons_days))

cells = [nbf.v4.new_markdown_cell(intro)]

sort_col = f"mae_{max(horizons_days)}d"

for target in TARGETS:
    target_metrics = metrics[metrics["target"] == target]

    rows = []
    for scenario_id, group in target_metrics.groupby("scenario_id"):
        group = group.set_index("horizon_days").sort_index()
        meta = group.iloc[0]
        rows.append(
            {
                "title": meta["title"],
                "description": meta["description"],
                "notebook": meta["notebook"],
                "valid": bool(meta["valid"]),
                **{f"mae_{h}d": group.loc[h, "mae"] for h in horizons_days},
                **{f"rmse_{h}d": group.loc[h, "rmse"] for h in horizons_days},
            }
        )
    table = pd.DataFrame(rows).sort_values(sort_col)

    header = (
        ["Model", "Valid"]
        + [f"MAE {h}d" for h in horizons_days]
        + [f"RMSE {h}d" for h in horizons_days]
        + ["Description", "Notebook"]
    )
    lines = [
        f"## {TARGET_LABELS[target]}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for _, r in table.iterrows():
        valid_mark = "✅" if r["valid"] else "⚠️"
        mae_cells = " | ".join(f"{r[f'mae_{h}d']:.0f}" for h in horizons_days)
        rmse_cells = " | ".join(f"{r[f'rmse_{h}d']:.0f}" for h in horizons_days)
        link = f"[{r['notebook']}]({r['notebook']}.ipynb)"
        lines.append(f"| {r['title']} | {valid_mark} | {mae_cells} | {rmse_cells} | {r['description']} | {link} |")

    cells.append(nbf.v4.new_markdown_cell("\n".join(lines)))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}}

output_file = paths.book_notebooks_path / "06_scenario_overview_tables.ipynb"
nbf.write(nb, output_file)
print(f"Saved {len(cells) - 1} tables -> {output_file}")
