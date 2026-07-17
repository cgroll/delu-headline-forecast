"""Scenario bookkeeping: each modeling scenario writes one metrics file here;
the overview pages (see pipeline/06_*, pipeline/07_*) read all of them back.

Mirrors the pattern in ~/research/flexa-challenge (see docs/modeling_plan.md
§ Bookkeeping convention), adapted from per-scenario JSON + a CSV-aggregation
stage to per-scenario parquet files read directly by the overview pages —
there's no separate "aggregate" stage since both overview pages already need
to load and reshape the full per-(target, horizon) detail, not just a single
summary number per scenario.
"""

import pandas as pd

from dhf.paths import ProjPaths

METADATA_COLUMNS = ["scenario_id", "title", "description", "notebook", "valid"]


def save_scenario_metrics(
    paths: ProjPaths,
    scenario_id: str,
    title: str,
    description: str,
    notebook: str,
    valid: bool,
    results: pd.DataFrame,
) -> None:
    """Attach scenario metadata to a (target, horizon, mae, rmse, ...) results frame and save it.

    `notebook` is the book notebook filename stem (no directory, no
    extension), used by the overview pages to link back to this scenario's
    detail page. `valid` should be False only once a scenario uses
    information that wouldn't really be available at forecast time (e.g.
    perfect-foresight weather) — see docs/modeling_plan.md.
    """
    out = results.copy()
    out["scenario_id"] = scenario_id
    out["title"] = title
    out["description"] = description
    out["notebook"] = notebook
    out["valid"] = valid
    out.to_parquet(paths.scenario_results_file(scenario_id), index=False)


def load_all_scenario_metrics(paths: ProjPaths) -> pd.DataFrame:
    """Load and concatenate every scenario's metrics file in `results_path`."""
    files = sorted(paths.results_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No scenario results found in {paths.results_path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
