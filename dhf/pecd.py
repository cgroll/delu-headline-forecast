"""Capacity-weighted aggregation of PECD zone-level capacity factors to a
single national (DE-LU bidding-zone-wide) time series, weighted by MaStR
installed capacity per region/zone.

This is the "single capacity-weighted national effective capacity factor
per technology" starting point described in docs/modeling_plan.md — later
refinements (e.g. bringing in MaStR's behind-the-meter PV split) are
separate, comparable experiments, not built into these helpers.
"""

import io
import zipfile

import pandas as pd


def load_capacity_factor_zip(zip_path) -> pd.DataFrame:
    """Read one PECD ZIP (one CSV per year inside) into a DE-only, hourly-indexed wide DataFrame.

    Shared by pipeline/11_process_pecd_capacity_factors.py (capacity
    factors) and pipeline/15_process_pecd_temperature.py (temperature) —
    same ZIP/CSV structure for both.
    """
    parts = []
    with zipfile.ZipFile(zip_path) as z:
        for csv_name in z.namelist():
            with z.open(csv_name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8")
                header_idx = next(i for i, line in enumerate(text) if line.startswith("Date,"))
            with z.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    skiprows=header_idx,
                    parse_dates=["Date"],
                    index_col="Date",
                    usecols=lambda c: c == "Date" or c.startswith("DE"),
                )
            parts.append(df)
    combined = pd.concat(parts).sort_index()
    combined.index.name = "timestamp"
    if combined.index.duplicated().any():
        n_dups = int(combined.index.duplicated().sum())
        print(f"  Dropping {n_dups} duplicate timestamps in {zip_path.name}")
        combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def hourly_capacity_by_region(capacity_by_region_month: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Broadcast a monthly (region_code, month, capacity_mw) panel to `target_index`'s hourly resolution.

    Capacity is held constant within each calendar month — MaStR's own
    resolution. Joins on year-month (a `Period`), not the raw month
    timestamp, since MaStR's `month` is an end-of-month date whose exact day
    doesn't need to match anything on the capacity-factor side.
    """
    wide = (
        capacity_by_region_month.assign(year_month=lambda d: d["month"].dt.to_period("M"))
        .groupby(["year_month", "region_code"])["capacity_mw"]
        .sum()
        .unstack("region_code")
    )
    hourly = wide.reindex(target_index.to_period("M"))
    hourly.index = target_index
    return hourly


def national_capacity_mw(capacity_by_region_month: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
    """Total installed capacity (MW), hourly, summed across all regions in the panel."""
    return hourly_capacity_by_region(capacity_by_region_month, target_index).sum(axis=1)


def national_capacity_weighted_cf(capacity_factor: pd.DataFrame, capacity_by_region_month: pd.DataFrame) -> pd.Series:
    """Capacity-weighted national mean capacity factor, hourly.

    `capacity_factor` is wide: hourly index, one column per region (for
    solar, collapse the technology axis first with
    `average_across_technologies`). Regions present in one input but not
    the other (e.g. a PEOF zone with capacity-factor data but no installed
    MaStR capacity yet) are dropped from the weighted average — equivalent
    to treating them as zero-weight.
    """
    capacity_hourly = hourly_capacity_by_region(capacity_by_region_month, capacity_factor.index)
    common_regions = capacity_factor.columns.intersection(capacity_hourly.columns)
    cf, cap = capacity_factor[common_regions], capacity_hourly[common_regions]
    return (cf * cap).sum(axis=1) / cap.sum(axis=1)


def average_across_technologies(capacity_factor: pd.DataFrame) -> pd.DataFrame:
    """Collapse a (technology, region) MultiIndex-column solar CF frame to one CF per region.

    Simple unweighted mean across the 4 PECD technology sub-types — there's
    no MaStR capacity split by technology (only by behind-the-meter
    category) to weight them by yet, so this is the "first feature"
    simplification (see docs/modeling_plan.md).
    """
    return capacity_factor.T.groupby(level="region").mean().T


def build_weather_capacity_features(
    capacity_factor: pd.DataFrame, capacity_by_region_month: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    """The phase-2 "start simple" feature pair for one technology (see docs/modeling_plan.md):

    - `{prefix}_effective_cf`: the capacity-weighted national mean capacity
      factor (dimensionless, 0..1) — the perfect-foresight weather signal.
    - `{prefix}_capacity_mw`: total installed capacity (MW) — the capacity
      growth trajectory, independent of weather.

    Both hourly, aligned to `capacity_factor`'s index. Feed into
    `dhf.features.build_supervised_frame`'s `extra_target_time_features` —
    both describe the target time, not the forecast origin. `prefix`
    (e.g. "pv", "wind_onshore") keeps columns distinct when a single
    target's feature set draws on more than one technology — e.g. load,
    which may respond to PV output through behind-the-meter self-consumption
    suppressing grid-visible demand, not just its own lags/calendar.
    """
    return pd.DataFrame(
        {
            f"{prefix}_effective_cf": national_capacity_weighted_cf(capacity_factor, capacity_by_region_month),
            f"{prefix}_capacity_mw": national_capacity_mw(capacity_by_region_month, capacity_factor.index),
        }
    )


def build_temperature_feature(temperature: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw `pecd_temperature_file` frame's "DE" column to a self-documenting name.

    No capacity weighting involved (temperature isn't tied to any
    installed-capacity panel) — this just gives `dhf.features.build_supervised_frame`'s
    `extra_target_time_features` a clearly-labeled column.
    """
    return temperature.rename(columns={"DE": "temperature_de_c"})
