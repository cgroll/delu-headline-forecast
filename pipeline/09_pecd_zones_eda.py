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
# # EDA: PECD v4.2 wind zones for Germany
#
# PECD v4.2 does not offer NUTS-level spatial aggregation for wind capacity
# factors (unlike solar, which has clean NUTS2 coverage) — only two
# pan-European zone schemes: PEON (onshore) and PEOF (offshore). Germany
# intersects 7 PEON zones (`DE01`..`DE07`) and 6 PEOF zones (`DE011`..`DE015`,
# `DE02`). See `docs/data_sources.md`.
#
# The zone boundaries aren't published as a shapefile, but the CDS "weights
# and masks" widget provides the same information rasterized: a 0.25 degree
# grid where each cell carries its fractional area coverage (0..1) for every
# zone (downloaded by `pipeline/08_download_pecd_masks.py`). This notebook
# turns that raster back into vector polygons and plots them against
# Germany's real NUTS2 boundaries, purely as a sanity check — if the zones
# look sensible, they become a candidate grouping key for MaStR wind
# capacity data (instead of NUTS2, which isn't an option for wind here).

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from shapely.ops import unary_union

from dhf.paths import ProjPaths
from dhf.viz import FIXED_CATEGORICAL_ORDER

paths = ProjPaths()

GRID_RESOLUTION_DEG = 0.25
COVERAGE_THRESHOLD = 0.5  # a cell "belongs" to a zone once the zone covers >50% of it


# %% [markdown]
# ## Polygonize the masks
#
# For each German zone, keep the grid cells the zone covers by more than
# `COVERAGE_THRESHOLD` and dissolve them into one (multi)polygon. This is a
# coarse, 0.25 degree-resolution approximation of the true ENTSO-E zone
# boundary — good enough to sanity-check shape and placement, not for
# precise area calculations.


# %%
def zones_from_mask(mask_file, scheme: str, region_suffix: str = "") -> gpd.GeoDataFrame:
    """Polygonize a CDS region mask's German zones into a GeoDataFrame.

    `region_suffix` strips a fixed suffix (e.g. "_OFF") off the raw region
    codes for a cleaner `zone_id` column.
    """
    ds = xr.open_dataset(mask_file)
    de_regions = sorted(r for r in ds.region.values.tolist() if str(r).startswith("DE"))

    half = GRID_RESOLUTION_DEG / 2
    lats, lons = ds.latitude.values, ds.longitude.values
    rows = []
    for region in de_regions:
        covered = (ds["mask"].sel(region=region) > COVERAGE_THRESHOLD).values  # (lat, lon) bool
        lat_idx, lon_idx = np.nonzero(covered)
        boxes = [
            box(lons[j] - half, lats[i] - half, lons[j] + half, lats[i] + half) for i, j in zip(lat_idx, lon_idx)
        ]
        zone_id = region[: -len(region_suffix)] if region_suffix else region
        rows.append({"zone_id": zone_id, "scheme": scheme, "n_cells": len(boxes), "geometry": unary_union(boxes)})
    ds.close()
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


onshore_zones = zones_from_mask(paths.peon_mask_file, scheme="onshore")
offshore_zones = zones_from_mask(paths.peof_mask_file, scheme="offshore", region_suffix="_OFF")

wind_zones = gpd.GeoDataFrame(
    pd.concat([onshore_zones, offshore_zones], ignore_index=True),
    crs="EPSG:4326",
)
wind_zones["area_km2"] = wind_zones.to_crs("EPSG:3035").geometry.area / 1e6

print(wind_zones[["zone_id", "scheme", "n_cells", "area_km2"]].to_string(index=False))

# %%
wind_zones.to_file(paths.pecd_wind_zones_file, driver="GeoJSON")
print(f"Saved {len(wind_zones)} zones -> {paths.pecd_wind_zones_file}")

# %% [markdown]
# ## Map: PEON onshore zones vs. real NUTS2 boundaries
#
# Onshore zones (colored, labeled) over Germany's actual NUTS2 boundaries
# (thin gray). If the PEON zones roughly follow contiguous groups of NUTS2
# regions with plausible wind-regime geography (e.g. north/south, coastal
# vs. inland), they're a reasonable grouping key for MaStR onshore wind
# capacity.
#
# ```{figure} ../../output/images/09_pecd_wind_zones_de.png
# :name: fig-09-pecd-wind-zones-de
# PECD v4.2 PEON (onshore, left) and PEOF (offshore, right) wind zones
# intersecting Germany, derived from the CDS region masks, against real
# NUTS2 / country boundaries.
# ```

# %%
nuts2 = gpd.read_file(paths.mastr_nuts_regions_file)
nuts2 = nuts2[nuts2["LEVL_CODE"] == 2]
germany_nuts2 = nuts2[nuts2["NUTS_ID"].str.startswith("DE")]

country_borders = gpd.read_file(paths.mastr_country_borders_file)

fig, (ax_on, ax_off) = plt.subplots(1, 2, figsize=(16, 8))

germany_nuts2.plot(ax=ax_on, facecolor="none", edgecolor="#cfcac0", linewidth=0.6, zorder=1)
for i, (_, zone) in enumerate(onshore_zones.iterrows()):
    color = FIXED_CATEGORICAL_ORDER[i % len(FIXED_CATEGORICAL_ORDER)]
    gpd.GeoSeries([zone.geometry], crs="EPSG:4326").plot(ax=ax_on, color=color, alpha=0.55, edgecolor=color, zorder=2)
    label_point = zone.geometry.representative_point()
    ax_on.annotate(zone["zone_id"], (label_point.x, label_point.y), ha="center", fontsize=9, fontweight="bold")
ax_on.set_title("PEON onshore zones")
ax_on.set_axis_off()

country_borders.plot(ax=ax_off, facecolor="#f2f0ea", edgecolor="#cfcac0", linewidth=0.6, zorder=1)
germany_nuts2.plot(ax=ax_off, facecolor="none", edgecolor="#cfcac0", linewidth=0.4, zorder=1)
for i, (_, zone) in enumerate(offshore_zones.iterrows()):
    color = FIXED_CATEGORICAL_ORDER[i % len(FIXED_CATEGORICAL_ORDER)]
    gpd.GeoSeries([zone.geometry], crs="EPSG:4326").plot(ax=ax_off, color=color, alpha=0.55, edgecolor=color, zorder=2)
    label_point = zone.geometry.representative_point()
    ax_off.annotate(zone["zone_id"], (label_point.x, label_point.y), ha="center", fontsize=9, fontweight="bold")
ax_off.set_xlim(3, 15)
ax_off.set_ylim(53, 56)
ax_off.set_title("PEOF offshore zones")
ax_off.set_axis_off()

fig.savefig(paths.images_path / "09_pecd_wind_zones_de.png", dpi=150, bbox_inches="tight")
plt.show()
