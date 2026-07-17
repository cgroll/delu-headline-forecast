# Data sources

Detail doc for [CLAUDE.md](../CLAUDE.md). Covers what each source provides,
where the reference implementations live, and known gotchas/mismatches
between them.

## SMARD (Bundesnetzagentur) — targets + benchmarks

SMARD (`smard.de`) publishes German/Luxembourg electricity market data:
generation, load, prices, forecasts, and installed capacities.

**Reference implementation:** `~/research/world-of-energy`, package
`woe.smard` (`woe/smard/api.py`, `woe/smard/config.py`), driven by
`pipeline/01_download_smard_price_analysis_data.py` and
`pipeline/06_download_smard_forecasts.py`.

### Access method — important gotcha

The commonly-advertised SMARD API is the documented REST API at
[smard.api.bund.dev](https://smard.api.bund.dev/). `world-of-energy` does
**not** use that — it hits the *undocumented* JSON endpoints behind the
`smard.de` web app directly:

```
GET https://www.smard.de/app/chart_data/{variable_id}/{region}/index_{resolution}.json
GET https://www.smard.de/app/chart_data/{variable_id}/{region}/{variable_id}_{region}_{resolution}_{timestamp}.json
```

The first call lists available block-start timestamps (SMARD serves data in
weekly-ish blocks); the second fetches one block. `download_smard_data()` in
`woe/smard/api.py` handles both steps and concatenates the result into a
`timestamp`-indexed DataFrame. This has worked reliably for `world-of-energy`
but is unofficial — re-verify it still works before building on it, and keep
an eye on the documented API as a fallback if it breaks.

### Variables needed for this project

From `woe/smard/config.py`'s `Variable` enum (SMARD's internal integer IDs):

| Target | Variable | ID | Region | Resolution |
|---|---|---|---|---|
| PV generation | `SOLAR` | 4068 | `DE-LU` | hour |
| Wind onshore generation | `WIND_ONSHORE` | 4067 | `DE-LU` | hour |
| Wind offshore generation | `WIND_OFFSHORE` | 1225 | `DE-LU` | hour |
| Load | `TOTAL_LOAD` | 410 | `DE-LU` | hour |

`Region.DE_LU = "DE-LU"` is the market area covering Germany + Luxembourg
*from 2018-10-01*; before that it was `DE-AT-LU`. SMARD data generally starts
2015-01-01 (`DEFAULT_START_DATE` in `woe/smard/config.py`), so there's a
~3.75-year window (2015-01 to 2018-09) where DE-LU-specific series don't
exist under that exact region code — check what SMARD actually returns for
`DE-LU` before that date (likely nothing) and decide whether to start the
project's usable window at 2018-10 or backfill from `DE-AT-LU`.

Also useful, not required for phase 1/2 but worth knowing about:

- **SMARD day-ahead/intraday forecasts** (`FORECAST_SOLAR`=125,
  `FORECAST_ONSHORE`=123, `FORECAST_OFFSHORE`=3791, `FORECAST_LOAD`=411, all
  region `DE`, quarter-hour resolution) — these are TSO-published forecasts,
  a natural benchmark to compare our own short-horizon (1-day) models
  against. See `pipeline/06_download_smard_forecasts.py`.
- **SMARD installed capacities** (monthly, `CAPACITY_SOLAR`=188,
  `CAPACITY_WIND_ONSHORE`=186, `CAPACITY_WIND_OFFSHORE`=4076, region `DE`) —
  a national-level cross-check against the MaStR-derived capacity figures
  below.

## PECD v4.2 (Copernicus Climate Data Store) — weather / capacity factors

The Pan-European Climate Database, co-developed by C3S and ENTSO-E, provides
hourly (and other) renewable capacity factors and their underlying weather
drivers, at multiple spatial aggregations, from a long reanalysis
back-catalogue.

**v4.2 is the version to use.** It was published on the CDS on 2025-06-24,
and v4.1 is being phased out (deprecated for anything v4.2 covers; not
extended beyond 2021). `world-of-energy`'s existing PECD pipeline
(`pipeline/35_download_pecd_regions.py`, `36_process_pecd_regions.py`,
`50_download_pecd_nuts1_de.py`, `51_process_pecd_nuts1_de.py`) targets the
**v4.1** dataset ID `sis-energy-derived-reanalysis`. Use it as a *pattern*
(ZIP download via `cdsapi`, metadata-header CSV parsing, wide-format
parquet with MultiIndex columns) — not as a direct dependency, since v4.2's
request schema and spatial-aggregation options differ substantially (see
below, confirmed empirically against the CDS API — the CDS web form doesn't
publish a static parameter reference, so this was reverse-engineered from
the dataset's `form.json`/`constraints.json` plus test downloads).

**Requires** a `~/.cdsapirc` file with a valid CDS API key. Requests queue
server-side and can take several minutes even for a single small file —
budget for that when scripting multiple downloads.

**Docs:**
[CDS dataset page](https://cds.climate.copernicus.eu/datasets/sis-energy-pecd) ·
[v4.2 product user guide](https://confluence.ecmwf.int/pages/viewpage.action?pageId=540941752) ·
[C3S forum: v4.2 release](https://forum.ecmwf.int/t/new-pecd-version-4-2-co-developed-with-entso-e-now-available-on-the-cds/13497) ·
[C3S forum: v4.1 deprecation](https://forum.ecmwf.int/t/updates-to-the-climate-and-energy-related-variables-from-the-pan-european-climate-database-pecd-v4-2-and-deprecation-of-v4-1/14348)

### v4.2 API essentials (`sis-energy-pecd`)

Every request for historical (ERA5-based) data needs these fixed keys:

```python
{
    "pecd_version": "pecd4_2",
    "temporal_period": "historical",
    "origin": "era5_reanalysis",
    "file_version": "fv1",
    # plus: variable, technology, (energy_scenario), spatial_resolution, year, month
}
```

No `energy_product_type` or `format` key is needed (unlike v4.1) — "capacity
factor" is baked into the variable name itself, and the API returns a ZIP
(containing one CSV) regardless.

**Variables** (capacity factor only — this project doesn't need the raw
weather drivers, see CLAUDE.md phase 2 scope):
`solar_photovoltaic_generation_capacity_factor`,
`wind_power_onshore_capacity_factor`, `wind_power_offshore_capacity_factor`.

**`technology`** selects a turbine/panel spec. To represent the *currently
installed* fleet (what this project needs for perfect-foresight weighting
by MaStR capacity) rather than a hypothetical future build, use **"Existing
technologies"**: `"20"` for offshore wind, `"30"` for onshore wind. Solar has
no "existing" catch-all — instead four sub-categories that roughly mirror
MaStR's behind-the-meter split: `"60"` (industrial rooftop), `"61"`
(residential rooftop), `"62"` (utility-scale fixed), `"63"` (utility-scale
1-axis tracker).

**`energy_scenario`** ("Wind power installation area assumption",
`resource_grade_a`/`resource_grade_b`) only applies to wind. Confirmed via
the CDS constraints: technology `"20"`/`"30"` ("Existing technologies") is
**only** offered under `resource_grade_b` — `resource_grade_a` doesn't
include it. Not applicable to solar.

**`spatial_resolution` — confirmed valid options per variable** (this is
the important gotcha, superseding the original assumption that NUTS2 would
cover all three variables):

| Variable | Valid `spatial_resolution` values |
|---|---|
| Solar PV | `nuts_0`, `nuts_2`, `p2on`, `peon`, `szon` |
| Wind onshore | `p2on`, `peon` **only** — no `nuts_0`, no `nuts_2` |
| Wind offshore | `p2of`, `peof` **only** |

So **solar cleanly joins to MaStR at NUTS2**; **wind (on- and offshore) does
not** — v4.2 offers no NUTS-anything for wind, only two pan-European zone
schemes: PEON/PEOF ("Pan-European Onshore/Offshore Zones", built for
ENTSO-E's ERAA2025) and the newer P2ON/P2OF (ERAA2026 revision). This
project uses PEON/PEOF (the more established of the two schemes; P2ON/P2OF
wasn't evaluated in detail, might be worth a second look if PEON/PEOF turns
out to be undesirable for some reason).

**`year`**: historical ERA5-based coverage runs **1950–2025** for the
variables/technologies above (confirmed via `constraints.json`; 2026 exists
as a separate, differently-constrained partial entry — not used here). The
CDS year filter is respected (a request for `year: "2020"` returns exactly
that year, not the full back-catalogue, unlike some v4.1 behavior).

### Germany's PEON/PEOF zone membership

Confirmed by inspecting the actual returned CSV column headers (the zone
codes aren't documented anywhere; this required a real API call):

- **PEON (onshore): 7 zones** — `DE01`, `DE02`, `DE03`, `DE04`, `DE05`,
  `DE06`, `DE07`. These are **not** NUTS codes — a different, ENTSO-E-defined
  partition. Polygonizing them (see below) shows they tile Germany
  contiguously with no overlaps, in geographically sensible groups (e.g.
  NW-coastal, NE, W, E-central, W-central, SW, SE/Bavaria); combined area
  ≈354,000 km², matching Germany's real area (≈357,000 km²).
- **PEOF (offshore): 6 zones** — `DE011_OFF`, `DE012_OFF`, `DE013_OFF`,
  `DE014_OFF`, `DE015_OFF` (5 sub-zones in the North Sea, ≈41,000 km²
  combined) and `DE02_OFF` (1 zone covering the whole Baltic Sea, ≈16,700
  km²). Maps reasonably well onto MaStR's two offshore pseudo-regions: `DE02`
  ↔ `DEZZ-OSTSEE` is a clean 1:1, while `DEZZ-NORDSEE` would need to fan out
  across the 5 North Sea sub-zones.

### Getting the underlying grid mask (zone geometry)

The zone boundaries aren't published as a shapefile — CDS's own product
guide states the source shapefiles (`DTU-PECD40-Polygons_VF20230127` for
PEON, `PECD42_VF20241002` for PEOF, provided to C3S by ENTSO-E) are **not
publicly available**. What *is* available is the same information
rasterized, via a separate "Weights and masks" variable group (distinct
from the timeseries `variable` group above — no `temporal_period`, `origin`,
`year`, `month`, `spatial_resolution`, `technology`, or `energy_scenario`
keys apply):

```python
{
    "pecd_version": "pecd4_2",
    "file_version": "fv1",
    "variable": "peon_region_mask",  # or peof_region_mask, nuts_2_region_mask, ...
}
```

Also available in this group: `p2on_region_mask`, `p2of_region_mask`,
`nuts_0_region_mask`, `szon_region_mask`, `szof_region_mask` (a third,
"synchronous zone" scheme), plus non-region layers (`solar_pv_mask`,
`population_density_mask`, wind-speed climatologies, `city_coords`,
`latitude_weights`).

**Structure** (NetCDF, zipped to a single `.nc` file inside): dims
`(region, latitude, longitude)`, data variable `mask`. The grid is fixed at
**0.25 degree resolution, 229 latitudes × 305 longitudes**, spanning lat
75.0..18.0 (descending) and lon -31.0..45.0 — the full PECD/ERA5-Europe
domain. **Every mask variable shares this identical grid** (confirmed:
`peon_region_mask` and `nuts_2_region_mask` have byte-identical
`latitude`/`longitude` coordinate arrays) — no reprojection is needed to
combine two different masks cell-by-cell.

**Values are fractional, not boolean.** `mask` gives the fraction (0.0–1.0)
of each 0.25° grid cell actually covered by that region's polygon — a
boundary cell can have nonzero membership in several regions at once, and
fractions across *all* of a scheme's ~100–600 European regions sum to ≈1.0
per cell. Confirmed example: the grid cell at lat=54.00, lon=10.75 (near the
German/Danish border, Schleswig-Holstein) has `DE01=0.8609`, `DE02=0.0429`
in the PEON mask — the remaining ≈9.6% belongs to a neighboring Danish PEON
zone.

File sizes: PEON mask ≈85 MB (153 European regions), PEOF mask ≈35 MB (124
regions), NUTS2 mask ≈335 MB (629 regions — Eurostat NUTS2 covers far more
of Europe than either wind zone scheme).

### Deriving zone polygons or a region-to-region crosswalk from the mask

**For a visual/vector polygon per zone** (e.g. to sanity-check the zones —
see `pipeline/09_pecd_zones_eda.py` in this project): threshold each cell
and dissolve. Thresholding each zone *independently* (e.g. "keep cells
>50%") leaves visible gaps at 3-way boundary splits, where no single zone
reaches the threshold — assigning each cell to whichever zone has the
**highest** fraction there (argmax) instead avoids this; every cell touched
by the scheme gets assigned to exactly one zone.

**For a capacity-weighted region-to-region crosswalk** (e.g. redistributing
MaStR's NUTS2-aggregated capacity to PEON zones) — this is the more useful
operation and doesn't require polygonizing anything. Since both masks share
the identical grid, the area overlap between a source region and a target
zone is a direct cell-by-cell reduction over the two masks:

```
overlap_fraction[source_region, target_zone] =
    sum_over_cells( source_mask[source_region, cell] * target_mask[target_zone, cell] )
    / sum_over_cells( source_mask[source_region, cell] )
```

i.e. "what fraction of `source_region`'s area falls inside `target_zone`."
Compute this once (pure geometry, no time dimension) as a weight matrix,
then redistribute any (region, month) capacity panel to zones via
`capacity_by_zone = capacity_by_region @ overlap_fraction`. This implicitly
assumes capacity is spread **uniformly by area** within each source region —
a real approximation (wind turbines cluster along ridgelines/coasts rather
than sitting uniformly across a whole NUTS2 region), but consistent with
only having region-aggregated MaStR capacity to start from (see the MaStR
section below — raw unit-level coordinates are deliberately not used here).
For MaStR's offshore pseudo-regions (`DEZZ-NORDSEE`/`DEZZ-OSTSEE`), use the
`offshore_regions_file` hull polygon (rasterized onto the same grid) as the
source mask in place of NUTS2.

**For point data** (only relevant if ever working from raw MaStR unit
coordinates instead of the region-aggregated panels): find the nearest grid
cell to the point, then take the argmax fraction **restricted to the
target country's own zones** (ignore a neighboring country's zone even if
it nominally wins that boundary cell — the point's country is already
known). Fall back to the nearest zone by centroid distance for the rare
cell with zero domestic-zone coverage (only occurs right at a border, where
the cell is entirely claimed by the neighboring country's zone).

### What's already available in this project (delu-headline-forecast)

- `data/downloads/pecd/peon_region_mask.nc`, `peof_region_mask.nc` — the two
  masks, downloaded by `pipeline/08_download_pecd_masks.py`.
- `data/processed/pecd_wind_zones.geojson` — derived vector polygons per
  zone (current version uses the simple per-zone >50% threshold, which has
  the boundary-gap artifact described above — switch to argmax if building
  on this elsewhere). From `pipeline/09_pecd_zones_eda.py`.
- The NUTS2 mask (`nuts_2_region_mask`) was fetched only as a one-off
  diagnostic (confirming grid alignment with PEON/PEOF) and isn't persisted
  here — re-download it wherever the NUTS2↔PEON/PEOF crosswalk is actually
  built (e.g. in `mastr-power-capacities-germany`).

## MaStR (Marktstammdatenregister) — installed capacity

The Bundesnetzagentur's public register of every power/gas unit in Germany.
Used here purely as an **installed-capacity weighting signal** — not
re-derived from raw MaStR in this project.

**Reference implementation / data provider:**
`~/research/mastr-power-capacities-germany`. This project already builds and
maintains the exact panel needed; **consume its processed parquet outputs
directly** rather than re-implementing the MaStR download/region-assignment
pipeline. Re-run `dvc repro` there if the data needs refreshing.

### Files to consume

- **`data/processed/capacity_by_nuts2_month.parquet`** — monthly
  (end-of-month), by NUTS2 region × `series` × month, from 2015-01. `series`
  values: `solar_full_feed_in`, `solar_self_consumption_no_storage`,
  `solar_self_consumption_with_storage`, `solar_unknown`, `wind_onshore`.
  Columns: `region_code` (NUTS2), `series`, `month`, `capacity_mw`,
  `unit_count`. This is the primary weighting input for PV and onshore wind.
  **Storage/battery capacity is excluded from this NUTS2 rollup** — it only
  exists at NUTS3 in `capacity_by_region_month.parquet`; aggregate to NUTS2
  yourself (sum by first 4 chars of the NUTS3 code, same convention the
  source project uses) if storage capacity is wanted as a feature.
- **`data/processed/capacity_by_offshore_month.parquet`** — monthly
  offshore wind capacity by the two synthetic pseudo-regions
  `DEZZ-NORDSEE` (North Sea) and `DEZZ-OSTSEE` (Baltic Sea), since offshore
  turbines have no NUTS region. These pseudo-regions are **not** official
  NUTS/Eurostat codes — any join against PECD offshore data will need a
  custom North Sea/Baltic Sea mapping, not a code-based join.

### PV behind-the-meter categories

Each solar unit's `pv_category` (used in the `series` column above) comes
from MaStR's `feed_in_type` field plus a location-based join against
co-located storage units:

- `full_feed_in` — 100% of output goes to the grid (mostly commercial/utility
  scale, ~61 kW average).
- `self_consumption_with_storage` — partial feed-in, battery present at the
  same location (residential/commercial PV+battery).
- `self_consumption_no_storage` — partial feed-in, no co-located storage.
- `unknown` — `feed_in_type` missing (~0.3% of solar capacity).

This matters because **SMARD's reported PV generation may not equal total
PV production** — self-consumed output behind the meter is not necessarily
fully visible in grid-metered generation figures the same way full-feed-in
output is. This is exactly the mismatch the phase-2 feature engineering
(per-category capacity aggregates, see CLAUDE.md) is meant to probe.

## Cross-source alignment checklist

Before joining these three sources, verify:

1. **NUTS2 code vintage** — MaStR's `region_code` is derived from a
   LAU→NUTS3 correspondence (Eurostat/GISCO) and truncated to NUTS2 (first 4
   characters); confirm PECD v4.2's NUTS2 codes use the same NUTS
   classification vintage (e.g. NUTS 2021 vs. 2024) so the codes actually
   match string-for-string.
2. **Offshore wind** — no shared code system between MaStR's two pseudo-
   regions and whatever PECD v4.2 offshore aggregation is used; needs a
   manual North Sea/Baltic Sea mapping (see PECD section above).
3. **DE-LU vs. DE** — SMARD targets are DE-LU market-area series; PECD and
   MaStR are Germany-only (no Luxembourg). Luxembourg's generation/load
   contribution to the DE-LU aggregate is presumably small but non-zero —
   decide whether to ignore it, or source Luxembourg capacity/weather
   separately, before treating the weather features as a complete
   explanation for a DE-LU target.
