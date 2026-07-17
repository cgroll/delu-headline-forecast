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
**v4.1** dataset ID `sis-energy-derived-reanalysis`, at NUTS0 (country) and
NUTS1 (German Bundesländer) resolution. Use it as a *pattern* (ZIP download
via `cdsapi`, metadata-header CSV parsing, wide-format parquet with
MultiIndex columns) — not as a direct dependency — since this project needs:

- **Dataset ID `sis-energy-pecd`** (v4.2), not `sis-energy-derived-reanalysis`.
- **NUTS2 spatial aggregation** (`sub_country_level`, `nuts_level: "2"`),
  not NUTS1 or country level — the resolution MaStR capacity data is
  available at.
- Variables: `solar_photovoltaic_power_generation`,
  `wind_power_generation_onshore`, `wind_power_generation_offshore`;
  product type `capacity_factor_ratio`.

**Known gotcha (offshore wind spatial aggregation):** per the CDS v4.2
documentation, the NUTS0 aggregation level for wind indicators (onshore and
offshore) has been removed from the download form due to methodological
inconsistencies, and v4.2 introduces two *new* aggregation schemes — P2ON
(Pan-European Onshore Zones) and P2OF (Pan-European Offshore Zones) — built
for ENTSO-E's ERAA 2026 adequacy assessment. Offshore wind has no NUTS2
concept (it's not part of any land NUTS region), so check what spatial units
v4.2 actually offers for offshore wind capacity factors (likely
`maritime_country_level`, as v4.1 used, and/or the new PEOF zones) before
assuming a clean join to MaStR's offshore pseudo-regions — see the MaStR
section below and the open question in CLAUDE.md.

**Requires** a `~/.cdsapirc` file with a valid CDS API key (same requirement
`world-of-energy`'s scripts have).

**Docs:**
[CDS dataset page](https://cds.climate.copernicus.eu/datasets/sis-energy-pecd) ·
[v4.2 product user guide](https://confluence.ecmwf.int/pages/viewpage.action?pageId=540941752) ·
[C3S forum: v4.2 release](https://forum.ecmwf.int/t/new-pecd-version-4-2-co-developed-with-entso-e-now-available-on-the-cds/13497) ·
[C3S forum: v4.1 deprecation](https://forum.ecmwf.int/t/updates-to-the-climate-and-energy-related-variables-from-the-pan-european-climate-database-pecd-v4-2-and-deprecation-of-v4-1/14348)

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
