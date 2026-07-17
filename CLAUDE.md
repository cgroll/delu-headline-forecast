# delu-headline-forecast

## Goal

Test whether weather data — specifically PECD v4.2 renewable capacity factors
— has explanatory power for Germany/Luxembourg (DE-LU) bidding-zone headline
variables, at forecast horizons far beyond the usual day-ahead window:

- **PV generation**
- **Wind onshore generation**
- **Wind offshore generation**
- **Load**

Horizons: **1, 3, 7, 14, 21, 28, 35 days** ahead.

## The three-phase plan

1. **Baseline forecasts.** For each (target, horizon) pair, build a regular
   forecast model using only autoregressive/calendar information (no
   weather). This is the reference every later scenario is measured against.
2. **Perfect-foresight weather value check.** Extend the same models with
   PECD v4.2 capacity factors — assumed known with **perfect foresight** —
   and MaStR installed-capacity data (NUTS2). If this doesn't beat phase 1 by
   a meaningful margin, real weather forecasts won't help either, and the
   project stops here having learned that.
3. **Real forecasts (future work, not started).** Only if phase 2 shows
   value: replace the perfect-foresight PECD capacity factors with actual
   weather *forecasts* and our own capacity-factor computation (e.g.
   pvlib/wind power-curve models, as prototyped in `flexa-challenge` and
   `world-of-energy`). Out of scope until phase 2 has a clear result.

This mirrors the "does the fancy input even help, before we pay for the hard
version of it" structure of `~/research/flexa-challenge`'s Task 1.

## Project structure & conventions

This project was created from `project-book-template`. **Read
[contribution_conventions.md](contribution_conventions.md) first** — it
defines the pipeline/DVC/MyST-book structure, path conventions
(`pkg/paths.py` → `ProjPaths`), and git conventions used throughout. This
file assumes you've read it and only adds project-specific context.

**Setup is not finished.** `pkg/` has not yet been renamed via
`init_project.py` (still contains the template placeholder `example_*`
files/paths) and `pyproject.toml` still says `name = "my-project"`. Before
writing real pipeline code, run `python init_project.py` (interactive —
prompts for a package abbreviation, e.g. `dhf`, and auto-commits) or ask the
user how they'd like to handle it.

## Data sources — see [docs/data_sources.md](docs/data_sources.md) for full detail

| Source | What we get | Reference implementation |
|---|---|---|
| **SMARD** (Bundesnetzagentur) | Hourly DE-LU generation (solar, wind onshore/offshore) and load — the forecast targets. Also day-ahead/intraday forecasts (useful as a benchmark) and monthly installed capacities. | `~/research/world-of-energy` (`woe/smard/`, `pipeline/01_download_smard_price_analysis_data.py`, `pipeline/06_download_smard_forecasts.py`) |
| **PECD v4.2** (Copernicus CDS) | Hourly renewable capacity factors (solar PV, wind onshore, wind offshore) and driving weather variables, at NUTS2 resolution — the "perfect foresight" weather input for phase 2. | `~/research/world-of-energy` pipeline scripts 35/36/50/51 use **v4.1** (`sis-energy-derived-reanalysis`, country/NUTS1) — treat as a pattern to adapt, not reuse as-is. This project needs v4.2 (`sis-energy-pecd`) at NUTS2. |
| **MaStR** (Marktstammdatenregister) | Installed capacity by NUTS2 region, technology, and behind-the-meter category (full feed-in / self-consumption with or without storage) — used to weight PECD capacity factors up to a national figure and as its own feature. | `~/research/mastr-power-capacities-germany` — **consume its processed output directly** (`data/processed/capacity_by_nuts2_month.parquet`, `capacity_by_offshore_month.parquet`), don't re-derive from raw MaStR. |

## Modeling approach — see [docs/modeling_plan.md](docs/modeling_plan.md) for full detail

Iteration and bookkeeping style follows `~/research/flexa-challenge`: each
new idea is a separate, named scenario with its own pipeline script and
notebook; results (forecast error by target/horizon/scenario, what worked,
what didn't, and why) get logged in a running "modeling learnings" document
as the project progresses, with a `TODO.md` for open follow-ups. Don't
create these bookkeeping files preemptively — start them once the first
scenario has a result to record.

Key decisions made so far (see docs/modeling_plan.md for the reasoning):

- **One independent model per (target, horizon) pair** — 4 targets × 7
  horizons = 28 forecasting pipelines — rather than a single model with
  horizon as a feature.
- **Capacity-factor aggregation is itself an open experiment**: start with a
  single capacity-weighted national effective capacity factor per
  technology, then compare against versions that also bring in the MaStR
  behind-the-meter split (feed-in-only vs. self-consumption vs. storage) —
  both in the national capacity aggregates used as extra features, and later
  in the weighting itself.

## Open questions / not yet decided

- Evaluation metric (MAE vs. RMSE vs. a capacity-normalized error) and the
  train/test split / walk-forward scheme.
- How to reconcile MaStR's offshore wind pseudo-regions (`DEZZ-NORDSEE`,
  `DEZZ-OSTSEE`) with whatever spatial units PECD v4.2 uses for offshore wind
  (NUTS0 is being removed for wind indicators in v4.2 — check the current
  offshore aggregation options, e.g. `maritime_country_level` or the new PEOF
  zones, before assuming a direct join works).
- Package abbreviation for `init_project.py` (see Setup above).
