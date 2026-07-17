# Modeling plan

Detail doc for [CLAUDE.md](../CLAUDE.md). Covers the forecasting matrix, the
iteration strategy, feature-engineering roadmap, and the bookkeeping
convention to follow as scenarios are implemented.

## The forecasting matrix

4 targets × 7 horizons = **28 independent forecasting pipelines**:

| | 1d | 3d | 7d | 14d | 21d | 28d | 35d |
|---|---|---|---|---|---|---|---|
| PV generation | | | | | | | |
| Wind onshore generation | | | | | | | |
| Wind offshore generation | | | | | | | |
| Load | | | | | | | |

**Decision (confirmed with the user): one independently trained/tuned model
per cell**, not a single model with horizon as a feature. Rationale
discussed: at 1 day the dominant signal is autoregressive/diurnal; at 35 days
autoregressive skill has mostly decayed and the relevant signal (if any) is
closer to climatology plus whatever the capacity-weighted weather features
can offer — these are different enough problems that sharing one model's
hyperparameters/features across all 7 horizons risks compromising both ends.
This means 28x the bookkeeping overhead of a shared-model approach — lean
into per-scenario tracking (see "Bookkeeping" below) from the start rather
than trying to hold 28 results in your head.

Given the size of this matrix, **don't build all 28 at once**. Suggested
order: pick one target (e.g. PV — clearest physical link to a weather
capacity factor) and implement all 7 horizons for it first, phase 1 then
phase 2, to validate the whole pipeline shape before replicating across the
other 3 targets.

## Phase 1 — baseline (no weather)

Per (target, horizon) cell, establish a reference error using only
information available at forecast time with no exogenous weather input:

- Naive baselines first (seasonal persistence — e.g. "same hour, N days ago
  scaled by trend", climatological mean/quantile for that hour-of-day ×
  day-of-year) — cheap, and per `flexa-challenge`'s experience, often a
  surprisingly strong floor, especially at short horizons.
- Then a regular ML/statistical forecast model (e.g. gradient boosting or a
  seasonal statistical model) using lags respecting the horizon (a 35-day
  model cannot use a 2-day lag), calendar/seasonality features, and MaStR
  installed-capacity trend (capacity growth alone shifts the achievable
  generation level over time, independent of weather).

This phase's numbers are the denominator for every "did weather help"
comparison in phase 2 — get them recorded before moving on.

## Phase 2 — perfect-foresight weather value check

Same models, extended with:

- **PECD v4.2 capacity factors** (NUTS2, hourly) for the matching technology,
  aggregated to a national DE-LU figure — assumed perfectly known at the
  forecast horizon (this is the "perfect foresight" simplification; real
  forecast uncertainty is deliberately ignored until phase 3).
- **MaStR installed capacity** (NUTS2, monthly), both as the weighting input
  for the capacity-factor aggregation and as its own feature (capacity
  growth trajectory).

### Capacity-factor aggregation — itself an open experiment

Per the user's guidance, this is **not fully decided** — build it
incrementally and compare versions as separate scenarios, cheapest first:

1. **Start simple**: total installed capacity per technology (all MaStR
   `pv_category` values summed) as the weight —
   `effective_cf(t) = sum_region(cf_region(t) * capacity_region) / sum_region(capacity_region)`.
2. **Add category-level aggregates as extra features**: national installed
   capacity split by `pv_category` (full feed-in / self-consumption with
   storage / self-consumption without storage) as additional inputs
   alongside the simple effective CF from step 1 — testing whether the
   *mix* of behind-the-meter categories explains residual error (e.g.
   because SMARD's metered generation doesn't fully capture self-consumed
   output, see `docs/data_sources.md`).
3. **Only then**, if step 2 shows the category split matters, consider
   folding it into the capacity-factor weighting itself (e.g. a
   category-specific derating before aggregating) rather than treating it as
   a flat additive feature.

For load, capacity factors aren't directly relevant the same way — instead
consider PECD's driving weather variables directly (temperature is the
classic load driver; also see `world-of-energy`'s
`38_bdew_profiles.py`/`39_demand_reconstruction.py` for a prior attempt at
demand reconstruction from weather + standard load profiles).

## Phase 3 — real forecasts (not started)

Only pursue this if phase 2 shows a clear, horizon-consistent improvement
over phase 1. Replace perfect-foresight PECD capacity factors with:

- Actual weather **forecasts** at the relevant lead time (source TBD — ECMWF
  extended-range/S2S products are the natural fit for horizons out to 35
  days; day-ahead-only NWP products won't cover the longer horizons).
- Our **own capacity-factor computation** from raw forecast weather
  variables (temperature, wind speed, irradiance) — see
  `flexa-challenge`'s `pv_power_proxy_flow` (pvlib Faiman/PVWatts chain) and
  `world-of-energy`'s `33/34_*_renewables_ninja_generations.py` for two
  different prior approaches to this exact problem.

Not scoped further here — revisit once phase 2 has a result.

## Bookkeeping convention

Adapted from `~/research/flexa-challenge`'s pattern (per-scenario JSON metric
files + a hardcoded-list Snakemake aggregation stage) to this project's DVC
setup — implemented starting with the two baseline scenarios
(`04_baseline_linear_regression`, `05_baseline_lightgbm`):

- **One pipeline script (+ notebook) per scenario**, named descriptively
  (e.g. `04_baseline_linear_regression.py`, later something like
  `xx_pv_pecd_capweighted.py`). Every scenario notebook follows the same
  internal structure — features used, MAE/RMSE table, actual-vs-predicted
  (a summer and a winter window), full-period rolling error, feature
  importance — so scenarios are easy to compare section-by-section; the
  shared computation/plotting code lives in `dhf/reporting.py`, not
  duplicated per notebook.
- **Each scenario writes one metrics file** to
  `data/processed/results/<scenario_id>.parquet` via
  `dhf.scenarios.save_scenario_metrics()` — one row per (target, horizon)
  with `mae`, `rmse`, plus scenario metadata (`title`, `description`,
  `notebook` link, `valid`). `valid=False` marks a scenario that uses
  information that wouldn't really be available at forecast time (e.g.
  perfect-foresight PECD capacity factors in phase 2, or an unvalidated
  real forecast in phase 3) — every scenario so far is `valid=True`, since
  lagged/calendar features are all genuinely available at forecast time
  (modulo the publication-lag caveat noted above, deliberately ignored for
  now).
- **Two overview pages read every scenario's metrics file back** — no
  numbers are recomputed, only reshaped/plotted:
  - `06_scenario_overview_tables.py` — a pure script (not jupytext; writes
    the notebook directly via `nbformat`, all-markdown, mirroring flexa's
    `09_task1_overview.py`) producing one GFM comparison table per headline
    variable: one row per scenario, MAE/RMSE per horizon, validity, and a
    link to that scenario's own notebook.
  - `07_scenario_overview_lead_times.py` — line charts (MAE and RMSE vs.
    horizon, one line per scenario, one panel per target), colors assigned
    via `dhf.viz.assign_scenario_colors()` so a scenario's color is stable
    across every chart it appears in.
  - Both glob `data/processed/results/*.parquet` (`dhf.scenarios.load_all_scenario_metrics`)
    rather than hardcoding a scenario list — adding scenario \#3 only means
    adding its pipeline script + a `dvc.yaml` dependency edge into these two
    stages, no code change in the overview pages themselves.
- **A running "modeling learnings" markdown file at repo root**
  (`modeling_learnings.md`, analogous to flexa's `task_1_modeling_learnings.md`)
  — not yet started; add once a scenario produces a result worth narrating
  beyond what the tables/charts already show (what changed, why it did or
  didn't help).
- **`TODO.md` at repo root** for open follow-ups not yet implemented, so
  ideas aren't lost between sessions.
- **Book narrative pages** (`book/markdown/`) summarizing the modeling
  journey once there's enough of one to tell — see
  `flexa-challenge/book/markdown/task1_modeling_journey.md` and
  `task1_final_model.md` for the target shape (journey = chronological
  levers-and-traps narrative; final-model page = complete end-state
  reference, decoupled from how it was arrived at).

## Still open (flagged in CLAUDE.md, repeated here for visibility)

- **Evaluation metric**: MAE (flexa's choice) vs. RMSE vs. a
  capacity-normalized error (e.g. MAE / installed capacity, which would make
  errors comparable across a growing capacity base and across technologies)
  — worth deciding before phase 1 baselines are scored, since it affects
  every subsequent comparison.
- **Train/test split / walk-forward scheme**: flexa used a weekly expanding-
  window walk-forward over a ~5-month test period. With horizons up to 35
  days, the walk-forward fold size and test-period length need rethinking —
  a 35-day-horizon model needs a test period long enough to contain several
  independent 35-day forecasts to be evaluable at all.
- **DE-LU vs. DE-only weather/capacity coverage** (Luxembourg gap — see
  `docs/data_sources.md`).
- **Publication lag on the short lags is currently ignored.** The baseline
  assumes e.g. a forecast issued at 14:00 tomorrow can use today's 14:00
  observation (a clean 24h lag) — in reality SMARD generation/load figures
  for "yesterday" are likely published in a daily batch, not continuously,
  so for some hours the 24h lag would actually only be a few hours old by
  publication time rather than a full day. Deliberately ignored for now: it
  only meaningfully affects the 1-day horizon, and the project's real
  interest is the 7-35 day horizons where a few hours of slop on a 24-48h
  lag is negligible. Revisit before ever treating a 1-day-horizon model as
  operational.
