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

Follow `~/research/flexa-challenge`'s pattern once the first scenario has a
result to record (don't create these files preemptively/empty):

- **One pipeline script (+ notebook) per scenario**, named descriptively
  (e.g. `xx_pv_1d_baseline_lgbm.py`, `xx_pv_1d_pecd_capweighted.py`) —
  mirrors this template's `pipeline/01_*`/`02_*` numbering convention (see
  `contribution_conventions.md`).
- **A running "modeling learnings" markdown file at repo root**
  (`modeling_learnings.md`, analogous to flexa's `task_1_modeling_learnings.md`)
  logging every scenario tried: what changed, the resulting error per
  target/horizon, whether it's a genuine improvement or a diagnostic-only
  scenario (mark data-snooping/upper-bound scenarios explicitly invalid, as
  flexa's `lgbm_scaled_snoop_decomposed` did), and *why* it did or didn't
  help.
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
