"""Feature engineering helpers for headline forecasting models.

Shared across pipeline scenarios — see docs/modeling_plan.md for the
baseline feature set implemented here: daily/annual seasonality, short lags
(1h, 2h, 24h, 48h), and — load only — weekday dummies plus a weekday-matched
lag.

Two different notions of "causal" are at play, and it matters which
timestamp each feature is computed from:

- Lags are only ever computed relative to the forecast *origin* `t` (the
  most recent data we actually have) — never the target time, which hasn't
  happened yet.
- Calendar/seasonality features (hour-of-day, day-of-year, weekday) are
  known with certainty arbitrarily far in advance, so they must describe
  the *target* time `t + horizon`, not the origin. Getting this backwards
  is a real bug, not just an approximation: e.g. a 1-day-ahead forecast
  made on a Friday for Saturday must see `weekday=Saturday`, not `Friday`.
  Hour-of-day happens to be numerically identical either way here (every
  horizon is an exact multiple of 24h), but day-of-year and weekday are not.
"""

import numpy as np
import pandas as pd

LAG_HOURS = [1, 2, 24, 48]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def weekday_matched_lag_hours(horizon_hours: int) -> int:
    """Hours back to the most recent day sharing the *target's* weekday.

    Requires `horizon_hours` to be a whole number of days. Forecasting
    Saturday from a Friday origin (1d horizon) -> last Saturday is 6 days
    back, not 7. Forecasting Saturday from a week or two out (7d/14d
    horizons, both multiples of 7 days) -> the target's weekday always
    equals the origin's, and the *most recent* matching weekday is always
    last week (7 days back) either way, not two weeks back for the 14d case.
    """
    horizon_days = horizon_hours // 24
    days_back = (-horizon_days) % 7
    return (days_back or 7) * 24


def add_seasonality_features(df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Add daily- and annual-cycle sin/cos features describing `target_index`."""
    df = df.copy()
    hour = target_index.hour + target_index.minute / 60
    day_of_year = target_index.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, horizon_hours: int, is_load: bool) -> pd.DataFrame:
    """Add lag features computed from `target_col`, relative to each row's own (origin) timestamp.

    `is_load` adds the weekday-matched lag (see `weekday_matched_lag_hours`),
    named separately from the fixed-hour lags since its actual lag length
    varies by horizon.
    """
    df = df.copy()
    for lag in LAG_HOURS:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)
    if is_load:
        df["lag_weekday_match"] = df[target_col].shift(weekday_matched_lag_hours(horizon_hours))
    return df


def add_weekday_dummies(df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Add one-hot weekday dummies for `target_index` (Monday dropped as the reference level)."""
    df = df.copy()
    weekday = pd.Categorical(target_index.day_name(), categories=WEEKDAY_NAMES)
    dummies = pd.get_dummies(weekday, prefix="weekday", drop_first=True)
    dummies.index = df.index
    return pd.concat([df, dummies], axis=1)


def build_supervised_frame(
    panel: pd.DataFrame,
    target_col: str,
    horizon_hours: int,
    is_load: bool = False,
    extra_target_time_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the (features, target) frame for one (target, horizon) baseline model.

    Lags are computed strictly from information at or before each row's own
    timestamp `t`; calendar features describe `t + horizon_hours` (see
    module docstring); the target is `target_col` at `t + horizon_hours`.
    Every row is causally valid regardless of how large `horizon_hours` is —
    see docs/modeling_plan.md's phase-1 baseline description.

    `extra_target_time_features`, if given, is a timestamp-indexed frame
    (e.g. a perfect-foresight PECD capacity factor) looked up the same way
    as the calendar features — by `t + horizon_hours`, not `t` — since it
    describes a property of the target time, not information available at
    the forecast origin. See docs/modeling_plan.md's phase-2 description;
    a scenario using this is only valid in the "perfect foresight" sense,
    not as a real operational forecast.
    """
    assert horizon_hours % 24 == 0, "horizons must be a whole number of days for calendar-feature alignment"

    df = panel[[target_col]].copy()
    target_index = df.index + pd.Timedelta(hours=horizon_hours)

    df = add_seasonality_features(df, target_index)
    df = add_lag_features(df, target_col, horizon_hours, is_load=is_load)
    if is_load:
        df = add_weekday_dummies(df, target_index)
    if extra_target_time_features is not None:
        extra = extra_target_time_features.reindex(target_index)
        extra.index = df.index
        df = pd.concat([df, extra], axis=1)
    df["target"] = df[target_col].shift(-horizon_hours)

    feature_cols = [c for c in df.columns if c not in (target_col, "target")]
    df = df.dropna(subset=feature_cols + ["target"])
    return df[feature_cols], df["target"]
