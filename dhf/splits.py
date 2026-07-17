"""Chronological train/validation/test split shared by every forecasting scenario.

See docs/modeling_plan.md: a fixed-cutoff split where the last `TEST_DAYS` is
reserved and not evaluated during model development; the `VAL_DAYS` before
that is used to compare model choices; everything earlier is training data.
Kept in one place so every scenario (and the EDA notebook's availability
chart) uses identical cutoffs.
"""

import pandas as pd

TEST_DAYS = 365
VAL_DAYS = 730


def time_split(
    index: pd.DatetimeIndex, test_days: int = TEST_DAYS, val_days: int = VAL_DAYS
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (val_start, test_start) cutoffs for a chronological split ending at `index.max()`."""
    end = index.max()
    test_start = end - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    return val_start, test_start


# Fixed 3-week example windows for actual-vs-predicted charts, chosen inside
# the validation period (2023-07-18 .. 2025-07-17) with enough margin before
# test_start that every horizon's target time (window end + up to 14 days)
# still stays out of the reserved test period.
EXAMPLE_WINDOWS = {
    "summer": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-21")),
    "winter": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-21")),
}
