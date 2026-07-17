"""Shared chart labels, colors, and styling for pipeline analysis scripts.

Colors are assigned by fixed categorical order (never cycled/re-picked per
chart) — see the project's dataviz conventions. Kept in one place so every
notebook that plots the same targets/models uses identical colors.
"""

import matplotlib.pyplot as plt

TARGET_LABELS = {
    "pv": "PV generation",
    "wind_onshore": "Wind onshore generation",
    "wind_offshore": "Wind offshore generation",
    "load": "Load",
}

# First four slots of the project's fixed categorical order.
TARGET_COLORS = {
    "pv": "#2a78d6",  # blue
    "wind_onshore": "#008300",  # green
    "wind_offshore": "#e87ba4",  # magenta
    "load": "#eda100",  # yellow
}

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}

# Slots 1 and 2 of the same fixed order — a different encoding (model
# identity), so reusing the first two slots here doesn't collide with
# TARGET_COLORS, which is never shown in the same chart.
MODEL_COLORS = {
    "linear_regression": "#2a78d6",  # blue
    "lightgbm": "#008300",  # green
}

ACTUAL_COLOR = "#0b0b0b"  # primary ink — "ground truth" is not a model series
GRIDLINE_COLOR = "#e1e0d9"

# The project's full fixed categorical order (8 slots) — assign in this
# order, never cycled/re-picked per chart.
FIXED_CATEGORICAL_ORDER = [
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Reserved slots for scenarios that already have an established color
# elsewhere (the per-scenario notebooks use MODEL_COLORS above) — keeping
# these in sync means the overview pages' colors agree with each
# individual scenario notebook's own color for its model.
KNOWN_SCENARIO_COLORS = {
    "baseline_linear_regression": FIXED_CATEGORICAL_ORDER[0],  # blue
    "baseline_lightgbm": FIXED_CATEGORICAL_ORDER[1],  # green
    "perfect_foresight_weather_lightgbm": FIXED_CATEGORICAL_ORDER[2],  # magenta
    "perfect_foresight_weather_lightgbm_quarterly_refit": FIXED_CATEGORICAL_ORDER[3],  # yellow
    "perfect_foresight_weather_xgboost_quarterly_refit": FIXED_CATEGORICAL_ORDER[4],  # aqua
    "perfect_foresight_weather_catboost_quarterly_refit": FIXED_CATEGORICAL_ORDER[5],  # orange
}

# One color per horizon, used within a single scenario's notebook to
# compare across lead times (a different encoding than TARGET_COLORS/
# MODEL_COLORS, so reusing the first three slots doesn't collide — horizon
# comparisons and target/model comparisons never appear in the same chart).
HORIZON_COLORS = {
    1: FIXED_CATEGORICAL_ORDER[0],  # blue
    7: FIXED_CATEGORICAL_ORDER[1],  # green
    14: FIXED_CATEGORICAL_ORDER[2],  # magenta
}


def assign_scenario_colors(scenario_ids: list[str]) -> dict[str, str]:
    """Map each scenario_id to a fixed-order categorical color, stable across runs.

    Known scenarios (`KNOWN_SCENARIO_COLORS`) keep their reserved slot, so a
    cross-scenario overview chart always agrees with each scenario's own
    notebook. Any new scenario_id is assigned the next unused slot, in
    sorted order for determinism.
    """
    colors = {sid: KNOWN_SCENARIO_COLORS[sid] for sid in scenario_ids if sid in KNOWN_SCENARIO_COLORS}
    used_slots = {FIXED_CATEGORICAL_ORDER.index(c) for c in colors.values()}

    unknown = sorted(sid for sid in scenario_ids if sid not in colors)
    available_slots = [i for i in range(len(FIXED_CATEGORICAL_ORDER)) if i not in used_slots]
    if len(unknown) > len(available_slots):
        raise ValueError("More scenarios than available categorical slots — extend the fixed order.")
    for sid, slot in zip(unknown, available_slots):
        colors[sid] = FIXED_CATEGORICAL_ORDER[slot]
    return colors


def style_axis(ax: plt.Axes) -> None:
    """Apply the project's recessive-gridline, spineless axis style."""
    ax.grid(axis="y", color=GRIDLINE_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
