from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.path_economic_labels import (
    PATH_ECONOMIC_LABEL_COLUMNS,
    materialize_path_economic_labels,
)
from extreme_price_movements.residual_state_discovery import (
    ReliabilityEventConfig,
    build_daily_reliability_cells,
    detect_unreliability_events,
)


def _taxonomy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ev_after_1pct": [-0.03, -0.01, -0.004, -0.006, -0.002, 0.02, 0.01, np.nan],
            "clean_exec": [0, 0, 1, 0, 0, 1, 0, 0],
            "dirty_positive": [1, 0, 0, 1, 0, 0, 1, 0],
            "first_touch_bad_mae_1r": [1, 0, 0, 0, 0, 0, 0, 0],
            "full_path_bad_mae_1r": [1, 0, 0, 0, 0, 0, 1, 0],
            "timeout": [1, 1, 0, 0, 0, 0, 0, 0],
        }
    )


def test_path_economic_labels_are_exclusive_and_preserve_failure_mechanism() -> None:
    labels = materialize_path_economic_labels(_taxonomy_frame())
    assert labels.loc[0, "path_economic_state"] == "acute_adverse"
    assert labels.loc[1, "path_economic_state"] == "slow_timeout_loss"
    assert labels.loc[2, "path_economic_state"] == "clean_negative_ev"
    assert labels.loc[3, "path_economic_state"] == "dirty_negative_ev"
    assert labels.loc[4, "path_economic_state"] == "other_negative_ev"
    assert labels.loc[5, "path_economic_state"] == "durable_clean_positive"
    assert labels.loc[6, "path_economic_state"] == "other_positive"
    assert labels.loc[7, "path_economic_state"] == "unavailable"
    assert labels.loc[:6, list(PATH_ECONOMIC_LABEL_COLUMNS)].sum(axis=1).eq(1.0).all()
    assert labels.loc[7, list(PATH_ECONOMIC_LABEL_COLUMNS)].sum() == 0.0
    assert all(labels[name].dtype == np.float32 for name in PATH_ECONOMIC_LABEL_COLUMNS)


def test_full_path_roughness_does_not_turn_a_positive_trade_into_acute_adversity() -> (
    None
):
    frame = _taxonomy_frame().iloc[[6]].copy()
    labels = materialize_path_economic_labels(frame)
    assert labels.iloc[0]["path_economic_state"] == "other_positive"


def test_daily_reliability_cells_and_events_expose_path_mechanisms() -> None:
    rows = []
    for day in pd.date_range("2025-01-01", periods=28, freq="D", tz="UTC"):
        for idx in range(12):
            adverse = day >= pd.Timestamp("2025-01-25", tz="UTC")
            rows.append(
                {
                    "__ts__": day + pd.Timedelta(minutes=15 * idx),
                    "__symbol__": f"S{idx}",
                    "side_name": "long",
                    "archetype_policy_key": "long_mixed",
                    "hit_probability": 0.70,
                    "clean_exec": 0.0 if adverse else 1.0,
                    "dirty_positive": float(adverse),
                    "full_path_bad_mae_1r": float(adverse),
                    "timeout": 0.0,
                    "ev_after_1pct": -0.03 if adverse else 0.01,
                }
            )
    cells = build_daily_reliability_cells(
        pd.DataFrame(rows), ReliabilityEventConfig(causal_min_days=10)
    )
    assert {
        "acute_adverse_rate",
        "slow_timeout_loss_rate",
        "clean_negative_ev_rate",
    }.issubset(cells.columns)
    assert (
        cells.loc[
            cells["day"].ge(pd.Timestamp("2025-01-25", tz="UTC")), "acute_adverse_rate"
        ]
        .eq(1.0)
        .all()
    )
    events, _ = detect_unreliability_events(
        cells, ReliabilityEventConfig(causal_min_days=10)
    )
    assert "state_failure_mechanism" in events.columns
    assert (events["state_failure_mechanism"] == "acute_adverse_path").any()
