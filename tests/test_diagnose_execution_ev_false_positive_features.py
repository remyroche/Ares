import numpy as np
import pandas as pd

from scripts.diagnose_execution_ev_false_positive_features import (
    TARGET,
    add_classes,
    add_decision_time_context,
    allowed_live_features,
    apply_frozen_screens,
    control_contrasts,
    freeze_screens,
)


def _frame(rows: int = 2400) -> pd.DataFrame:
    index = np.arange(rows)
    selected = index < 240
    high = (index < 120) | ((index >= 600) & (index < 840))
    side = np.where(index % 2 == 0, "long", "short")
    # `live_signal` deliberately separates selected TP from FP consistently
    # in both sides; target/action/calendar fields must not enter candidates.
    return pd.DataFrame(
        {
            "__ts__": pd.Timestamp("2026-06-01T00:00:00Z") + pd.to_timedelta(index, unit="h"),
            "__symbol__": ["BTC/USD:USD"] * rows,
            "side_name": side,
            "candidate_id": [f"c{i}" for i in index],
            "direct_mapped_score": np.where(selected, 1000 - index, -index).astype(float),
            "capture_mapped_score": np.where(selected, 900 - index, -2.0 * index).astype(float),
            "live_signal": np.where(index < 120, 2.0, np.where(selected, -2.0, 0.0)),
            "execution_exit_reason": ["timeout"] * rows,
            "hour_sin": np.sin(index),
            TARGET: np.where(high, 0.010, -0.010),
        }
    )


def test_allowed_features_exclude_outcomes_exit_and_calendar_shortcuts() -> None:
    frame = add_decision_time_context(_frame())
    columns = allowed_live_features(frame)
    assert "live_signal" in columns
    assert TARGET not in columns
    assert "execution_exit_reason" not in columns
    assert "hour_sin" not in columns


def test_control_only_freeze_has_side_stable_field_and_applies_without_refit() -> None:
    control = add_classes(add_decision_time_context(_frame()))
    contrasts = control_contrasts(control, ["live_signal"])
    screens = freeze_screens(control, contrasts)
    assert screens["feature"].tolist() == ["live_signal"]
    assert screens["direction_tp_over_fp"].iloc[0] == 1.0

    later = add_classes(add_decision_time_context(_frame()))
    scored = apply_frozen_screens(later, screens)
    assert "frozen_equal_weight_composite" in scored
    assert scored["frozen_component_pass_count"].max() == 1
