from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from run_historical_exact_model_failure_ablation import (  # noqa: E402
    add_active_health_interactions,
    failure_window_groups,
    risk_tail_economics,
)


def test_failure_context_windows_are_indivisible_groups() -> None:
    timestamp = pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "source_utc": timestamp,
            "segment_id": 0,
            "target": [0] * 30 + [1] * 3 + [0] * 39,
            "event": [None] * 30 + ["failure"] * 3 + [None] * 39,
        }
    )
    groups = failure_window_groups(
        frame, target_column="target", event_column="event"
    )
    window = frame["source_utc"].between(timestamp[18], timestamp[44])
    assert len(set(groups[window])) == 1
    assert groups[30] == groups[32]


def test_active_health_interactions_are_explicit_products() -> None:
    frame = pd.DataFrame(
        {
            "active_transition_probability_oos": [0.5],
            **{column: [2.0] for column in (
                "health__mapped_net_std",
                "health__raw_mapped_rank_abs_gap",
                "health__low_map_support_share",
                "health__selected_symbol_hhi",
                "health__recent_resolved_net_ev_hl3d",
                "health__recent_resolved_hit_rate_hl3d",
                "health__recent_resolved_mapping_error_hl3d",
                "health__recent_resolved_full_stop_rate_hl3d",
            )},
        }
    )
    augmented, columns = add_active_health_interactions(frame)
    assert len(columns) == 8
    assert np.allclose(augmented[columns], 1.0)


def test_risk_tail_is_global_across_hours() -> None:
    frame = pd.DataFrame(
        {
            "source_utc": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "target": [0, 1, 0, 1],
            "post_12h_net_ev_mean": [0.1, -0.1, 0.1, -0.2],
            "post_minus_pre_mapping_residual": [0.0, -0.1, 0.0, -0.2],
            "active_transition_probability_oos": [0.0, 0.1, 0.0, 0.2],
        }
    )
    metrics = risk_tail_economics(
        frame,
        np.array([0.1, 0.8, 0.2, 0.9]),
        target_column="target",
        fraction=0.5,
    )
    assert metrics["risk_tail_rows"] == 2
    assert metrics["risk_tail_failure_rate"] == 1.0
    assert np.isclose(metrics["risk_tail_post_12h_net_bps"], -1500.0)
