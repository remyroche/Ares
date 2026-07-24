from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_meta_residual_event_balanced_error_overlay import (
    EVENT,
    RISK_PCT,
    SIDE_EVENT,
    SIDE_RISK_PCT,
    TARGET,
    _add_risk_variants,
    _apply_selected_overlays,
    _sample_weights,
    _targeted_temporal_features,
    _timestamp_training_frame,
)


def test_event_weights_give_each_chronological_block_equal_mass() -> None:
    days = pd.to_datetime(
        ["2025-01-01"] * 10 + ["2025-01-02"] * 2 + ["2025-01-10"] * 3,
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "day": days,
            EVENT: np.ones(len(days), dtype=np.int8),
            TARGET: np.ones(len(days), dtype=np.int8),
            "ev_after_1pct": np.full(len(days), -0.01, dtype=np.float32),
        }
    )

    weights = _sample_weights(frame)

    # Jan 1-2 is one contiguous block and Jan 10 is the second block.
    assert np.isclose(weights[:12].sum(), weights[12:].sum(), rtol=1e-5)


def test_timestamp_state_uses_observable_median_and_max_event_label() -> None:
    timestamps = pd.to_datetime(
        ["2025-01-01", "2025-01-01", "2025-01-01 01:00"],
        format="mixed",
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "day": timestamps.floor("D"),
            "observable": [1.0, 3.0, 5.0],
            SIDE_EVENT: [0, 1, 0],
            "ev_after_1pct": [0.01, -0.01, 0.02],
            "clean_exec": [1.0, 0.0, 1.0],
        }
    )

    states = _timestamp_training_frame(
        frame,
        ["observable"],
        target_column=SIDE_EVENT,
        event_column=SIDE_EVENT,
    )

    assert states.columns.tolist().count(SIDE_EVENT) == 1
    assert states.loc[0, "observable"] == 2.0
    assert states.loc[0, SIDE_EVENT] == 1


def test_selected_overlay_uses_accepted_risk_variant() -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["short", "short"],
            "archetype_policy_key": ["default", "default"],
            "parent_rank_v9": np.array([0.91, 0.91], dtype=np.float32),
            RISK_PCT: np.array([0.1, 0.1], dtype=np.float32),
            SIDE_RISK_PCT: np.array([0.95, 0.1], dtype=np.float32),
        }
    )
    frame = _add_risk_variants(frame)
    params = {
        ("short", "default"): {
            "risk_variant": "residual_risk_side",
            "threshold": 0.90,
            "mode": "hard_block",
            "alpha": 0.0,
        }
    }

    adjusted, flagged = _apply_selected_overlays(frame, params, "parent_rank_v9")

    assert flagged.tolist() == [True, False]
    assert adjusted[0] < 0.90
    assert adjusted[1] >= 0.90


def test_temporal_mechanism_families_are_local_to_the_intended_archetype() -> None:
    long_compression = _targeted_temporal_features(
        "long", "long_volcompression_wideslow_candidate"
    )
    short_default = _targeted_temporal_features("short", "short_default_clean_path")
    short_breakout = _targeted_temporal_features("short", "short_breakout_precision")

    assert "compression_quality_consistency" in long_compression
    assert "short_default_damage_integral_5d" not in long_compression
    assert "short_default_damage_integral_5d" in short_default
    assert "breakout_efficiency_4h" in short_breakout
    assert not _targeted_temporal_features("long", "long_mixed_wideslow_tentative")
