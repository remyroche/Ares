"""Regression checks for the frozen-history daily failure-state loader."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_daily_observable_failure_state_screen.py"
)
SPEC = importlib.util.spec_from_file_location("daily_failure_state_screen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_observable_screen_excludes_resolved_label_columns() -> None:
    frame = pd.DataFrame({
        "__ts__": pd.date_range("2026-01-01", periods=5, tz="UTC"),
        "__symbol__": ["A/USD:USD"] * 5,
        "side_name": ["long"] * 5,
        "archetype_policy_key": ["long_default"] * 5,
        "observable_price_state": [0.0, 1.0, 2.0, 3.0, 4.0],
        "score_meta_base_soft_label": [0.1, 0.2, 0.3, 0.4, 0.5],
        "__u_policy_net__": [0.5, -0.1, 0.2, 0.1, 0.4],
        "__first_touch_mae_to_sl__": [0.2, 1.2, 0.4, 0.5, 0.1],
        "__path_full_bad_mae_1r__": [0, 1, 0, 0, 0],
        "__area_underwater_before_mfe_1r__": [0.0, 1.0, 0.2, 0.3, 0.1],
    })

    selected = MODULE._observable_columns(frame, maximum=32)

    assert "observable_price_state" in selected
    assert "score_meta_base_soft_label" in selected
    assert "__u_policy_net__" not in selected
    assert "__first_touch_mae_to_sl__" not in selected
    assert "__path_full_bad_mae_1r__" not in selected
    assert "__area_underwater_before_mfe_1r__" not in selected


def test_top10_equivalent_tail_is_side_local_and_observable() -> None:
    frame = pd.DataFrame({
        "__ts__": [pd.Timestamp("2026-01-01", tz="UTC")] * 6,
        "side_name": ["long"] * 3 + ["short"] * 3,
        "score_meta_base_soft_label": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    })

    selected = MODULE._top10_equivalent_tail(
        frame, score_column="score_meta_base_soft_label", fraction_of_top30=1.0 / 3.0
    )

    assert selected.groupby("side_name").size().to_dict() == {"long": 1, "short": 1}
    assert set(selected["score_meta_base_soft_label"]) == {0.9, 0.6}


def test_canonical_residual_event_context_is_eligible_as_observable_state() -> None:
    frame = pd.DataFrame({
        "__ts__": pd.date_range("2026-01-01", periods=8, tz="UTC"),
        "__symbol__": ["A/USD:USD"] * 8,
        "side_name": ["long"] * 8,
        "archetype_policy_key": ["long_default"] * 8,
        "resid_event_aegmm_local_support_log1p": [1.0] * 8,
        "resid_event_aegmm_gmm_entropy": list(range(8)),
        "resid_event_aegmm_expected_market_peer_surprise": [0.1 * i for i in range(8)],
        "resid_event_aegmm_expected_ev_timestamp_neutral_surprise": [-0.1 * i for i in range(8)],
    })

    selected = MODULE._observable_columns(frame, maximum=16)

    assert "resid_event_aegmm_gmm_entropy" in selected
    assert "resid_event_aegmm_expected_market_peer_surprise" in selected
    assert "resid_event_aegmm_expected_ev_timestamp_neutral_surprise" in selected
