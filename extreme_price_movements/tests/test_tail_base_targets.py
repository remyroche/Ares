import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.tail_base_targets import (
    TailBaseTargetError,
    build_tail_base_targets,
    grade_atr_normalized_net,
    grade_exact_net_bps,
    grade_first_touch_tbm,
)


def test_exact_net_grades_are_inclusive_at_declared_upper_bounds():
    net = np.array([-51.0, -50.0, 50.0, 150.0, 250.0, 350.0, 350.01, np.nan])
    valid = np.array([True, True, True, True, True, True, True, False])
    assert grade_exact_net_bps(net, valid).tolist() == [0, 0, 1, 2, 3, 4, 5, -1]


def test_atr_grades_normalise_exact_net_with_decision_time_atr():
    grades, z = grade_atr_normalized_net(
        np.array([-101.0, -100.0, 0.0, 100.0, 200.0, 300.0, 301.0]),
        np.full(7, 100.0),
        np.ones(7, dtype=bool),
    )
    assert grades.tolist() == [0, 0, 1, 2, 3, 4, 5]
    assert np.allclose(z, [-1.01, -1.0, 0.0, 1.0, 2.0, 3.0, 3.01])


def test_tbm_uses_nested_first_touch_contract_and_adverse_tie_break():
    # severe adverse, strong favourable, moderate adverse, moderate favourable,
    # timeout, and equal-time adverse outer-contract tie.
    grades = grade_first_touch_tbm(
        first_tp4_minute=[4, 2, 8, 2, -1, 3],
        first_tp6_minute=[-1, 5, -1, -1, -1, 3],
        first_sl4_minute=[2, -1, 4, 7, -1, 3],
        first_sl6_minute=[5, -1, -1, -1, -1, 3],
        label_valid=[True] * 6,
    )
    assert grades.tolist() == [0, 4, 1, 3, 2, 0]


def test_tbm_rejects_broken_path_nesting_and_invalid_minutes():
    with pytest.raises(TailBaseTargetError, match="requires its"):
        grade_first_touch_tbm(
            first_tp4_minute=[-1], first_tp6_minute=[5],
            first_sl4_minute=[-1], first_sl6_minute=[-1], label_valid=[True],
        )
    with pytest.raises(TailBaseTargetError, match="integer -1"):
        grade_first_touch_tbm(
            first_tp4_minute=[1.5], first_tp6_minute=[-1],
            first_sl4_minute=[-1], first_sl6_minute=[-1], label_valid=[True],
        )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "__decision_ts__": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"]
            ),
            "__symbol__": ["BTC", "ETH", "SOL"],
            "side_name": ["long", "long", "short"],
            "label_valid": [True, True, False],
            "exact_net_bps": [60.0, -200.0, np.nan],
            "atr_bps": [50.0, 100.0, np.nan],
            "first_tp4_minute": [2, -1, -1],
            "first_tp6_minute": [-1, -1, -1],
            "first_sl4_minute": [8, 3, -1],
            "first_sl6_minute": [-1, 5, -1],
        }
    )


def test_frame_builder_preserves_identity_and_never_encodes_invalid_as_loss():
    out = build_tail_base_targets(_frame())
    assert out.columns.tolist() == [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "tail_target_valid", "tail_target_net_grade_0_5",
        "tail_target_atr_grade_0_5", "tail_target_atr_z",
        "tail_target_tbm_grade_0_4",
    ]
    assert out.tail_target_net_grade_0_5.tolist() == [2, 0, -1]
    assert out.tail_target_atr_grade_0_5.tolist() == [3, 0, -1]
    assert out.tail_target_tbm_grade_0_4.tolist() == [3, 0, -1]
    assert np.isnan(out.tail_target_atr_z.iloc[2])


def test_frame_builder_rejects_duplicate_identity_and_invalid_valid_row_economics():
    duplicate = _frame()
    duplicate.loc[1, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]] = duplicate.loc[0, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]]
    with pytest.raises(TailBaseTargetError, match="unique"):
        build_tail_base_targets(duplicate)
    bad = _frame()
    bad.loc[0, "atr_bps"] = 0.0
    with pytest.raises(TailBaseTargetError, match="positive finite atr_bps"):
        build_tail_base_targets(bad)
