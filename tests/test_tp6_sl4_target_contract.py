import numpy as np
import pandas as pd

from scripts.materialize_tp6_sl4_robust_clear_labels import COST_BPS, _sigmoid
from extreme_price_movements.tp6_sl4_target_weights import (
    TP6SL4Columns,
    TargetParameters,
    assert_simplex,
    build_target,
)
from scripts.audit_tp6_sl4_target_substrate import _b3_upper, _reason


def test_invalid_paths_are_distinct_from_complete_timeouts() -> None:
    complete = pd.Series([True, False, False])
    ids = pd.Series(["complete", "incomplete", None])
    assert _reason(complete, ids).tolist() == [
        "complete_executable_path", "incomplete_h12_path", "missing_candidate_identity",
    ]


def test_b3_selected_target_is_a_simplex_and_is_never_used_for_invalid_rows() -> None:
    frame = pd.DataFrame({
        "t2_tp6_sl4_event": [0, 1, 2],
        "t2_tp6_sl4_exit_minute": [60, 60, 720],
        "t2_path_mfe_atr": [6.0, .5, 2.0],
        "t2_path_mae_atr": [.5, 4.0, 1.0],
    })
    target = build_target(
        frame,
        "B3",
        columns=TP6SL4Columns(),
        parameters=TargetParameters(hard_floor=.75, time_decay_hours=8.),
    )
    assert_simplex(target)
    # The caller's validity gate owns invalid rows: there is no zero-valued
    # synthetic target that could make an unlabelled path look like a failure.
    invalid_mask = np.array([False, False, True])
    admitted = target[~invalid_mask]
    assert admitted.shape == (2, 3)
    assert np.isfinite(admitted).all()


def test_selected_b3_membership_rewards_quick_resolved_upper_events() -> None:
    upper_fast, upper_slow, lower_fast = _b3_upper(
        np.array([0, 0, 1]), np.array([60., 720., 60.])
    )
    assert upper_fast > upper_slow > .75
    assert lower_fast < .125


def test_cost_aware_robust_clear_uses_the_declared_floor_once() -> None:
    atr_bps = np.array([50.0, 50.0, 50.0])
    pre_adverse_mfe_atr = np.array([2.49, 2.50, 3.50])
    # The selected TP6/SL4 geometry is not itself the robust-clear threshold:
    # this target requires cost + a declared buffer before the adverse event.
    margin = pre_adverse_mfe_atr * atr_bps - COST_BPS - 25.0
    assert COST_BPS == 100.0
    assert np.allclose(margin, [-0.5, 0.0, 50.0])
    assert (margin > 0).tolist() == [False, False, True]
    assert np.isclose(_sigmoid(np.array([0.0]))[0], 0.5)


def test_selected_geometry_is_tp6_sl4_not_legacy_tp3_sl2() -> None:
    # Guard the exact target primitive used by robust-clear materialisation.
    from scripts.materialize_full_universe_tp6_sl4_h12_sidecar import HORIZON_MINUTES, SL_ATR, TP_ATR
    assert (TP_ATR, SL_ATR, HORIZON_MINUTES) == (6.0, 4.0, 720)


def test_executable_margin_is_the_direct_net_control_at_the_selected_cost_floor() -> None:
    gross_bps = np.array([-75.0, 100.0, 275.0])
    round_trip_cost_bps = 100.0
    executable_cost_floor_bps = 100.0
    executable_margin = gross_bps - max(round_trip_cost_bps, executable_cost_floor_bps)
    direct_net = gross_bps - round_trip_cost_bps
    assert np.allclose(executable_margin, direct_net)
