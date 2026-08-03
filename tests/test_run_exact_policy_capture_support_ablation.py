from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from scripts.run_exact_policy_capture_support_ablation import (
    FROZEN_BASE_MARGIN_SCREEN,
    apply_final_veto,
    apply_quantile_veto,
    compose_distributional_net,
    compose_multitask_distributional_net,
    load_frozen_base_margin_interaction,
    margin_capture_soft_interaction,
    standardized_capture_blends,
    support_feature_matrix,
)


def test_support_feature_matrix_keeps_capture_only_nested() -> None:
    raw = {
        "direct_net": np.array([0.1, 0.2]),
        "capture_probability": np.array([0.7, 0.8]),
        "severe_loss_probability": np.array([0.2, 0.1]),
        "favorable_order_probability": np.array([0.6, 0.9]),
        "capture_ratio": np.array([0.3, 0.4]),
        "giveback_log_bps": np.array([2.0, 1.0]),
    }
    capture = support_feature_matrix(raw, full=False)
    full = support_feature_matrix(raw, full=True)
    assert capture.columns.tolist() == ["direct_net", "capture_probability"]
    assert full.shape == (2, 6)


def test_quantile_veto_only_demotes_rejected_rows() -> None:
    score = np.array([0.03, 0.02, 0.01])
    risk = np.array([0.1, 0.5, 0.9])
    low = apply_quantile_veto(
        score, risk, threshold=0.2, high_is_bad=False
    )
    high = apply_quantile_veto(
        score, risk, threshold=0.8, high_is_bad=True
    )
    assert low[0] < min(score)
    np.testing.assert_allclose(low[1:], score[1:])
    assert high[2] < min(score)
    np.testing.assert_allclose(high[:2], score[:2])


def test_final_veto_survives_flat_ev_mapping() -> None:
    mapped = np.array([0.01, 0.01, 0.01])
    result = apply_final_veto(mapped, np.array([False, True, False]))
    assert result[1] < result[0]
    assert result[1] < result[2]


def test_standardized_blends_are_nested_between_rank_sources() -> None:
    direct = np.array([-1.0, 0.0, 1.0])
    capture = np.array([1.0, 0.0, -1.0])
    oof, evaluation = standardized_capture_blends(
        direct, capture, direct, capture
    )
    np.testing.assert_allclose(oof["direct_capture_blend50"], 0.0)
    assert oof["direct_capture_blend25"][2] > oof["direct_capture_blend25"][0]
    assert oof["direct_capture_blend75"][2] < oof["direct_capture_blend75"][0]
    np.testing.assert_allclose(
        evaluation["direct_capture_blend50"],
        oof["direct_capture_blend50"],
    )


def test_distributional_net_combines_both_conditional_magnitudes() -> None:
    score = compose_distributional_net(
        np.array([0.75]),
        np.array([np.log1p(200.0)]),
        np.array([np.log1p(100.0)]),
    )
    np.testing.assert_allclose(score, [0.0125])


def test_multitask_distributional_net_is_positive_minus_loss() -> None:
    score = compose_multitask_distributional_net(
        np.array([[0.03, 0.01], [-0.02, 0.04]])
    )
    np.testing.assert_allclose(score, [0.02, -0.04])


def test_frozen_margin_interaction_is_smooth_and_only_amplifies_confidence() -> None:
    contract = load_frozen_base_margin_interaction(FROZEN_BASE_MARGIN_SCREEN)
    direct = np.tile(np.array([-1.0, 0.0, 1.0, 2.0]), 8)
    capture = direct.copy()
    low_margin = np.full(len(direct), contract["threshold"] - 10.0 * contract["robust_scale"])
    high_margin = np.full(len(direct), contract["threshold"] + 10.0 * contract["robust_scale"])
    low_oof, low_eval, _ = margin_capture_soft_interaction(
        direct, capture, low_margin, direct, capture, low_margin, contract=contract
    )
    high_oof, high_eval, report = margin_capture_soft_interaction(
        direct, capture, high_margin, direct, capture, high_margin, contract=contract
    )
    # The two low-confidence rows have zero positive confidence and therefore
    # cannot be promoted merely because the margin interaction exists.
    np.testing.assert_allclose(low_oof[:2], high_oof[:2])
    np.testing.assert_allclose(low_eval[:2], high_eval[:2])
    assert high_oof[3] > low_oof[3]
    assert high_eval[3] > low_eval[3]
    assert report["margin_gate_evaluation_mean"] > 0.99


def test_frozen_margin_screen_hash_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "frozen_screens.csv"
    changed.write_text("feature,direction_tp_over_fp,frozen_selected_book_median,frozen_control_scale\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_base_margin_interaction(changed)
