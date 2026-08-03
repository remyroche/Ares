import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.controlled_target_supportive_ablation import (
    AcceptanceGates,
    derive_economic_targets,
    hurdle_decomposition_score,
    matched_support_population,
    pooled_global_top_k_metrics,
    stable_pooled_global_top_k,
    strict_oof_support_predictions,
    support_columns,
    validate_causal_raw_features,
)


def _panel() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=12, freq="D", tz="UTC")
    event = np.array([0, 1, 2] * 4)
    net = np.array([-0.01, -0.02, 0.03] * 4)
    return pd.DataFrame(
        {
            "candidate_id": [f"c{i:02}" for i in range(12)], "__ts__": ts,
            "__label_available_at__": ts + pd.Timedelta(hours=12),
            "oof_fold": np.repeat([0, 1, 2], 4), "side_name": ["long", "short"] * 6,
            "x": np.arange(12, dtype=float), "__first_touch_target_soft__": np.linspace(0, 1, 12),
            "execution_net_ev_12h": net, "execution_gross_ev_12h": net + .01, "execution_cost_return": .01,
            "__opportunity_occurred_12h__": (event == 2).astype(float),
            "favorable_first": (event == 2).astype(int), "adverse_first": (event == 1).astype(int), "timeout": (event == 0).astype(int),
            "__peak_mfe_atr_12h__": np.arange(12, dtype=float),
            "__time_to_first_meaningful_mfe_hours_12h__": np.arange(12, dtype=float),
            "__mae_before_meaningful_mfe_atr_12h__": np.arange(12, dtype=float),
            "__future_slope_atr_per_hour_12h__": np.arange(12, dtype=float),
        }
    )


def test_targets_are_cost_consistent_and_competing_events_are_explicit() -> None:
    result = derive_economic_targets(_panel(), hurdle_bps=25.0)
    assert result.target_t3_competing_class.tolist()[:3] == [0, 1, 2]
    assert result.target_t4_clear.tolist()[:3] == [0, 0, 1]
    assert result.target_t4_fail.tolist()[:3] == [1, 1, 0]
    assert result.target_t4_clear_excess_return.tolist()[:3] == pytest.approx([0.0, 0.0, 0.0275])
    assert result.target_t4_fail_shortfall_return.tolist()[:3] == pytest.approx([0.0125, 0.0225, 0.0])


def test_t4_hurdle_decomposition_uses_the_exact_clear_and_fail_identity() -> None:
    score = hurdle_decomposition_score(
        clear_probability=np.array([0.8, 0.1]),
        clear_conditional_excess=np.array([0.020, 0.040]),
        fail_probability=np.array([0.2, 0.9]),
        fail_conditional_shortfall=np.array([0.010, 0.030]),
        hurdle_bps=25.0,
    )
    expected = np.array([
        0.0025 + 0.8 * 0.020 - 0.2 * 0.010,
        0.0025 + 0.1 * 0.040 - 0.9 * 0.030,
    ])
    assert score == pytest.approx(expected)


def test_supports_are_strict_oof_and_all_arms_use_the_full_support_intersection() -> None:
    calls = []
    def predictor(train_x, train_y, test_x, kind):
        calls.append((len(train_x), len(test_x), kind))
        return np.full(len(test_x), train_y.mean())
    result = strict_oof_support_predictions(derive_economic_targets(_panel()), feature_columns=["x"], fold_column="oof_fold", predictor=predictor)
    # Fold zero cannot be OOF; it is removed for S0 as well as S1-S5.
    matched = matched_support_population(result)
    assert len(matched) == 8
    assert support_columns("S5")[-1] == "support_oof__future_slope_atr_per_hour"
    assert {kind for _, _, kind in calls} == {"binary", "regression"}


def test_candidate_policy_is_one_pooled_global_book_and_gates_latest_month() -> None:
    frame = _panel().iloc[:10].copy()
    frame["score"] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    selected = stable_pooled_global_top_k(frame, "score", fraction=.20)
    assert selected.candidate_id.tolist() == ["c00", "c01"]
    # This constructed book is net-negative and must fail, rather than being
    # accidentally accepted because a gross score or a per-side tail passed.
    metrics = pooled_global_top_k_metrics(frame, "score", gates=AcceptanceGates(minimum_selected_rows=1))
    assert metrics["selection_basis"] == "pooled_global_post_score_top_k"
    assert metrics["acceptance_passed"] is False


def test_future_targets_are_rejected_from_the_raw_feature_contract() -> None:
    try:
        validate_causal_raw_features(["x", "__peak_mfe_atr_12h__"])
    except ValueError as error:
        assert "non-causal" in str(error)
    else:
        raise AssertionError("path outcome label was accepted as a raw feature")
