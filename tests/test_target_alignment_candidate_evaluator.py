from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.target_alignment_candidate_evaluator import AlignmentEvaluationError, evaluate_target_arms, validate_pairing


def _frame() -> pd.DataFrame:
    rows = []
    timestamp = pd.Timestamp("2025-01-01T00:00:00Z")
    for arm, bump in (("T0_native24_control", 0.0), ("T1_clean_opportunity", .01), ("T2_direct_net", .02), ("T3_competing_risk_expected_net", .03), ("T4_hurdle_decomposition", .04)):
        for index, score in enumerate((.1, .9, .8, .2, .7, .6)):
            decision = timestamp + pd.Timedelta(hours=index)
            net = (.02 if index in (1, 2) else -.01)
            rows.append({"candidate_id": f"c{index}", "target_arm": arm, "support_stage": "S0", "score": score + bump,
                         "__decision_ts__": decision, "__label_available_at__": decision + pd.Timedelta(hours=12),
                         "prediction_fit_end_ts": decision - pd.Timedelta(hours=1), "prediction_generated_ts": decision,
                         "strict_prequential_oof": True, "diagnostic_noncausal_oof": False,
                         "execution_gross_ev_12h": net + .01, "execution_cost_return": .01, "execution_net_ev_12h": net,
                         "side_name": "long" if index < 3 else "short", "favorable_first": int(net > 0), "adverse_first": int(net <= 0), "timeout": 0})
    return pd.DataFrame(rows)


def test_target_evaluator_selects_one_pooled_global_tail_before_side_attribution() -> None:
    result = evaluate_target_arms(_frame(), entry_threshold=None)
    membership = result["global_tail_membership"]
    top20 = membership[(membership.target_arm == "T2_direct_net") & (membership.global_tail_fraction == .20)]
    # Scores 0.9 and 0.8 are both long: a side-local top20 implementation
    # would force a short row, while the pooled book deliberately does not.
    assert top20.candidate_id.tolist() == ["c1", "c2"]
    tails = result["global_tail_metrics"]
    assert tails.loc[(tails.target_arm == "T2_direct_net") & (tails.top_fraction == .20), "rows"].item() == 2
    assert result["correctness_checks"].passed.all()


def test_target_evaluator_requires_identical_candidate_rows_across_arms() -> None:
    frame = _frame().loc[lambda x: ~((x.target_arm == "T4_hurdle_decomposition") & (x.candidate_id == "c5"))].copy()
    checks = validate_pairing(frame)
    assert not next(row for row in checks if row["check"] == "identical_evaluation_rows_across_target_arms")["passed"]
    with pytest.raises(AlignmentEvaluationError, match="identical_evaluation_rows"):
        evaluate_target_arms(frame)


def test_target_evaluator_rejects_blocked_or_future_oof_rows() -> None:
    frame = _frame(); frame.loc[0, "strict_prequential_oof"] = False
    checks = validate_pairing(frame)
    assert not checks[0]["passed"]
    with pytest.raises(AlignmentEvaluationError, match="strict_prequential"):
        evaluate_target_arms(frame)


def test_threshold_policy_is_candidate_only_and_cost_is_counted_once() -> None:
    result = evaluate_target_arms(_frame(), entry_threshold=.75)
    threshold = result["threshold_entry_metrics"]
    row = threshold.loc[threshold.target_arm.eq("T2_direct_net")].iloc[0]
    assert row.selection_scope == "candidate_score_threshold_no_portfolio_constraints"
    assert row.rows == 2
    assert row.net_bps_per_trade == 200.0
    assert row.gross_bps_per_trade == 300.0
    assert row.cost_bps_per_trade == 100.0
