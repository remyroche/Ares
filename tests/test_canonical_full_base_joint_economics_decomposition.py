import numpy as np
import pandas as pd
import pytest

from scripts import run_canonical_full_base_joint_economics_decomposition as runner


def _frame() -> pd.DataFrame:
    timestamp = pd.date_range("2025-02-01", periods=24 * 50, freq="h", tz="UTC")
    rows = []
    for index, value in enumerate(timestamp):
        for side in runner.SIDES:
            exit_class = runner.EXIT_CLASSES[index % len(runner.EXIT_CLASSES)]
            rows.append(
                {
                    "candidate_id": f"{index:05d}-{side}",
                    "side_name": side,
                    "__symbol__": "BTC",
                    "__ts__": value,
                    "__decision_ts__": value + pd.Timedelta(hours=1),
                    "execution_label_end_utc": value + pd.Timedelta(hours=13),
                    "effective_label_resolution_utc": value + pd.Timedelta(hours=13),
                    "execution_gross_ev_12h": 0.02,
                    "execution_cost_return": 0.01,
                    "execution_net_ev_12h": 0.01,
                    "opportunity_gross_above_cost_0bps": True,
                    "opportunity_gross_above_cost_25bps": True,
                    "execution_exit_class": exit_class,
                    **{f"exit_is_{name}": name == exit_class for name in runner.EXIT_CLASSES},
                }
            )
    return pd.DataFrame(rows)


def test_strict_expanding_folds_use_prior_resolved_rows_only():
    frame = _frame()
    folds = runner.make_expanding_folds()
    assert [(fold.validation_start, fold.validation_end) for fold in folds] == [
        (pd.Timestamp(start), pd.Timestamp(end)) for start, end in runner.FOLD_BOUNDARIES
    ]
    for fold in folds:
        train, validation = runner.fold_masks(frame, fold)
        assert frame.loc[train, "execution_label_end_utc"].lt(fold.validation_start).all()
        assert frame.loc[validation, "__ts__"].ge(fold.validation_start).all()
        assert frame.loc[validation, "__ts__"].lt(fold.validation_end).all()


def test_execution_label_end_is_the_only_temporal_availability_contract():
    frame = _frame().drop(columns=["effective_label_resolution_utc"])
    assert "effective_label_resolution_utc" not in runner.required_columns()
    development, _ = runner.split_development_april(frame)
    fold = runner.make_expanding_folds()[1]
    train, _ = runner.fold_masks(development, fold)
    assert development.loc[train, "execution_label_end_utc"].lt(fold.validation_start).all()


def test_opportunity_and_exit_composition_preserve_sign_and_do_not_double_count_adverse():
    component = pd.DataFrame(
        {
            "direct_net": [0.01],
            "p_gross_gt_cost": [0.60],
            "p_gross_gt_cost_25bps": [0.50],
            "conditional_favorable_payoff": [0.04],
            "conditional_adverse_loss_severity": [0.03],
            "p_exit_trailing": [0.25], "conditional_net_trailing": [0.05],
            "p_exit_timeout": [0.25], "conditional_net_timeout": [0.01],
            "p_exit_full_stop": [0.25], "conditional_net_full_stop": [-0.04],
            "p_exit_adverse_exit": [0.25], "conditional_net_adverse_exit": [-0.08],
        }
    )
    composed = runner.compose_component_scores(component)
    assert composed.loc[0, "opportunity_score"] == pytest.approx(0.60 * 0.04 - 0.40 * 0.03)
    expected_exit = 0.25 * (0.05 + 0.01 - 0.04 - 0.08)
    assert composed.loc[0, "exit_mixture_score"] == pytest.approx(expected_exit)
    # The exit mixture already contains the adverse-exit payoff exactly once.
    assert composed.loc[0, "exit_mixture_score"] != pytest.approx(expected_exit - 0.40 * 0.03)


def test_action_layer_and_outcome_fields_are_never_model_features():
    for name in (
        "mapped_expected_gross",
        "opportunity_margin_0bps",
        "execution_net_ev_12h",
        "exit_is_full_stop",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "target_price_atr",
        "wait_action",
        "timing_prediction",
    ):
        with pytest.raises(ValueError, match="forbidden model feature"):
            runner.validate_feature_names([name])
    assert runner.arm_features("S0", "long") == ("base_oof_score",)
    assert "base_oof_score" in runner.arm_features("S1+B", "short")


def test_side_collapse_uses_direct_anchor_then_abstains():
    frame = pd.DataFrame(
        {
            "candidate_id": [f"candidate-{index:02d}" for index in range(20)],
            "side_name": ["long", "long", "short", "short"] * 5,
        }
    )
    composed = [100.0, 99.0] + list(np.arange(18, 0, -1, dtype=float))
    direct = [100.0, 1.0, 99.0] + list(np.arange(17, 0, -1, dtype=float))
    # Composed top-10% collapses into long, while direct anchor is balanced.
    mask, mode = runner.side_balance_gate(
        frame, composed, direct, 0.01,
        min_side_rows=1, min_share=0.05,
    )
    # top-1% contains one row, so balance is impossible: deliberate abstention.
    assert not mask.any()
    assert mode == "abstain_side_collapse"

    mask, mode = runner.side_balance_gate(
        frame, composed, direct, 0.10,
        min_side_rows=1, min_share=0.05,
    )
    assert mode == "direct_anchor_fallback"
    assert frame.loc[mask, "side_name"].nunique() == 2


def test_global_ranking_is_deterministic_and_candidate_id_breaks_ties():
    frame = pd.DataFrame(
        {"candidate_id": ["z-long", "a-short", "b-long", "c-short"], "side_name": ["long", "short", "long", "short"]}
    )
    first = runner.stable_global_top_mask(frame, [1.0, 1.0, 0.5, 0.4], 0.20)
    second = runner.stable_global_top_mask(frame, [1.0, 1.0, 0.5, 0.4], 0.20)
    assert np.array_equal(first, second)
    assert frame.loc[first, "candidate_id"].tolist() == ["a-short"]


def test_fit_budget_is_fixed_depth_five_without_search():
    budget = runner.fit_budget()
    assert runner.GEOMETRY.name == "fixed_d5"
    assert budget["hpo_model_fits"] == 0
    assert budget["feature_selection_fits"] == 0
