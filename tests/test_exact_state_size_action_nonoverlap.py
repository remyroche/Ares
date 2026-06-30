import numpy as np
import pandas as pd
from pathlib import Path

from scripts.run_exact_state_size_action_learning import (
    GROUP_LABEL_COLUMNS,
    STAGE1_CONTEXT_INTERACTION_FEATURES,
    _add_stage1_context_interaction_features,
    _apply_action_secondary_schedule_acceptance,
    _apply_secondary_schedule_acceptance,
    _apply_strategy_secondary_schedule_acceptance,
    _choose_action_secondary_schedule_acceptance,
    _choose_secondary_schedule_acceptance,
    _choose_strategy_secondary_schedule_acceptance,
    _choose_stage1_calibrated_fixed_gate,
    _choose_strategy_stage1_calibrated_fixed_gates,
    _compact_stage1_feature_cols,
    _group_action_table,
    _group_opportunity_classifier_feature_cols,
    _group_rows_for_opportunity_classifier,
    _numeric_cols,
    _strategy_opportunity_fixed_schedule,
    _schedule_action_quality,
    _secondary_acceptance_feature_rows,
    _selector_transfer_diagnostics,
    _stage1_positive_gain_boost,
    _stage1_training_labels,
    _strategy_stage1_calibrated_fixed_schedule,
    _suppress_nearby_union_actions,
    _zero_cut_classifier_feature_cols,
)


HIGH_VALUE_ZERO_CLASSIFIER_ARMS = (
    "C3ej_bagged_safety_c3ed_or_high_value_zero_classifier_union_gate",
    "C3ek_bagged_safety_c3ed_or_high_value_zero_classifier_relaxed_union_gate",
    "C3el_bagged_safety_c3ed_or_high_value_zero_classifier_broad_union_gate",
    "C3em_bagged_safety_c3ed_or_high_value_zero_classifier_expanded_union_gate",
)

C3EN_RELAXED_ARM = "C3en_bagged_safety_c3ea_or_relaxed_pressure_opportunity_action_tail_veto_union_gate"
C3EO_RECALL_ARM = "C3eo_bagged_safety_c3ea_or_recall_pressure_opportunity_action_tail_veto_union_gate"
C3EP_RECALL_OPPORTUNITY_ARM = "C3ep_bagged_safety_c3ea_or_recall_opportunity_action_tail_veto_union_gate"
C3EQ_RECALL_OPPORTUNITY_TAIL_ONLY_ARM = "C3eq_bagged_safety_c3ea_or_recall_opportunity_action_tail_only_union_gate"
C3ER_RECALL_OPPORTUNITY_FAMILY_ARM = "C3er_bagged_safety_c3ea_or_recall_opportunity_group_action_family_tail_union_gate"
C3ES_RECALL_OPPORTUNITY_RANKER_ARM = "C3es_bagged_safety_c3ea_or_recall_opportunity_group_action_ranker_tail_union_gate"
C3ET_RECALL_OPPORTUNITY_ZERO_NO_TAIL_ARM = "C3et_bagged_safety_c3ea_or_recall_opportunity_zero_no_tail_union_gate"
C3EU_HIGH_RANK_GROUP_CLASSIFIER_ARM = "C3eu_bagged_safety_c3ed_or_high_rank_group_classifier_tail_union_gate"
C3EV_HIGH_RANK_GROUP_CLASSIFIER_NO_TAIL_ARM = "C3ev_bagged_safety_c3ed_or_high_rank_group_classifier_no_tail_union_gate"
C3EW_HIGH_RANK_GROUP_CLASSIFIER_MODERATE_NO_TAIL_ARM = (
    "C3ew_bagged_safety_c3ed_or_high_rank_group_classifier_moderate_no_tail_union_gate"
)
C3FB_HIGH_RANK_GROUP_CLASSIFIER_SECONDARY_ONLY_ARM = (
    "C3fb_high_rank_group_classifier_moderate_secondary_only_diagnostic"
)
C3EX_STRATEGY_CALIBRATED_ZERO_UNION_ARM = "C3ex_bagged_safety_c3ed_or_strategy_calibrated_zero_union_gate"
C3EY_FILTERED_STRATEGY_CALIBRATED_ZERO_UNION_ARM = (
    "C3ey_bagged_safety_c3ed_or_filtered_strategy_calibrated_zero_union_gate"
)
C3EZ_NO_HARM_STRATEGY_CALIBRATED_ZERO_UNION_ARM = (
    "C3ez_bagged_safety_c3ed_or_no_harm_strategy_calibrated_zero_union_gate"
)
C3FC_CAPACITY_RELEASE_OPPORTUNITY_ARM = (
    "C3fc_bagged_safety_c3ed_or_capacity_release_opportunity_action_tail_union_gate"
)
C3FD_LOW_STAGE1_MISSED_ORACLE_ARM = (
    "C3fd_bagged_safety_c3ed_or_low_stage1_missed_oracle_classifier_tail_union_gate"
)
C3FE_RELAXED_LOW_STAGE1_MISSED_ORACLE_ARM = (
    "C3fe_bagged_safety_c3ed_or_relaxed_low_stage1_missed_oracle_tail_union_gate"
)
C3FF_RELAXED_LOW_STAGE1_MISSED_ORACLE_PRECONDITION_ARM = (
    "C3ff_bagged_safety_c3ed_or_relaxed_low_stage1_missed_oracle_precondition_only_union_gate"
)
C3FG_ACTION_TAIL_RANKED_LOW_STAGE1_MISSED_ORACLE_ARM = (
    "C3fg_bagged_safety_c3ed_or_action_tail_ranked_low_stage1_missed_oracle_union_gate"
)
C3FH_ACTION_TAIL_RANKED_LOW_STAGE1_MISSED_ORACLE_PREFER_SECONDARY_ARM = (
    "C3fh_bagged_safety_c3ed_or_action_tail_ranked_low_stage1_missed_oracle_prefer_secondary_union_gate"
)
C3FI_ACTION_TAIL_RANKED_LOW_STAGE1_MISSED_ORACLE_LCB_GUARD_ARM = (
    "C3fi_bagged_safety_c3ed_or_action_tail_ranked_low_stage1_missed_oracle_lcb_guard_union_gate"
)
C3FJ_C3ED_STAGE1_LCB_GUARD_ARM = "C3fj_bagged_safety_c3ed_stage1_lcb_guard_gate"
C3FK_CALIBRATED_GROUP_HURDLE_OPPORTUNITY_UNION_ARM = (
    "C3fk_bagged_safety_c3ed_or_calibrated_group_hurdle_opportunity_union_gate"
)
C3FL_HIGH_RANK_CALIBRATED_GROUP_HURDLE_CONSENSUS_ARM = (
    "C3fl_bagged_safety_c3ed_or_high_rank_calibrated_group_hurdle_consensus_union_gate"
)
C3FM_CALIBRATED_GROUP_HURDLE_NO_ACTION_TAIL_ARM = (
    "C3fm_bagged_safety_c3ed_or_calibrated_group_hurdle_no_action_tail_union_gate"
)
C3FN_CALIBRATED_GROUP_HURDLE_FOLDFIT_ACTION_ACCEPTANCE_ARM = (
    "C3fn_bagged_safety_c3ed_or_calibrated_group_hurdle_foldfit_action_acceptance_union_gate"
)
C3FO_CALIBRATED_GROUP_HURDLE_POSITIVE_VALUE_ACCEPTANCE_ARM = (
    "C3fo_bagged_safety_c3ed_or_calibrated_group_hurdle_positive_value_acceptance_union_gate"
)
C3FP_RANK_CALIBRATED_VALUE_ARM = "C3fp_bagged_safety_c3ed_or_rank_calibrated_value_secondary_union_gate"
C3FQ_CALIBRATED_STAGE1_GROUP_ACTION_SELECTOR_ARM = "C3fq_calibrated_stage1_group_action_selector"
C3FA_CALIBRATED_GROUP_HURDLE_ARM = "C3fa_calibrated_group_hurdle_decomposed_gate"


def test_high_value_zero_classifier_arms_share_dependency_gates() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()
    target = f'"{HIGH_VALUE_ZERO_CLASSIFIER_ARMS[0]}" in selected_arms'
    positions = [idx for idx in range(len(source)) if source.startswith(target, idx)]

    assert positions, "expected at least one high-value classifier dependency gate"
    for pos in positions:
        block_end = source.find("):", pos)
        assert block_end > pos, f"could not locate dependency condition end near offset {pos}"
        block = source[pos:block_end]
        for arm in HIGH_VALUE_ZERO_CLASSIFIER_ARMS:
            assert f'"{arm}" in selected_arms' in block


def test_zero_cut_classifier_feature_cols_exclude_counterfactual_labels() -> None:
    frame = pd.DataFrame(
        {
            "strategy_id": ["short_asset", "short_asset", "long_dist"],
            "timestamp": pd.date_range("2026-05-01", periods=3, freq="h", tz="UTC"),
            "split": ["train", "train", "train"],
            "multiplier": [0.0, 0.0, 0.0],
            "zero_cut_target": [True, False, True],
            "zero_cut_trainable": [True, True, True],
            "best_gain": [100.0, -100.0, 50.0],
            "best_multiplier": [0.0, 1.0, 0.5],
            "best_margin": [80.0, 10.0, 25.0],
            "group_can_bind": [1.0, 1.0, 1.0],
            "strategy_rank_q90": [0.9, 0.8, 0.7],
            "wallet": [1000.0, 950.0, 1025.0],
            "strategy_expected_cost_mean": [1.0, 2.0, 3.0],
        }
    )

    cols = _zero_cut_classifier_feature_cols(frame)

    assert "strategy_rank_q90" in cols
    assert "wallet" in cols
    assert "best_gain" not in cols
    assert "best_multiplier" not in cols
    assert "best_margin" not in cols
    assert "group_can_bind" not in cols


def test_selector_transfer_diagnostics_reports_secondary_block_reason() -> None:
    ts = pd.Timestamp("2026-05-01 00:00:00", tz="UTC")
    schedule = pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": ["short_asset"],
            "multiplier": [1.0],
            "calibrated_group_hurdle_no_action_tail_secondary": [1.0],
            "calibrated_group_hurdle_positive_value_acceptance_secondary": [0.0],
            "action_secondary_acceptance_pass": [1.0],
            "p_intervene": [0.72],
            "cal_mean_delta_J": [-4.0],
            "pred_delta_J": [-2.0],
        }
    )
    eval_panel = pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "strategy_id": ["short_asset", "short_asset"],
            "multiplier": [1.0, 0.5],
            "delta_full_J": [0.0, 25.0],
            "delta_immediate_J": [0.0, 10.0],
            "delta_full_net_pnl": [0.0, 25.0],
            "delta_full_cost_pnl": [0.0, 0.0],
            "action_binds": [0.0, 1.0],
            "best_multiplier": [0.5, 0.5],
            "best_gain": [25.0, 25.0],
            "best_margin": [20.0, 20.0],
            "best_nonbaseline_gain": [25.0, 25.0],
            "best_nonbaseline_multiplier": [0.5, 0.5],
            "y_intervene": [1.0, 1.0],
            "affected_notional": [1000.0, 1000.0],
            "group_can_bind": [1.0, 1.0],
        }
    )

    diag = _selector_transfer_diagnostics("test_arm", 2, schedule, eval_panel)

    assert len(diag) == 1
    row = diag.iloc[0]
    assert bool(row["missed_positive_oracle"])
    assert bool(row["selector_secondary_candidate_available"])
    assert row["selector_transfer_reason"] == "blocked_positive_value_floor"
    assert "calibrated_group_hurdle_no_action_tail_secondary" in diag.columns
    assert "calibrated_group_hurdle_positive_value_acceptance_secondary" in diag.columns


def test_c3en_relaxed_pressure_action_tail_arm_is_registered_with_relaxed_rank_guard() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EN_RELAXED_ARM}"' in source
    assert f'"{C3EN_RELAXED_ARM}": c3en_relaxed_pressure_opportunity_action_tail_veto_union_schedule_factory' in source
    assert '"strategy_rank_q90": 0.70' in source


def test_c3eo_recall_pressure_action_tail_arm_is_registered_with_recall_acceptance() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EO_RECALL_ARM}"' in source
    assert f'"{C3EO_RECALL_ARM}": c3eo_recall_pressure_opportunity_action_tail_veto_union_schedule_factory' in source
    assert "secondary_recall_min_precision" in source
    assert "min_precision=0.85" in source


def test_c3ep_recall_opportunity_action_tail_arm_uses_recall_opportunity_thresholds() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EP_RECALL_OPPORTUNITY_ARM}"' in source
    assert f'"{C3EP_RECALL_OPPORTUNITY_ARM}": c3ep_recall_opportunity_action_tail_veto_union_schedule_factory' in source
    assert "strategy_opportunity_recall_thresholds" in source
    assert "strategy_opportunity_recall_min_precision" in source
    assert "strategy_opportunity_recall_require_no_harm" in source
    assert "require_no_harm=False" in source
    assert "max_intervention_rate=0.20" in source


def test_c3eq_recall_opportunity_tail_only_arm_bypasses_learned_secondary_acceptance() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EQ_RECALL_OPPORTUNITY_TAIL_ONLY_ARM}"' in source
    assert (
        f'"{C3EQ_RECALL_OPPORTUNITY_TAIL_ONLY_ARM}": '
        "c3eq_recall_opportunity_action_tail_only_union_schedule_factory"
    ) in source
    assert "recall_strategy_opportunity_action_tail_only_fixed_zero_gate" in source
    assert '"secondary_acceptance_enabled": False' in source
    assert '"secondary_action_tail_veto_enabled": True' in source


def test_c3er_recall_opportunity_group_action_family_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3ER_RECALL_OPPORTUNITY_FAMILY_ARM}"' in source
    assert (
        f'"{C3ER_RECALL_OPPORTUNITY_FAMILY_ARM}": '
        "c3er_recall_opportunity_group_action_family_tail_union_schedule_factory"
    ) in source
    assert "recall_strategy_opportunity_group_action_family_tail_gate" in source
    assert 'mode="group_action_family_only"' in source
    assert "allowed_multipliers={0.0, 0.5, 0.75}" in source


def test_c3es_recall_opportunity_group_action_ranker_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3ES_RECALL_OPPORTUNITY_RANKER_ARM}"' in source
    assert (
        f'"{C3ES_RECALL_OPPORTUNITY_RANKER_ARM}": '
        "c3es_recall_opportunity_group_action_ranker_tail_union_schedule_factory"
    ) in source
    assert "recall_strategy_opportunity_group_action_ranker_tail_gate" in source
    assert 'mode="group_action_ranker_recall_only"' in source
    assert "value_min=-np.inf" in source
    assert "secondary_action_ranker_score_shifted_within_group" in source
    assert "allowed_multipliers={0.0, 0.5, 0.75}" in source


def test_c3et_recall_opportunity_zero_no_tail_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3ET_RECALL_OPPORTUNITY_ZERO_NO_TAIL_ARM}"' in source
    assert (
        f'"{C3ET_RECALL_OPPORTUNITY_ZERO_NO_TAIL_ARM}": '
        "c3et_recall_opportunity_zero_no_tail_union_schedule_factory"
    ) in source
    assert "recall_strategy_opportunity_zero_no_action_tail_gate" in source
    assert '"secondary_action_tail_veto_enabled": False' in source
    assert "c3et_recall_opportunity_zero_no_tail_union_schedule_factory" in source


def test_c3eu_high_rank_group_classifier_arm_is_registered_with_tail_veto() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EU_HIGH_RANK_GROUP_CLASSIFIER_ARM}"' in source
    assert (
        f'"{C3EU_HIGH_RANK_GROUP_CLASSIFIER_ARM}": '
        "c3eu_high_rank_group_classifier_tail_union_schedule_factory"
    ) in source
    assert "high_rank_group_opportunity_classifier_holdout_top_fraction" in source
    assert "high_rank_group_opportunity_classifier" in source
    assert '"secondary_action_tail_veto_enabled": True' in source
    assert "feature_columns_json" in source


def test_c3ev_high_rank_group_classifier_no_tail_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EV_HIGH_RANK_GROUP_CLASSIFIER_NO_TAIL_ARM}"' in source
    assert (
        f'"{C3EV_HIGH_RANK_GROUP_CLASSIFIER_NO_TAIL_ARM}": '
        "c3ev_high_rank_group_classifier_no_tail_union_schedule_factory"
    ) in source
    assert "high_rank_group_opportunity_classifier_holdout_top_fraction" in source
    assert '"secondary_action_tail_veto_enabled": False' in source


def test_c3ew_high_rank_group_classifier_moderate_no_tail_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EW_HIGH_RANK_GROUP_CLASSIFIER_MODERATE_NO_TAIL_ARM}"' in source
    assert (
        f'"{C3EW_HIGH_RANK_GROUP_CLASSIFIER_MODERATE_NO_TAIL_ARM}": '
        "c3ew_high_rank_group_classifier_moderate_no_tail_union_schedule_factory"
    ) in source
    assert '"secondary_classifier_min_precision": 0.75' in source
    assert '"secondary_classifier_max_intervention_rate": 0.05' in source


def test_c3fb_high_rank_group_classifier_secondary_only_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FB_HIGH_RANK_GROUP_CLASSIFIER_SECONDARY_ONLY_ARM}"' in source
    assert (
        f'"{C3FB_HIGH_RANK_GROUP_CLASSIFIER_SECONDARY_ONLY_ARM}": '
        "c3fb_high_rank_group_classifier_moderate_secondary_only_schedule_factory"
    ) in source
    assert "high_rank_group_classifier_moderate_secondary_only_diagnostic" in source
    assert '"union_primary_arm": "none"' in source


def test_c3ex_strategy_calibrated_zero_union_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EX_STRATEGY_CALIBRATED_ZERO_UNION_ARM}"' in source
    assert (
        f'"{C3EX_STRATEGY_CALIBRATED_ZERO_UNION_ARM}": '
        "c3ex_strategy_calibrated_zero_union_schedule_factory"
    ) in source
    assert "C3dn_strategy_stage1_calibrated_fixed_zero_gate" in source
    assert "strategy_calibrated_zero_enabled" in source


def test_c3ey_filtered_strategy_calibrated_zero_union_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EY_FILTERED_STRATEGY_CALIBRATED_ZERO_UNION_ARM}"' in source
    assert (
        f'"{C3EY_FILTERED_STRATEGY_CALIBRATED_ZERO_UNION_ARM}": '
        "c3ey_filtered_strategy_calibrated_zero_union_schedule_factory"
    ) in source
    assert "strategy_calibrated_secondary_acceptance_enabled" in source
    assert "secondary_acceptance_enabled" in source


def test_c3ez_no_harm_strategy_calibrated_zero_union_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3EZ_NO_HARM_STRATEGY_CALIBRATED_ZERO_UNION_ARM}"' in source
    assert (
        f'"{C3EZ_NO_HARM_STRATEGY_CALIBRATED_ZERO_UNION_ARM}": '
        "c3ez_no_harm_strategy_calibrated_zero_union_schedule_factory"
    ) in source
    assert "strategy_calibrated_secondary_acceptance_no_harm" in source
    assert "require_no_harm=True" in source


def test_c3fc_capacity_release_opportunity_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FC_CAPACITY_RELEASE_OPPORTUNITY_ARM}"' in source
    assert (
        f'"{C3FC_CAPACITY_RELEASE_OPPORTUNITY_ARM}": '
        "c3fc_capacity_release_opportunity_action_tail_union_schedule_factory"
    ) in source
    assert "capacity_release_strategy_opportunity_acceptance_action_tail_fixed_zero_gate" in source
    assert '"strategy_rank_q90": 0.70' in source
    assert "positions_exiting_1h" in source
    assert "notional_exiting_1h_share" in source


def test_c3fd_low_stage1_missed_oracle_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FD_LOW_STAGE1_MISSED_ORACLE_ARM}"' in source
    assert (
        f'"{C3FD_LOW_STAGE1_MISSED_ORACLE_ARM}": '
        "c3fd_low_stage1_missed_oracle_classifier_tail_union_schedule_factory"
    ) in source
    assert "_low_stage1_missed_oracle_classifier_schedule" in source
    assert "low_stage1_missed_oracle_classifier_enabled" in source
    assert "max_stage1_p=0.40" in source
    assert '"cal_q10_delta_J_min": -90.0' in source


def test_c3fe_relaxed_low_stage1_missed_oracle_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FE_RELAXED_LOW_STAGE1_MISSED_ORACLE_ARM}"' in source
    assert (
        f'"{C3FE_RELAXED_LOW_STAGE1_MISSED_ORACLE_ARM}": '
        "c3fe_relaxed_low_stage1_missed_oracle_tail_union_schedule_factory"
    ) in source
    assert "relaxed_candidate_generator" in source
    assert "min_precision=0.0" in source
    assert "max_intervention_rate=0.03" in source


def test_c3ff_relaxed_low_stage1_missed_oracle_precondition_only_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FF_RELAXED_LOW_STAGE1_MISSED_ORACLE_PRECONDITION_ARM}"' in source
    assert (
        f'"{C3FF_RELAXED_LOW_STAGE1_MISSED_ORACLE_PRECONDITION_ARM}": '
        "c3ff_relaxed_low_stage1_missed_oracle_precondition_only_union_schedule_factory"
    ) in source
    assert "relaxed_precondition_only_diagnostic" in source
    assert "apply_action_tail_gate=False" in source


def test_c3fg_action_tail_ranked_low_stage1_missed_oracle_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FG_ACTION_TAIL_RANKED_LOW_STAGE1_MISSED_ORACLE_ARM}"' in source
    assert (
        f'"{C3FG_ACTION_TAIL_RANKED_LOW_STAGE1_MISSED_ORACLE_ARM}": '
        "c3fg_action_tail_ranked_low_stage1_missed_oracle_union_schedule_factory"
    ) in source
    assert "action_tail_ranked_candidate_generator" in source
    assert "action_tail_rank_weight=0.50" in source
    assert "low_stage1_action_tail_rank_score" in source


def test_c3fh_action_tail_ranked_low_stage1_missed_oracle_prefer_secondary_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FH_ACTION_TAIL_RANKED_LOW_STAGE1_MISSED_ORACLE_PREFER_SECONDARY_ARM}"' in source
    assert (
        f'"{C3FH_ACTION_TAIL_RANKED_LOW_STAGE1_MISSED_ORACLE_PREFER_SECONDARY_ARM}": '
        "c3fh_action_tail_ranked_low_stage1_missed_oracle_prefer_secondary_union_schedule_factory"
    ) in source
    assert "action_tail_ranked_prefer_secondary_diagnostic" in source
    assert 'prefer_source="secondary"' in source


def test_c3fi_action_tail_ranked_low_stage1_missed_oracle_lcb_guard_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FI_ACTION_TAIL_RANKED_LOW_STAGE1_MISSED_ORACLE_LCB_GUARD_ARM}"' in source
    assert (
        f'"{C3FI_ACTION_TAIL_RANKED_LOW_STAGE1_MISSED_ORACLE_LCB_GUARD_ARM}": '
        "c3fi_action_tail_ranked_low_stage1_missed_oracle_lcb_guard_union_schedule_factory"
    ) in source
    assert "action_tail_ranked_stage1_lcb_guard_candidate_generator" in source
    assert "secondary_stage1_lcb_mean_gain_min=20.0" in source


def test_c3fj_c3ed_stage1_lcb_guard_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FJ_C3ED_STAGE1_LCB_GUARD_ARM}"' in source
    assert f'"{C3FJ_C3ED_STAGE1_LCB_GUARD_ARM}": c3fj_c3ed_stage1_lcb_guard_schedule_factory' in source
    assert "stage1_lcb_guard_enabled" in source
    assert "stage1_lcb_guard_min" in source
    assert "stage1_lcb_guard_pass" in source


def test_low_stage1_missed_oracle_union_preserves_classifier_audit_on_primary() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()
    factory_start = source.index("def _low_stage1_missed_oracle_union_schedule_factory")
    factory_end = source.index("def c3fd_low_stage1_missed_oracle_classifier_tail_union_schedule_factory", factory_start)
    factory_source = source[factory_start:factory_end]

    assert "primary[\"low_stage1_missed_oracle_classifier_diag\"] = classifier_diag_json" in factory_source
    assert "primary[\"low_stage1_missed_oracle_classifier_selected_eval_rows\"]" in factory_source
    assert "primary[\"low_stage1_missed_oracle_classifier_raw_secondary_rows\"]" in factory_source
    assert "primary[\"low_stage1_missed_oracle_classifier_precondition_secondary_rows\"]" in factory_source
    assert "primary[\"low_stage1_missed_oracle_classifier_accepted_secondary_rows\"]" in factory_source
    assert "low_stage1_missed_oracle_classifier_stage1_lcb_mean_gain_min" in factory_source
    assert "low_stage1_missed_oracle_classifier_stage1_lcb_guard_pass_rows" in factory_source


def test_c3fa_calibrated_group_hurdle_arm_is_standalone_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FA_CALIBRATED_GROUP_HURDLE_ARM}"' in source
    assert (
        f'"{C3FA_CALIBRATED_GROUP_HURDLE_ARM}": '
        "c3fa_calibrated_group_hurdle_decomposed_schedule_factory"
    ) in source
    assert "stage1_group_hurdle_plus_decomposed_action_calibration" in source
    assert "_choose_calibrated_group_hurdle_gate" in source
    assert "_calibrated_group_hurdle_schedule" in source


def test_c3fk_calibrated_group_hurdle_opportunity_union_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FK_CALIBRATED_GROUP_HURDLE_OPPORTUNITY_UNION_ARM}"' in source
    assert (
        f'"{C3FK_CALIBRATED_GROUP_HURDLE_OPPORTUNITY_UNION_ARM}": '
        "c3fk_calibrated_group_hurdle_opportunity_union_schedule_factory"
    ) in source
    assert "calibrated_group_hurdle_recall_thresholds" in source
    assert "secondary_action_tail_veto_thresholds" in source
    assert "c3ed_pressure_opportunity_action_tail_veto_union_schedule_factory().copy()" in source


def test_c3fl_high_rank_calibrated_group_hurdle_consensus_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FL_HIGH_RANK_CALIBRATED_GROUP_HURDLE_CONSENSUS_ARM}"' in source
    assert (
        f'"{C3FL_HIGH_RANK_CALIBRATED_GROUP_HURDLE_CONSENSUS_ARM}": '
        "c3fl_high_rank_calibrated_group_hurdle_consensus_union_schedule_factory"
    ) in source
    assert "c3ed_primary_high_rank_calibrated_group_hurdle_consensus_secondary_union" in source
    assert "_high_rank_group_opportunity_classifier_schedule" in source
    assert "_calibrated_group_hurdle_schedule" in source
    assert "high_rank_calibrated_group_hurdle_consensus_secondary" in source
    assert "secondary_consensus_multiplier_policy" in source
    assert "least_severe_selected_multiplier" in source
    assert "secondary_action_tail_veto_thresholds" in source
    assert "calibrated_group_hurdle_recall_thresholds" in source


def test_c3fm_calibrated_group_hurdle_no_action_tail_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FM_CALIBRATED_GROUP_HURDLE_NO_ACTION_TAIL_ARM}"' in source
    assert (
        f'"{C3FM_CALIBRATED_GROUP_HURDLE_NO_ACTION_TAIL_ARM}": '
        "c3fm_calibrated_group_hurdle_no_action_tail_union_schedule_factory"
    ) in source
    assert "c3ed_primary_calibrated_group_hurdle_no_action_tail_secondary_union" in source
    assert "calibrated_group_hurdle_no_action_tail_secondary" in source
    assert '"secondary_action_tail_veto_enabled": False' in source
    assert '"secondary_action_tail_veto_thresholds": json.dumps({}, sort_keys=True)' in source
    assert "calibrated_group_hurdle_recall_thresholds" in source


def test_c3fn_calibrated_group_hurdle_foldfit_action_acceptance_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FN_CALIBRATED_GROUP_HURDLE_FOLDFIT_ACTION_ACCEPTANCE_ARM}"' in source
    assert (
        f'"{C3FN_CALIBRATED_GROUP_HURDLE_FOLDFIT_ACTION_ACCEPTANCE_ARM}": '
        "c3fn_calibrated_group_hurdle_foldfit_action_acceptance_union_schedule_factory"
    ) in source
    assert "c3ed_primary_calibrated_group_hurdle_foldfit_action_acceptance_secondary_union" in source
    assert "calibrated_group_hurdle_foldfit_action_acceptance_secondary" in source
    assert "calibrated_group_hurdle_action_acceptance_thresholds" in source
    assert "calibrated_group_hurdle_action_acceptance_diag" in source
    assert "_choose_action_secondary_schedule_acceptance" in source
    assert "require_no_harm=True" in source


def test_c3fo_calibrated_group_hurdle_positive_value_acceptance_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FO_CALIBRATED_GROUP_HURDLE_POSITIVE_VALUE_ACCEPTANCE_ARM}"' in source
    assert (
        f'"{C3FO_CALIBRATED_GROUP_HURDLE_POSITIVE_VALUE_ACCEPTANCE_ARM}": '
        "c3fo_calibrated_group_hurdle_positive_value_acceptance_union_schedule_factory"
    ) in source
    assert "c3ed_primary_calibrated_group_hurdle_positive_value_acceptance_secondary_union" in source
    assert "calibrated_group_hurdle_positive_value_acceptance_secondary" in source
    assert "secondary_positive_value_floor_cal_mean_delta_J_min" in source
    assert "secondary_positive_value_floor_pred_delta_J_min" in source


def test_c3fp_rank_calibrated_value_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FP_RANK_CALIBRATED_VALUE_ARM}"' in source
    assert (
        f'"{C3FP_RANK_CALIBRATED_VALUE_ARM}": '
        "c3fp_rank_calibrated_value_secondary_union_schedule_factory"
    ) in source
    assert "_choose_rank_calibrated_value_gate" in source
    assert "_rank_calibrated_value_schedule" in source
    assert "rank_calibrated_value_secondary" in source
    assert 'positive_value_thresholds["cal_mean_delta_J_min"]' in source
    assert 'positive_value_thresholds["pred_delta_J_min"]' in source


def test_c3fq_stage1_group_action_selector_arm_is_registered() -> None:
    source = Path("scripts/run_exact_state_size_action_learning.py").read_text()

    assert f'"{C3FQ_CALIBRATED_STAGE1_GROUP_ACTION_SELECTOR_ARM}"' in source
    assert "_choose_stage1_group_gate" in source
    assert "_stage1_group_gate_schedule" in source
    assert "calibrated_stage1_group_gate_then_action_selector" in source


def test_group_opportunity_classifier_features_exclude_counterfactual_labels() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=4, freq="h", tz="UTC"),
            "strategy_id": ["short_asset", "short_asset", "long_dist", "short_boll"],
            "multiplier": [1.0, 1.0, 1.0, 1.0],
            "group_can_bind": [1.0, 1.0, 1.0, 1.0],
            "strategy_rank_q90": [0.9, 0.8, 0.95, 0.75],
            "y_intervene": [1.0, 0.0, 0.0, 1.0],
            "best_multiplier": [0.0, 1.0, 1.0, 0.5],
            "best_gain": [100.0, -10.0, 0.0, 80.0],
            "best_margin": [40.0, 0.0, 0.0, 30.0],
            "best_gain_per_notional": [0.01, 0.0, 0.0, 0.008],
            "best_margin_per_notional": [0.004, 0.0, 0.0, 0.003],
            "best_nonbaseline_gain": [100.0, -5.0, 0.0, 80.0],
            "worst_nonbaseline_gain": [25.0, -80.0, -60.0, 20.0],
            "strategy_rank_mean": [0.8, 0.7, 0.9, 0.6],
            "strategy_score_max": [0.9, 0.5, 0.7, 0.8],
            "positions_exiting_1h": [0.0, 1.0, 0.0, 2.0],
            "notional_exiting_1h_share": [0.0, 0.2, 0.0, 0.4],
            "wallet": [1000.0, 990.0, 980.0, 970.0],
        }
    )

    rows = _group_rows_for_opportunity_classifier(frame, material_gain=50.0, min_rank_q90=0.70)
    cols = _group_opportunity_classifier_feature_cols(rows)

    assert rows["group_opportunity_target"].sum() == 2
    assert rows["group_opportunity_trainable"].sum() >= 3
    assert "strategy_rank_mean" in cols
    assert "strategy_score_max" in cols
    assert "positions_exiting_1h" in cols or "notional_exiting_1h_share" in cols
    assert "best_gain" not in cols
    assert "best_margin" not in cols


def test_low_stage1_missed_oracle_classifier_features_exclude_targets() -> None:
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=4, freq="h", tz="UTC"),
            "strategy_id": ["short_asset", "short_asset", "long_dist", "short_boll"],
            "missed_oracle_target": [True, False, True, False],
            "missed_oracle_trainable": [True, True, True, True],
            "group_opportunity_target": [True, False, True, False],
            "group_opportunity_trainable": [True, True, True, True],
            "best_gain": [100.0, -50.0, 80.0, -20.0],
            "best_margin": [60.0, 0.0, 40.0, 0.0],
            "y_intervene": [1.0, 0.0, 1.0, 0.0],
            "p_intervene": [0.05, 0.10, 0.20, 0.30],
            "strategy_rank_q90": [0.9, 0.8, 0.95, 0.75],
            "strategy_score_max": [0.9, 0.5, 0.7, 0.8],
            "wallet": [1000.0, 990.0, 980.0, 970.0],
        }
    )

    cols = _group_opportunity_classifier_feature_cols(rows)

    assert "p_intervene" in cols
    assert "strategy_rank_q90" in cols
    assert "missed_oracle_target" not in cols
    assert "missed_oracle_trainable" not in cols
    assert "group_opportunity_target" not in cols
    assert "best_gain" not in cols
    assert "best_gain_per_notional" not in cols
    assert "best_margin_per_notional" not in cols
    assert "best_multiplier" not in cols
    assert "best_nonbaseline_gain" not in cols
    assert "worst_nonbaseline_gain" not in cols
    assert "group_can_bind" not in cols
    assert "y_intervene" not in cols


def test_suppress_nearby_union_actions_prefers_secondary_within_gap() -> None:
    schedule = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-20 13:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.0,
                "eligible_action": True,
                "selection_score": 10.0,
                "union_preferred_source": "secondary",
            },
            {
                "timestamp": "2026-05-21 21:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.5,
                "eligible_action": True,
                "selection_score": 100.0,
                "union_preferred_source": "primary",
            },
            {
                "timestamp": "2026-05-25 21:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.5,
                "eligible_action": True,
                "selection_score": 1.0,
                "union_preferred_source": "primary",
            },
        ]
    )

    filtered = _suppress_nearby_union_actions(schedule, min_gap_hours=48.0, prefer_source="secondary")
    active = filtered.loc[pd.to_numeric(filtered["multiplier"], errors="coerce") < 1.0]

    assert len(active) == 2
    assert pd.Timestamp("2026-05-20 13:00:00+00:00") in set(active["timestamp"])
    assert pd.Timestamp("2026-05-25 21:00:00+00:00") in set(active["timestamp"])
    suppressed = filtered.loc[filtered["timestamp"].eq(pd.Timestamp("2026-05-21 21:00:00+00:00"))].iloc[0]
    assert suppressed["multiplier"] == 1.0
    assert bool(suppressed["nonoverlap_suppressed"])


def test_suppress_nearby_union_actions_handles_duplicate_index_labels() -> None:
    schedule = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-20 13:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.0,
                "eligible_action": True,
                "selection_score": 10.0,
                "union_preferred_source": "primary",
            },
            {
                "timestamp": "2026-05-20 13:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.0,
                "eligible_action": False,
                "selection_score": 100.0,
                "union_preferred_source": "secondary",
            },
        ],
        index=[0, 0],
    )

    filtered = _suppress_nearby_union_actions(schedule, min_gap_hours=48.0, prefer_source="primary")
    active = filtered.loc[pd.to_numeric(filtered["multiplier"], errors="coerce").fillna(1.0) < 1.0]

    assert not filtered.duplicated(["timestamp", "strategy_id"]).any()
    assert len(active) == 1
    assert active.iloc[0]["union_preferred_source"] == "primary"


def test_suppress_nearby_union_actions_deduplicates_to_active_row() -> None:
    schedule = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-20 13:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 1.0,
                "eligible_action": False,
                "selection_score": 200.0,
                "union_preferred_source": "primary",
            },
            {
                "timestamp": "2026-05-20 13:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.0,
                "eligible_action": True,
                "selection_score": 10.0,
                "union_preferred_source": "secondary",
            },
        ]
    )

    filtered = _suppress_nearby_union_actions(schedule, min_gap_hours=48.0, prefer_source="primary")

    assert len(filtered) == 1
    assert filtered.iloc[0]["multiplier"] == 0.0
    assert filtered.iloc[0]["union_preferred_source"] == "secondary"


def test_suppress_nearby_union_actions_treats_multiplier_cut_as_active_without_flag() -> None:
    schedule = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-29 07:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 1.0,
                "eligible_action": False,
                "selection_score": 0.0,
                "union_preferred_source": "primary",
            },
            {
                "timestamp": "2026-04-29 07:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.0,
                "selection_score": 4.0,
                "union_preferred_source": "secondary",
            },
        ]
    )

    filtered = _suppress_nearby_union_actions(schedule, min_gap_hours=72.0, prefer_source="primary")

    assert len(filtered) == 1
    assert filtered.iloc[0]["multiplier"] == 0.0
    assert filtered.iloc[0]["union_preferred_source"] == "secondary"


def test_apply_secondary_schedule_acceptance_can_be_chained() -> None:
    schedule = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.0,
                "selection_score": 5.0,
                "eligible_action": True,
            },
            {
                "timestamp": "2026-05-01 01:00:00+00:00",
                "strategy_id": "long_dist",
                "multiplier": 0.0,
                "selection_score": 6.0,
                "eligible_action": True,
            },
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.0,
                "strategy_open_count": 2.0,
            },
            {
                "timestamp": "2026-05-01 01:00:00+00:00",
                "strategy_id": "long_dist",
                "multiplier": 0.0,
                "strategy_open_count": 4.0,
            },
        ]
    )

    first = _apply_secondary_schedule_acceptance(
        schedule,
        panel,
        {"enabled": True, "min_thresholds": {}, "max_thresholds": {"strategy_open_count": 5.0}},
    )
    second = _apply_secondary_schedule_acceptance(
        first,
        panel,
        {"enabled": True, "min_thresholds": {}, "max_thresholds": {"strategy_open_count": 2.0}},
    )

    active = second.loc[pd.to_numeric(second["multiplier"], errors="coerce").fillna(1.0) < 1.0]
    vetoed = second.loc[second["strategy_id"].eq("long_dist")].iloc[0]
    assert len(active) == 1
    assert active.iloc[0]["strategy_id"] == "short_boll"
    assert vetoed["multiplier"] == 1.0
    assert not bool(vetoed["eligible_action"])


def test_apply_secondary_schedule_acceptance_can_fail_closed_when_rule_disabled() -> None:
    schedule = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.0,
                "selection_score": 5.0,
                "eligible_action": True,
            }
        ]
    )

    filtered = _apply_secondary_schedule_acceptance(
        schedule,
        pd.DataFrame(),
        {"enabled": False, "min_thresholds": {}, "max_thresholds": {}},
        default_pass=False,
    )

    assert filtered.iloc[0]["multiplier"] == 1.0
    assert filtered.iloc[0]["selection_score"] == 0.0
    assert not bool(filtered.iloc[0]["eligible_action"])
    assert not bool(filtered.iloc[0]["secondary_acceptance_pass"])


def test_schedule_action_quality_ignores_numerical_dust_positive_delta() -> None:
    schedule = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "long_dist",
                "multiplier": 0.0,
                "selection_score": 1.0,
            }
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "long_dist",
                "multiplier": 0.0,
                "delta_full_J": 5e-7,
                "delta_immediate_J": 0.0,
                "delta_full_net_pnl": 5e-7,
                "delta_full_cost_pnl": 0.0,
                "delta_full_turnover": 0.0,
                "action_binds": 1.0,
                "best_multiplier": 0.0,
                "best_gain": 5e-7,
                "y_intervene": 0.0,
            },
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "long_dist",
                "multiplier": 1.0,
                "delta_full_J": 0.0,
                "delta_immediate_J": 0.0,
                "delta_full_net_pnl": 0.0,
                "delta_full_cost_pnl": 0.0,
                "delta_full_turnover": 0.0,
                "action_binds": 0.0,
                "best_multiplier": 0.0,
                "best_gain": 5e-7,
                "y_intervene": 0.0,
            },
        ]
    )

    quality, _deciles = _schedule_action_quality("dust_arm", 0, schedule, panel)

    assert quality["intervention_count"] == 1
    assert quality["positive_action_count"] == 0
    assert quality["positive_action_rate"] == 0.0


def test_strategy_opportunity_fixed_schedule_applies_strategy_rule() -> None:
    scored = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_asset",
                "group_can_bind": 1.0,
                "strategy_candidate_count": 4.0,
                "strategy_rank_q90": 0.95,
                "strategy_open_count": 1.0,
            },
            {
                "timestamp": "2026-05-01 01:00:00+00:00",
                "strategy_id": "short_asset",
                "group_can_bind": 1.0,
                "strategy_candidate_count": 1.0,
                "strategy_rank_q90": 0.60,
                "strategy_open_count": 1.0,
            },
            {
                "timestamp": "2026-05-01 02:00:00+00:00",
                "strategy_id": "long_dist",
                "group_can_bind": 1.0,
                "strategy_candidate_count": 5.0,
                "strategy_rank_q90": 0.99,
                "strategy_open_count": 1.0,
            },
        ]
    )
    rules = pd.DataFrame(
        [
            {
                "strategy_id": "short_asset",
                "rule_kind": "min",
                "feature_a": "strategy_candidate_count",
                "threshold_a": 3.0,
                "feature_b": "",
                "threshold_b": np.nan,
                "top_fraction": 1.0,
                "multiplier": 0.0,
            }
        ]
    )

    schedule = _strategy_opportunity_fixed_schedule(scored, rules)
    active = schedule.loc[pd.to_numeric(schedule["multiplier"], errors="coerce").fillna(1.0) < 1.0]

    assert len(active) == 1
    assert active.iloc[0]["strategy_id"] == "short_asset"
    assert active.iloc[0]["timestamp"] == pd.Timestamp("2026-05-01 00:00:00+00:00")


def test_secondary_acceptance_features_use_selected_multiplier_action_impact() -> None:
    schedule = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.5,
                "selection_score": 4.0,
                "p_intervene": 0.8,
            }
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 1.0,
                "strategy_open_count": 2.0,
                "projected_notional_removed": 0.0,
                "projected_notional_removed_to_remaining_capital": 0.0,
            },
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.5,
                "strategy_open_count": 2.0,
                "projected_notional_removed": 500.0,
                "projected_notional_removed_to_remaining_capital": 0.25,
            },
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_boll",
                "multiplier": 0.0,
                "strategy_open_count": 2.0,
                "projected_notional_removed": 1000.0,
                "projected_notional_removed_to_remaining_capital": 0.50,
            },
        ]
    )

    rows = _secondary_acceptance_feature_rows(schedule, panel, include_labels=False)

    assert len(rows) == 1
    assert rows.iloc[0]["strategy_open_count"] == 2.0
    assert rows.iloc[0]["projected_notional_removed"] == 500.0
    assert rows.iloc[0]["projected_notional_removed_to_remaining_capital"] == 0.25


def test_stage1_positive_gain_boost_only_boosts_positive_labels() -> None:
    frame = pd.DataFrame(
        {
            "best_gain": [100.0, 100.0, 0.0],
            "best_gain_per_notional": [0.004, 0.004, 0.0],
        }
    )
    boost = _stage1_positive_gain_boost(
        frame,
        pd.Series([1, 0, 1]),
        epsilon_gain=50.0,
        epsilon_gain_per_notional=0.001,
        positive_gain_weight=0.25,
    )

    assert boost[0] > 1.0
    assert boost[1] == 1.0
    assert boost[2] == 1.0
    assert np.nanmax(boost) <= 5.0


def test_group_action_table_aggregates_nonbaseline_action_impact() -> None:
    panel = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_asset",
                "multiplier": 1.0,
                "delta_full_J": 0.0,
                "delta_immediate_J": 0.0,
                "affected_notional": 1000.0,
                "action_binds": False,
                "projected_notional_removed": 0.0,
                "projected_removed_trade_count": 0.0,
            },
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_asset",
                "multiplier": 0.5,
                "delta_full_J": 60.0,
                "delta_immediate_J": 10.0,
                "affected_notional": 1000.0,
                "action_binds": True,
                "projected_notional_removed": 500.0,
                "projected_removed_trade_count": 2.0,
            },
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_asset",
                "multiplier": 0.0,
                "delta_full_J": 20.0,
                "delta_immediate_J": 5.0,
                "affected_notional": 1000.0,
                "action_binds": True,
                "projected_notional_removed": 1000.0,
                "projected_removed_trade_count": 4.0,
            },
        ]
    )

    groups = _group_action_table(
        panel,
        epsilon_gain=50.0,
        epsilon_margin=25.0,
        epsilon_gain_per_notional=0.001,
        epsilon_margin_per_notional=0.0005,
    )
    row = groups.iloc[0]

    assert row["group_max_projected_notional_removed"] == 1000.0
    assert row["group_best_projected_notional_removed"] == 500.0
    assert row["group_max_projected_removed_trade_count"] == 4.0
    assert row["group_best_projected_removed_trade_count"] == 2.0


def test_compact_stage1_action_impact_features_are_opt_in() -> None:
    frame = pd.DataFrame(
        {
            "strategy_code": [0.0, 1.0],
            "group_affected_notional": [100.0, 200.0],
            "group_best_projected_notional_removed_to_remaining_capital": [0.1, 0.2],
            "group_best_projected_removed_trade_share_timestamp": [0.2, 0.3],
            "group_max_projected_notional_removed_to_remaining_capital": [0.3, 0.4],
            "group_max_projected_removed_trade_share_timestamp": [0.4, 0.5],
            "group_max_projected_notional_removed": [50.0, 100.0],
            "group_best_projected_notional_removed": [25.0, 50.0],
        }
    )
    candidates = list(frame.columns)

    default_cols = _compact_stage1_feature_cols(frame, candidates)
    opt_in_cols = _compact_stage1_feature_cols(frame, candidates, include_action_impact=True, action_impact_cap=2)

    assert "group_max_projected_notional_removed" not in default_cols
    assert "group_best_projected_notional_removed" not in default_cols
    action_impact_cols = [c for c in opt_in_cols if "projected_" in c]
    assert action_impact_cols == [
        "group_max_projected_notional_removed_to_remaining_capital",
        "group_max_projected_removed_trade_share_timestamp",
    ]
    assert "group_best_projected_notional_removed_to_remaining_capital" not in opt_in_cols
    assert "group_best_projected_removed_trade_share_timestamp" not in opt_in_cols


def test_oracle_best_projected_features_are_label_columns() -> None:
    frame = pd.DataFrame(
        {
            "strategy_code": [0.0],
            "group_best_projected_notional_removed": [10.0],
            "group_best_projected_action_strength": [1.0],
            "group_max_projected_notional_removed": [20.0],
        }
    )

    cols = _numeric_cols(frame, set(GROUP_LABEL_COLUMNS))

    assert "group_best_projected_notional_removed" not in cols
    assert "group_best_projected_action_strength" not in cols
    assert "group_max_projected_notional_removed" in cols


def test_stage1_context_interaction_features_are_finite_and_compact() -> None:
    frame = pd.DataFrame(
        {
            "strategy_code": [0.0, 1.0],
            "group_affected_notional": [200.0, 0.0],
            "group_can_bind": [1.0, 0.0],
            "remaining_slots": [1.0, 0.0],
            "remaining_capital": [1000.0, 0.0],
            "open_notional": [5000.0, 0.0],
            "open_positions": [4.0, 0.0],
            "strategy_above_threshold_count": [3.0, 2.0],
            "strategy_requested_notional_above_threshold": [600.0, 100.0],
            "group_max_projected_notional_removed": [150.0, 50.0],
            "strategy_rank_q90": [0.95, 0.55],
            "strategy_rank_q75": [0.80, 0.50],
            "strategy_rank_mean": [0.65, 0.40],
            "strategy_score_std": [0.20, 0.10],
            "strategy_score_mean": [1.0, 0.0],
            "timestamp_rank_q90": [0.90, 0.70],
            "timestamp_rank_mean": [0.55, 0.50],
            "timestamp_rank_std": [0.10, 0.05],
            "strategy_candidate_occupied_symbol_share": [0.50, 0.25],
            "strategy_candidate_cooldown_symbol_share": [0.25, 0.10],
            "cooldown_count": [2.0, 1.0],
            "strategy_open_notional_share": [0.20, 0.0],
            "strategy_notional_hhi": [0.40, 0.0],
            "side_notional_hhi": [0.50, 0.0],
            "side_long_open_count": [1.0, 0.0],
            "side_short_open_count": [3.0, 0.0],
            "strategy_expected_cost_q75": [0.04, 0.02],
            "strategy_candidate_count": [5.0, 2.0],
            "notional_exiting_24h_share": [0.10, 0.0],
            "positions_exiting_24h": [1.0, 0.0],
            "notional_exiting_72h_share": [0.30, 0.0],
            "positions_exiting_72h": [2.0, 0.0],
        }
    )

    out = _add_stage1_context_interaction_features(frame)

    for col in STAGE1_CONTEXT_INTERACTION_FEATURES:
        assert col in out.columns
        assert np.isfinite(out[col].to_numpy(dtype=float)).all()
    assert out.loc[0, "stage1_strategy_slot_pressure"] > 0.0
    assert out.loc[1, "stage1_strategy_slot_pressure"] == 0.0
    default_selected = _compact_stage1_feature_cols(out, list(out.columns))
    opt_in_selected = _compact_stage1_feature_cols(out, list(out.columns), include_context_interactions=True)

    assert "stage1_strategy_slot_pressure" not in default_selected
    assert "stage1_action_release_to_capacity" not in default_selected
    assert "stage1_strategy_slot_pressure" in opt_in_selected
    assert "stage1_action_release_to_capacity" in opt_in_selected


def test_compact_stage1_context_interactions_do_not_leak_from_cached_panels() -> None:
    frame = pd.DataFrame(
        {
            "strategy_code": [0.0],
            "group_affected_notional": [100.0],
            "group_can_bind": [1.0],
            "stage1_strategy_slot_pressure": [2.0],
            "stage1_action_release_to_capacity": [0.5],
        }
    )
    candidates = list(frame.columns)

    default_cols = _compact_stage1_feature_cols(frame, candidates)
    opt_in_cols = _compact_stage1_feature_cols(frame, candidates, include_context_interactions=True)

    assert "stage1_strategy_slot_pressure" not in default_cols
    assert "stage1_action_release_to_capacity" not in default_cols
    assert "stage1_strategy_slot_pressure" in opt_in_cols
    assert "stage1_action_release_to_capacity" in opt_in_cols


def test_stage1_relaxed_training_labels_preserve_strict_labels() -> None:
    groups = pd.DataFrame(
        {
            "y_intervene": [1.0, 0.0, 0.0, 0.0],
            "group_can_bind": [1.0, 1.0, 1.0, 0.0],
            "best_multiplier": [0.5, 0.5, 0.5, 0.5],
            "best_gain": [120.0, 80.0, 80.0, 80.0],
            "best_gain_per_notional": [0.010, 0.002, 0.0001, 0.002],
        }
    )

    strict = _stage1_training_labels(groups, mode="strict", epsilon_gain=50.0, epsilon_gain_per_notional=0.001)
    relaxed = _stage1_training_labels(groups, mode="relaxed_gain", epsilon_gain=50.0, epsilon_gain_per_notional=0.001)

    assert strict.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert relaxed.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert groups["y_intervene"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_stage1_decomposed_training_labels_require_clean_immediate_or_capacity_gain() -> None:
    groups = pd.DataFrame(
        {
            "y_intervene": [1.0, 0.0, 0.0, 0.0, 0.0],
            "group_can_bind": [1.0, 1.0, 1.0, 1.0, 1.0],
            "best_multiplier": [0.5, 0.5, 0.5, 0.5, 0.5],
            "best_gain": [120.0, 80.0, 80.0, 80.0, 80.0],
            "best_gain_per_notional": [0.010, 0.002, 0.002, 0.002, 0.002],
            "best_immediate_gain": [10.0, 60.0, -10.0, 10.0, 60.0],
            "best_capacity_gain": [110.0, 20.0, 90.0, 70.0, 20.0],
            "best_immediate_gain_per_notional": [0.001, 0.0015, -0.0002, 0.0001, 0.0015],
            "best_capacity_gain_per_notional": [0.009, 0.0005, 0.0022, 0.0018, 0.0005],
            "worst_nonbaseline_gain": [0.0, 0.0, 0.0, -60.0, -60.0],
        }
    )

    strict = _stage1_training_labels(groups, mode="strict", epsilon_gain=50.0, epsilon_gain_per_notional=0.001)
    relaxed = _stage1_training_labels(groups, mode="relaxed_gain", epsilon_gain=50.0, epsilon_gain_per_notional=0.001)
    immediate = _stage1_training_labels(groups, mode="immediate_gain", epsilon_gain=50.0, epsilon_gain_per_notional=0.001)
    capacity = _stage1_training_labels(groups, mode="capacity_gain", epsilon_gain=50.0, epsilon_gain_per_notional=0.001)
    decomposed = _stage1_training_labels(groups, mode="decomposed_gain", epsilon_gain=50.0, epsilon_gain_per_notional=0.001)

    assert strict.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert relaxed.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert immediate.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert capacity.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert decomposed.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert groups["y_intervene"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_stage1_moderate_capacity_gain_expands_training_only_recall() -> None:
    groups = pd.DataFrame(
        {
            "y_intervene": [0.0, 0.0, 0.0],
            "group_can_bind": [1.0, 1.0, 1.0],
            "best_multiplier": [0.0, 0.0, 0.0],
            "best_gain": [35.0, 35.0, 20.0],
            "best_gain_per_notional": [0.00075, 0.00075, 0.00075],
            "best_immediate_gain": [0.0, 30.0, 0.0],
            "best_capacity_gain": [35.0, 5.0, 20.0],
            "best_immediate_gain_per_notional": [0.0, 0.00075, 0.0],
            "best_capacity_gain_per_notional": [0.00075, 0.00010, 0.00075],
            "worst_nonbaseline_gain": [0.0, 0.0, 0.0],
        }
    )

    strict_capacity = _stage1_training_labels(
        groups,
        mode="capacity_gain",
        epsilon_gain=50.0,
        epsilon_gain_per_notional=0.001,
    )
    moderate_capacity = _stage1_training_labels(
        groups,
        mode="moderate_capacity_gain",
        epsilon_gain=50.0,
        epsilon_gain_per_notional=0.001,
    )

    assert strict_capacity.tolist() == [0.0, 0.0, 0.0]
    assert moderate_capacity.tolist() == [1.0, 0.0, 0.0]


def test_stage1_zero_cut_gain_targets_clean_full_cut_opportunities() -> None:
    groups = pd.DataFrame(
        {
            "y_intervene": [0.0, 0.0, 0.0, 0.0, 1.0],
            "group_can_bind": [1.0, 1.0, 1.0, 0.0, 1.0],
            "best_multiplier": [0.0, 0.5, 0.0, 0.0, 0.5],
            "best_gain": [35.0, 35.0, 35.0, 35.0, 10.0],
            "best_gain_per_notional": [0.00075, 0.00075, 0.00075, 0.00075, 0.0001],
            "worst_nonbaseline_gain": [0.0, 0.0, -60.0, 0.0, -100.0],
        }
    )

    labels = _stage1_training_labels(
        groups,
        mode="zero_cut_gain",
        epsilon_gain=50.0,
        epsilon_gain_per_notional=0.001,
    )

    assert labels.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert groups["y_intervene"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_group_action_table_adds_capacity_gain_diagnostics() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-05-01 00:00:00+00:00"] * 3,
            "strategy_id": ["short_asset"] * 3,
            "multiplier": [1.0, 0.5, 0.0],
            "delta_full_J": [0.0, 80.0, 30.0],
            "delta_immediate_J": [0.0, 20.0, 40.0],
            "affected_notional": [10000.0, 10000.0, 10000.0],
            "action_binds": [0.0, 1.0, 1.0],
        }
    )

    groups = _group_action_table(
        panel,
        epsilon_gain=50.0,
        epsilon_margin=25.0,
        epsilon_gain_per_notional=0.001,
        epsilon_margin_per_notional=0.0005,
    )

    row = groups.iloc[0]
    assert row["best_multiplier"] == 0.5
    assert row["best_gain"] == 80.0
    assert row["best_immediate_gain"] == 20.0
    assert row["best_capacity_gain"] == 60.0
    assert row["best_capacity_gain_per_notional"] == 0.006


def test_stage1_calibrated_fixed_gate_allows_rare_positive_rate_bins() -> None:
    timestamps = pd.date_range("2026-05-01", periods=20, freq="h", tz="UTC")
    scored = pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_id": ["short_asset"] * len(timestamps),
            "p_intervene": [0.40, 0.30] + [0.05] * 18,
            "stage1_cal_positive_rate": [0.30, 0.25] + [0.10] * 18,
            "stage1_cal_mean_gain": [80.0, 60.0] + [1.0] * 18,
            "stage1_cal_lcb_mean_gain": [20.0, 10.0] + [-10.0] * 18,
            "stage1_cal_bin_n": [20.0] * len(timestamps),
            "group_can_bind": [1.0] * len(timestamps),
        }
    )
    panel = pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_id": ["short_asset"] * len(timestamps),
            "multiplier": [0.5] * len(timestamps),
            "delta_full_J": [120.0, 90.0] + [-20.0] * 18,
            "action_binds": [1.0] * len(timestamps),
        }
    )

    thresholds, diag = _choose_stage1_calibrated_fixed_gate(
        scored,
        panel,
        multiplier=0.5,
        threshold_holdout_frac=0.0,
        min_precision=0.67,
    )

    assert np.isfinite(thresholds["p_min"])
    assert thresholds["cal_positive_rate_min"] < 0.50
    assert diag["train_interventions"] > 0
    assert diag["train_precision"] >= 0.67
    assert diag["train_delta_full_J_sum"] > 0.0


def test_strategy_stage1_calibrated_fixed_gate_keeps_bad_strategy_noop() -> None:
    timestamps = pd.date_range("2026-05-01", periods=20, freq="h", tz="UTC")
    scored = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "strategy_id": ["good"] * len(timestamps),
                    "p_intervene": [0.40, 0.30] + [0.05] * 18,
                    "stage1_cal_positive_rate": [0.30, 0.25] + [0.10] * 18,
                    "stage1_cal_mean_gain": [80.0, 60.0] + [1.0] * 18,
                    "stage1_cal_lcb_mean_gain": [20.0, 10.0] + [-10.0] * 18,
                    "stage1_cal_bin_n": [20.0] * len(timestamps),
                    "group_can_bind": [1.0] * len(timestamps),
                }
            ),
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "strategy_id": ["bad"] * len(timestamps),
                    "p_intervene": [0.45, 0.35] + [0.05] * 18,
                    "stage1_cal_positive_rate": [0.35, 0.30] + [0.10] * 18,
                    "stage1_cal_mean_gain": [90.0, 70.0] + [1.0] * 18,
                    "stage1_cal_lcb_mean_gain": [30.0, 20.0] + [-10.0] * 18,
                    "stage1_cal_bin_n": [20.0] * len(timestamps),
                    "group_can_bind": [1.0] * len(timestamps),
                }
            ),
        ],
        ignore_index=True,
    )
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "strategy_id": ["good"] * len(timestamps),
                    "multiplier": [0.0] * len(timestamps),
                    "delta_full_J": [120.0, 90.0] + [-20.0] * 18,
                    "action_binds": [1.0] * len(timestamps),
                }
            ),
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "strategy_id": ["bad"] * len(timestamps),
                    "multiplier": [0.0] * len(timestamps),
                    "delta_full_J": [-120.0, -90.0] + [-20.0] * 18,
                    "action_binds": [1.0] * len(timestamps),
                }
            ),
        ],
        ignore_index=True,
    )

    thresholds, diag = _choose_strategy_stage1_calibrated_fixed_gates(
        scored,
        panel,
        multiplier=0.0,
        threshold_holdout_frac=0.0,
        min_precision=0.67,
    )
    schedule = _strategy_stage1_calibrated_fixed_schedule(scored, thresholds)
    active = schedule.loc[pd.to_numeric(schedule["multiplier"], errors="coerce").fillna(1.0) < 1.0]

    assert diag["train_interventions"] > 0
    assert set(active["strategy_id"]) == {"good"}
    assert thresholds.loc[thresholds["strategy_id"].eq("bad"), "p_min"].iloc[0] == np.inf


def test_strategy_stage1_calibrated_fixed_schedule_top_fraction_uses_strategy_population() -> None:
    timestamps = pd.date_range("2026-05-01", periods=20, freq="h", tz="UTC")
    scored = pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_id": ["good"] * len(timestamps),
            "p_intervene": [0.9, 0.8, 0.7, 0.6, 0.5] + [0.0] * 15,
            "stage1_cal_positive_rate": [0.9] * 5 + [0.0] * 15,
            "stage1_cal_mean_gain": [100.0, 90.0, 80.0, 70.0, 60.0] + [0.0] * 15,
            "stage1_cal_lcb_mean_gain": [50.0] * 5 + [0.0] * 15,
            "stage1_cal_bin_n": [10.0] * 20,
            "group_can_bind": [1.0] * 20,
        }
    )
    thresholds = pd.DataFrame(
        [
            {
                "strategy_id": "good",
                "p_min": 0.5,
                "cal_positive_rate_min": 0.5,
                "cal_mean_gain_min": 1.0,
                "cal_lcb_mean_gain_min": 1.0,
                "cal_bin_n_min": 0.0,
                "top_fraction": 0.10,
                "multiplier": 0.0,
                "score_mode": "calibrated_mean",
            }
        ]
    )

    schedule = _strategy_stage1_calibrated_fixed_schedule(scored, thresholds)
    active = schedule.loc[pd.to_numeric(schedule["multiplier"], errors="coerce").fillna(1.0) < 1.0]

    assert len(active) == 2
    assert set(active["timestamp"]) == set(timestamps[:2])


def test_strategy_secondary_acceptance_rejects_missing_strategy_rule() -> None:
    timestamps = pd.date_range("2026-05-01", periods=40, freq="h", tz="UTC")
    active_ts = list(timestamps[:4])
    noop_ts = list(timestamps[4:])
    schedule = pd.DataFrame(
        {
            "timestamp": active_ts + active_ts + noop_ts + noop_ts,
            "strategy_id": ["good"] * 4 + ["bad"] * 4 + ["good"] * len(noop_ts) + ["bad"] * len(noop_ts),
            "multiplier": [0.0] * 8 + [1.0] * (2 * len(noop_ts)),
            "selection_score": [10.0, 11.0, 12.0, 13.0, 10.0, 11.0, 12.0, 13.0] + [0.0] * (2 * len(noop_ts)),
            "p_intervene": [0.8] * 8 + [0.0] * (2 * len(noop_ts)),
            "stage1_cal_positive_rate": [0.8] * 8 + [0.0] * (2 * len(noop_ts)),
            "stage1_cal_mean_gain": [50.0] * 8 + [0.0] * (2 * len(noop_ts)),
            "stage1_cal_lcb_mean_gain": [25.0] * 8 + [0.0] * (2 * len(noop_ts)),
        }
    )
    panel = pd.DataFrame(
        {
            "timestamp": active_ts + active_ts + noop_ts + noop_ts,
            "strategy_id": ["good"] * 4 + ["bad"] * 4 + ["good"] * len(noop_ts) + ["bad"] * len(noop_ts),
            "multiplier": [0.0] * 8 + [1.0] * (2 * len(noop_ts)),
            "delta_full_J": [20.0, 25.0, 30.0, 35.0, -20.0, -25.0, -30.0, -35.0] + [0.0] * (2 * len(noop_ts)),
            "delta_immediate_J": [1.0] * 8 + [0.0] * (2 * len(noop_ts)),
            "action_binds": [1.0] * 8 + [0.0] * (2 * len(noop_ts)),
            "best_gain": [20.0, 25.0, 30.0, 35.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * (2 * len(noop_ts)),
            "best_margin": [20.0, 25.0, 30.0, 35.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * (2 * len(noop_ts)),
            "y_intervene": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * (2 * len(noop_ts)),
        }
    )

    rules, diag = _choose_strategy_secondary_schedule_acceptance(
        schedule,
        panel,
        threshold_holdout_frac=0.0,
        min_precision=0.8,
    )
    filtered = _apply_strategy_secondary_schedule_acceptance(schedule, panel, rules, default_pass=False)
    active = filtered.loc[pd.to_numeric(filtered["multiplier"], errors="coerce").fillna(1.0) < 1.0]

    assert diag["strategy_rule_count"] == 1
    assert set(rules) == {"good"}
    assert set(active["strategy_id"]) == {"good"}


def test_secondary_acceptance_require_no_harm_rejects_harmful_rule() -> None:
    timestamps = pd.date_range("2026-05-01", periods=46, freq="h", tz="UTC")
    active_ts = list(timestamps[:6])
    noop_ts = list(timestamps[6:])
    schedule = pd.DataFrame(
        {
            "timestamp": active_ts + noop_ts,
            "strategy_id": ["short_asset"] * len(timestamps),
            "multiplier": [0.0] * len(active_ts) + [1.0] * len(noop_ts),
            "selection_score": [10.0] * len(active_ts) + [0.0] * len(noop_ts),
            "p_intervene": [0.8] * len(timestamps),
            "stage1_cal_positive_rate": [0.8] * len(timestamps),
            "stage1_cal_mean_gain": [50.0] * len(timestamps),
            "stage1_cal_lcb_mean_gain": [25.0] * len(timestamps),
        }
    )
    panel = pd.DataFrame(
        {
            "timestamp": active_ts + noop_ts,
            "strategy_id": ["short_asset"] * len(timestamps),
            "multiplier": [0.0] * len(active_ts) + [1.0] * len(noop_ts),
            "delta_full_J": [40.0, 35.0, 30.0, -5.0, 25.0, 20.0] + [0.0] * len(noop_ts),
            "delta_immediate_J": [1.0] * len(active_ts) + [0.0] * len(noop_ts),
            "action_binds": [1.0] * len(active_ts) + [0.0] * len(noop_ts),
            "best_gain": [40.0, 35.0, 30.0, 0.0, 25.0, 20.0] + [0.0] * len(noop_ts),
            "best_margin": [40.0, 35.0, 30.0, 0.0, 25.0, 20.0] + [0.0] * len(noop_ts),
            "y_intervene": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0] + [0.0] * len(noop_ts),
        }
    )

    loose_rule, loose_diag = _choose_secondary_schedule_acceptance(
        schedule,
        panel,
        threshold_holdout_frac=0.0,
        min_precision=0.8,
        require_no_harm=False,
    )
    strict_rule, strict_diag = _choose_secondary_schedule_acceptance(
        schedule,
        panel,
        threshold_holdout_frac=0.0,
        min_precision=0.8,
        require_no_harm=True,
    )

    assert loose_rule["enabled"]
    assert loose_diag["train_interventions"] == 6
    assert loose_diag["false_selection_penalty"] == 5.0
    assert not strict_rule["enabled"]
    assert strict_diag["train_interventions"] == 0
    assert strict_diag["require_no_harm"]


def test_action_secondary_acceptance_uses_action_level_safety() -> None:
    timestamps = pd.date_range("2026-05-01", periods=46, freq="h", tz="UTC")
    active_ts = list(timestamps[:6])
    noop_ts = list(timestamps[6:])
    schedule = pd.DataFrame(
        {
            "timestamp": active_ts + noop_ts,
            "strategy_id": ["short_asset"] * len(timestamps),
            "multiplier": [0.0] * len(active_ts) + [1.0] * len(noop_ts),
            "selection_score": [10.0] * len(active_ts) + [0.0] * len(noop_ts),
        }
    )
    scored_actions = pd.DataFrame(
        {
            "timestamp": active_ts,
            "strategy_id": ["short_asset"] * len(active_ts),
            "multiplier": [0.0] * len(active_ts),
            "action_binds": [1.0] * len(active_ts),
            "p_action_economic_positive": [0.9, 0.85, 0.8, 0.1, 0.75, 0.7],
            "p_action_positive": [0.8, 0.8, 0.75, 0.8, 0.7, 0.7],
            "cal_positive_rate": [0.8, 0.8, 0.75, 0.8, 0.7, 0.7],
            "cal_mean_delta_J": [40.0, 35.0, 30.0, 45.0, 25.0, 20.0],
            "cal_lcb_mean_delta_J": [10.0, 10.0, 10.0, 10.0, 5.0, 5.0],
            "cal_q25_delta_J": [5.0] * len(active_ts),
            "pred_delta_J": [40.0, 35.0, 30.0, 45.0, 25.0, 20.0],
            "ranker_score": [40.0, 35.0, 30.0, 45.0, 25.0, 20.0],
            "delta_full_J": [40.0, 35.0, 30.0, -5.0, 25.0, 20.0],
            "delta_immediate_J": [1.0] * len(active_ts),
        }
    )

    thresholds, diag = _choose_action_secondary_schedule_acceptance(
        schedule,
        scored_actions,
        threshold_holdout_frac=0.0,
        min_precision=1.0,
        require_no_harm=True,
    )
    filtered = _apply_action_secondary_schedule_acceptance(schedule, scored_actions, thresholds, default_pass=False)
    active = filtered.loc[pd.to_numeric(filtered["multiplier"], errors="coerce").fillna(1.0) < 1.0]

    assert thresholds["enabled"]
    assert diag["train_precision"] == 1.0
    assert diag["false_selection_penalty"] == 0.0
    assert len(active) > 0
    assert pd.Timestamp(active_ts[3]) not in set(active["timestamp"])


def test_action_secondary_acceptance_fails_closed_without_enabled_rule() -> None:
    schedule = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=3, freq="h", tz="UTC"),
            "strategy_id": ["short_asset"] * 3,
            "multiplier": [0.0, 0.0, 1.0],
            "selection_score": [10.0, 9.0, 0.0],
        }
    )
    filtered = _apply_action_secondary_schedule_acceptance(schedule, pd.DataFrame(), {"enabled": False}, default_pass=False)

    assert (pd.to_numeric(filtered["multiplier"], errors="coerce").fillna(1.0) == 1.0).all()
    assert not filtered.get("eligible_action", pd.Series(False, index=filtered.index)).fillna(False).astype(bool).any()
