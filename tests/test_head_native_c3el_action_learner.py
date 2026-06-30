import argparse

import pandas as pd
import pytest

from scripts.run_head_native_c3el_action_learner import (
    _apply_action_feature_min_guard,
    _apply_action_feature_max_guard,
    _apply_preset,
    _choose_threshold,
    _make_action_target,
    _parse_float_grid,
    _parse_head_feature_min_map,
    _parse_head_feature_max_map,
    _parse_head_float_map,
    _parse_head_float_list_map,
    _parse_head_int_map,
    _parse_head_str_map,
    _pre_start_deployable_candidates,
    _resolve_active_heads,
    _resolve_head_config,
    _resolve_selected_heads,
    _schedule_action_summary,
    _schedule_for_heads,
    _score_actions,
)


def test_parse_head_maps() -> None:
    assert _parse_head_float_map("short_asset=320,short_boll=50") == {
        "short_asset": 320.0,
        "short_boll": 50.0,
    }
    assert _parse_head_float_map("short_asset=320;short_boll=50") == {
        "short_asset": 320.0,
        "short_boll": 50.0,
    }
    assert _parse_head_str_map("short_boll=quantile") == {"short_boll": "quantile"}
    assert _parse_head_str_map("long_bars=strict;short_boll=positive_nonbaseline") == {
        "long_bars": "strict",
        "short_boll": "positive_nonbaseline",
    }
    assert _parse_head_feature_min_map("short_boll:projected_removed_trade_share_strategy=0.4") == {
        "short_boll": {"projected_removed_trade_share_strategy": 0.4}
    }
    assert _parse_head_feature_min_map(
        "short_boll:projected_removed_trade_share_strategy=0.4;short_asset:x=1"
    ) == {
        "short_boll": {"projected_removed_trade_share_strategy": 0.4},
        "short_asset": {"x": 1.0},
    }
    assert _parse_head_feature_max_map("short_asset:cooldown_count=38.5") == {
        "short_asset": {"cooldown_count": 38.5}
    }
    assert _parse_float_grid("0, 50,320") == [0.0, 50.0, 320.0]
    assert _parse_head_int_map("short_asset=60,short_boll=4.2") == {"short_asset": 60, "short_boll": 4}
    assert _parse_head_float_list_map("short_asset=0|320;short_boll=5|50") == {
        "short_asset": [0.0, 320.0],
        "short_boll": [5.0, 50.0],
    }


def test_resolve_active_heads_makes_all_heads_explicit() -> None:
    requested, effective = _resolve_active_heads("")

    assert requested == set()
    assert effective == {"long_bars", "long_dist", "short_asset", "short_boll"}


def test_resolve_active_heads_validates_unknown_heads() -> None:
    with pytest.raises(ValueError, match="Unknown active head"):
        _resolve_active_heads("short_asset,unknown_head")


def test_resolve_selected_heads_defaults_to_active_heads() -> None:
    requested, effective = _resolve_selected_heads("", active_heads={"short_asset", "short_boll"})

    assert requested == set()
    assert effective == {"short_asset", "short_boll"}


def test_resolve_selected_heads_can_explicitly_noop() -> None:
    requested, effective = _resolve_selected_heads("none", active_heads={"short_asset", "short_boll"})

    assert requested == set()
    assert effective == set()


def test_resolve_selected_heads_rejects_inactive_heads() -> None:
    with pytest.raises(ValueError, match="not active/scored"):
        _resolve_selected_heads("short_boll", active_heads={"short_asset"})


def test_schedule_for_heads_reverts_unselected_heads_to_noop() -> None:
    ts = pd.Timestamp("2026-06-22 00:00:00", tz="UTC")
    schedule = pd.DataFrame(
        {
            "timestamp": [ts, ts, ts],
            "strategy_id": ["short_asset_a", "short_boll_a", "long_bars_a"],
            "head": ["short_asset", "short_boll", "long_bars"],
            "multiplier": [0.0, 0.5, 0.75],
        }
    )

    filtered = _schedule_for_heads(schedule, {"short_boll"})

    by_head = filtered.set_index("head")["multiplier"].to_dict()
    assert by_head["short_boll"] == 0.5
    assert by_head["short_asset"] == 1.0
    assert by_head["long_bars"] == 1.0


def test_schedule_action_summary_reports_interventions_by_head() -> None:
    ts = pd.Timestamp("2026-06-22 00:00:00", tz="UTC")
    schedule = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts],
            "strategy_id": ["short_asset_a", "short_asset_b", "short_boll_a"],
            "head": ["short_asset", "short_asset", "short_boll"],
            "multiplier": [1.0, 0.5, 0.0],
        }
    )

    summary = _schedule_action_summary(schedule, label="applied").set_index("head")

    assert summary.loc["short_asset", "groups"] == 2
    assert summary.loc["short_asset", "intervention_groups"] == 1
    assert summary.loc["short_boll", "intervention_groups"] == 1


def test_resolve_head_config_keeps_full_contract_head_specific() -> None:
    args = argparse.Namespace(
        group_target_mode="positive_nonbaseline",
        epsilon_gain=50.0,
        epsilon_margin=25.0,
        epsilon_gain_per_notional=0.001,
        epsilon_margin_per_notional=0.0005,
        min_train_groups=80,
        min_positive_groups=8,
        min_threshold_keep=2,
        threshold_holdout_frac=0.3,
        max_group_features=48,
        max_action_features=64,
        eval_keep_multiplier=4.0,
        max_eval_keep_share=0.1,
        action_model_objective="regression",
        action_quantile_alpha=0.2,
        action_score_mode="value",
        action_positive_epsilon=50.0,
    )

    config = _resolve_head_config(
        head="short_boll",
        args=args,
        threshold_grid=[0.35, 0.55],
        base_min_pred_delta_grid=[0.0],
        allowed_multipliers={0.0, 0.5, 0.75},
        group_target_mode_by_head={"short_boll": "strict"},
        epsilon_gain_by_head={"short_boll": 20.0},
        epsilon_margin_by_head={},
        epsilon_gain_per_notional_by_head={},
        epsilon_margin_per_notional_by_head={},
        threshold_grid_by_head={"short_boll": [0.1, 0.2]},
        min_pred_delta_grid_by_head={"short_boll": [5.0, 25.0]},
        allowed_multipliers_by_head={"short_boll": [0.0, 0.25]},
        min_train_groups_by_head={"short_boll": 20},
        min_positive_groups_by_head={"short_boll": 3},
        min_threshold_keep_by_head={"short_boll": 1},
        threshold_holdout_frac_by_head={"short_boll": 0.15},
        max_group_features_by_head={"short_boll": 12},
        max_action_features_by_head={"short_boll": 16},
        eval_keep_multiplier_by_head={"short_boll": 2.0},
        max_eval_keep_share_by_head={"short_boll": 0.05},
        action_model_objective_by_head={"short_boll": "quantile"},
        action_quantile_alpha_by_head={"short_boll": 0.1},
        action_score_mode_by_head={"short_boll": "prob_x_value"},
        action_positive_epsilon_by_head={"short_boll": 10.0},
        fallback_thresholds={"short_boll": 0.9},
        fallback_min_delta_by_head={"short_boll": 50.0},
    )

    assert config["group_target_mode"] == "strict"
    assert config["epsilon_gain"] == 20.0
    assert config["threshold_grid"] == [0.1, 0.2, 0.9]
    assert config["min_pred_delta_grid"] == [5.0, 25.0, 50.0]
    assert config["allowed_multipliers"] == [0.0, 0.25]
    assert config["min_train_groups"] == 20
    assert config["min_positive_groups"] == 3
    assert config["min_threshold_keep"] == 1
    assert config["threshold_holdout_frac"] == 0.15
    assert config["max_group_features"] == 12
    assert config["max_action_features"] == 16
    assert config["eval_keep_multiplier"] == 2.0
    assert config["max_eval_keep_share"] == 0.05
    assert config["action_model_objective"] == "quantile"
    assert config["action_quantile_alpha"] == 0.1
    assert config["action_score_mode"] == "prob_x_value"
    assert config["action_positive_epsilon"] == 10.0


def test_short_asset_default_preset_sets_validated_defaults() -> None:
    args = argparse.Namespace(
        preset="short_asset_default",
        active_heads="",
        min_train_groups=80,
        min_positive_groups=8,
        fallback_thresholds="",
        fallback_max_eval_keep_share_by_head="",
        fallback_min_pred_delta_by_head="",
        guard_low_strategy_candidate_count_max_by_head="",
        guard_min_removed_trade_share_timestamp_by_head="",
    )

    _apply_preset(args)

    assert args.active_heads == "short_asset"
    assert args.min_train_groups == 60
    assert args.min_positive_groups == 5
    assert args.fallback_thresholds == "short_asset=0.8"
    assert args.fallback_max_eval_keep_share_by_head == "short_asset=0.25"
    assert args.fallback_min_pred_delta_by_head == "short_asset=320"
    assert args.guard_low_strategy_candidate_count_max_by_head == "short_asset=24"
    assert args.guard_min_removed_trade_share_timestamp_by_head == "short_asset=0.55"


def test_short_boll_challenger_preset_sets_research_guard() -> None:
    args = argparse.Namespace(
        preset="short_asset_plus_shortboll_guard04",
        active_heads="",
        min_train_groups=80,
        min_positive_groups=8,
        fallback_thresholds="",
        fallback_max_eval_keep_share_by_head="",
        fallback_min_pred_delta_by_head="",
        guard_low_strategy_candidate_count_max_by_head="",
        guard_min_removed_trade_share_timestamp_by_head="",
        action_model_objective_by_head="",
        action_quantile_alpha_by_head="",
        action_feature_min_by_head="",
    )

    _apply_preset(args)

    assert args.active_heads == "short_asset,short_boll"
    assert args.fallback_thresholds == "short_asset=0.8,short_boll=0.0"
    assert args.fallback_min_pred_delta_by_head == "short_asset=320,short_boll=50"
    assert args.action_model_objective_by_head == "short_boll=quantile"
    assert args.action_quantile_alpha_by_head == "short_boll=0.2"
    assert args.action_feature_min_by_head == "short_boll:projected_removed_trade_share_strategy=0.4"


def test_action_feature_min_guard_reverts_only_selected_rows_below_threshold() -> None:
    ts = pd.Timestamp("2026-06-22 12:00:00", tz="UTC")
    selection = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
            "strategy_id": ["a", "b", "c"],
            "selected_multiplier": [0.0, 0.0, 1.0],
            "gate_keep": [True, True, False],
        }
    )
    action_rows = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
            "strategy_id": ["a", "b", "c"],
            "multiplier": [0.0, 0.0, 1.0],
            "projected_removed_trade_share_strategy": [0.39, 0.41, 0.0],
        }
    )

    guarded, count = _apply_action_feature_min_guard(
        selection,
        action_rows,
        min_rules={"projected_removed_trade_share_strategy": 0.4},
    )

    assert count == 1
    assert not bool(guarded.loc[0, "gate_keep"])
    assert guarded.loc[0, "selected_multiplier"] == 1.0
    assert bool(guarded.loc[0, "guard_action_feature_min"])
    assert bool(guarded.loc[1, "gate_keep"])
    assert guarded.loc[1, "selected_multiplier"] == 0.0
    assert not bool(guarded.loc[2, "gate_keep"])


def test_action_feature_min_guard_fails_closed_when_required_feature_missing() -> None:
    ts = pd.Timestamp("2026-06-22 12:00:00", tz="UTC")
    selection = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["a", "b"],
            "selected_multiplier": [0.0, 1.0],
            "gate_keep": [True, False],
        }
    )
    action_rows = pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": ["a"],
            "multiplier": [0.0],
        }
    )

    guarded, count = _apply_action_feature_min_guard(
        selection,
        action_rows,
        min_rules={"projected_removed_trade_share_strategy": 0.4},
    )

    assert count == 1
    assert not bool(guarded.loc[0, "gate_keep"])
    assert guarded.loc[0, "selected_multiplier"] == 1.0
    assert bool(guarded.loc[0, "guard_action_feature_min"])
    assert not bool(guarded.loc[1, "gate_keep"])


def test_action_feature_max_guard_reverts_only_selected_rows_above_threshold() -> None:
    ts = pd.Timestamp("2026-06-22 12:00:00", tz="UTC")
    selection = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
            "strategy_id": ["a", "b", "c"],
            "selected_multiplier": [0.0, 0.0, 1.0],
            "gate_keep": [True, True, False],
        }
    )
    action_rows = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
            "strategy_id": ["a", "b", "c"],
            "multiplier": [0.0, 0.0, 1.0],
            "cooldown_count": [39.0, 38.0, 55.0],
        }
    )

    guarded, count = _apply_action_feature_max_guard(
        selection,
        action_rows,
        max_rules={"cooldown_count": 38.5},
    )

    assert count == 1
    assert not bool(guarded.loc[0, "gate_keep"])
    assert guarded.loc[0, "selected_multiplier"] == 1.0
    assert bool(guarded.loc[0, "guard_action_feature_max"])
    assert bool(guarded.loc[1, "gate_keep"])
    assert guarded.loc[1, "selected_multiplier"] == 0.0
    assert not bool(guarded.loc[2, "gate_keep"])


def test_action_feature_max_guard_fails_closed_when_required_feature_missing() -> None:
    ts = pd.Timestamp("2026-06-22 12:00:00", tz="UTC")
    selection = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["a", "b"],
            "selected_multiplier": [0.0, 1.0],
            "gate_keep": [True, False],
        }
    )
    action_rows = pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": ["a"],
            "multiplier": [0.0],
        }
    )

    guarded, count = _apply_action_feature_max_guard(
        selection,
        action_rows,
        max_rules={"cooldown_count": 38.5},
    )

    assert count == 1
    assert not bool(guarded.loc[0, "gate_keep"])
    assert guarded.loc[0, "selected_multiplier"] == 1.0
    assert bool(guarded.loc[0, "guard_action_feature_max"])
    assert not bool(guarded.loc[1, "gate_keep"])


def test_pre_start_deployable_candidates_filters_eval_rows() -> None:
    start = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-05-31 23:00:00", tz="UTC"),
                pd.Timestamp("2026-06-01 00:00:00", tz="UTC"),
                pd.Timestamp("2026-06-01 01:00:00", tz="UTC"),
            ],
            "strategy_id": ["a", "a", "a"],
        }
    )

    out = _pre_start_deployable_candidates(frame, start=start)

    assert len(out) == 1
    assert out["timestamp"].max() < start


def test_pre_start_deployable_candidates_fails_closed_without_history() -> None:
    start = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-06-01 00:00:00", tz="UTC"),
                pd.Timestamp("2026-06-01 01:00:00", tz="UTC"),
            ],
            "strategy_id": ["a", "a"],
        }
    )

    with pytest.raises(ValueError, match="Refusing to fit EV curves"):
        _pre_start_deployable_candidates(frame, start=start)


def test_choose_threshold_selects_profitable_min_pred_delta_gate() -> None:
    ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    groups = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_asset_bad", "short_asset_good"],
            "head": ["short_asset", "short_asset"],
        }
    )
    action_scores = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_asset_bad", "short_asset_good"],
            "multiplier": [0.0, 0.0],
            "pred_action_delta_J": [100.0, 350.0],
            "action_binds": [1.0, 1.0],
        }
    )
    action_rows = action_scores.drop(columns=["pred_action_delta_J"]).copy()
    action_rows["delta_full_J"] = [-10.0, 100.0]

    threshold, diag = _choose_threshold(
        groups,
        action_rows,
        p=[0.9, 0.85],
        action_scores=action_scores,
        grid=[0.8],
        min_keep=1,
        min_pred_delta_grid=[0.0, 300.0],
        allowed_multipliers={0.0},
    )

    assert threshold == 0.8
    assert diag["min_pred_delta"] == 300.0
    assert diag["threshold_value"] == 100.0
    assert diag["threshold_keep"] == 1
    assert len(diag["threshold_trials"]) == 2


def test_choose_threshold_applies_feature_min_rules_before_value_selection() -> None:
    ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    groups = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_boll_bad", "short_boll_good"],
            "head": ["short_boll", "short_boll"],
        }
    )
    action_scores = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_boll_bad", "short_boll_good"],
            "multiplier": [0.0, 0.0],
            "pred_action_delta_J": [250.0, 240.0],
            "action_binds": [1.0, 1.0],
        }
    )
    action_rows = action_scores.drop(columns=["pred_action_delta_J"]).copy()
    action_rows["delta_full_J"] = [-100.0, 75.0]
    action_rows["cooldown_hours_max"] = [18.0, 24.0]

    threshold, diag = _choose_threshold(
        groups,
        action_rows,
        p=[0.9, 0.85],
        action_scores=action_scores,
        grid=[0.8],
        min_keep=1,
        min_pred_delta_grid=[0.0],
        allowed_multipliers={0.0},
        feature_min_rules={"cooldown_hours_max": 21.8583},
    )

    assert threshold == 0.8
    assert diag["threshold_keep"] == 1
    assert diag["threshold_value"] == 75.0
    assert diag["threshold_trials"][0]["feature_min_guarded"] == 1


def test_choose_threshold_fails_closed_when_all_eligible_actions_lose_money() -> None:
    ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    groups = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_boll_bad", "short_boll_worse"],
            "head": ["short_boll", "short_boll"],
        }
    )
    action_scores = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_boll_bad", "short_boll_worse"],
            "multiplier": [0.0, 0.0],
            "pred_action_delta_J": [250.0, 240.0],
            "action_binds": [1.0, 1.0],
        }
    )
    action_rows = action_scores.drop(columns=["pred_action_delta_J"]).copy()
    action_rows["delta_full_J"] = [-10.0, -100.0]

    threshold, diag = _choose_threshold(
        groups,
        action_rows,
        p=[0.9, 0.85],
        action_scores=action_scores,
        grid=[0.8],
        min_keep=1,
        min_pred_delta_grid=[0.0],
        allowed_multipliers={0.0},
    )

    assert threshold > 1.0
    assert diag["threshold_keep"] == 0
    assert diag["threshold_value"] == 0.0
    assert not diag["threshold_has_positive_value"]
    assert diag["threshold_trials"][0]["eligible"]
    assert diag["threshold_trials"][0]["value"] == -110.0


def test_make_action_target_requires_binding_and_positive_delta() -> None:
    rows = pd.DataFrame(
        {
            "action_binds": [1.0, 1.0, 0.0, 1.0],
            "delta_full_J": [75.0, 25.0, 100.0, -10.0],
        }
    )

    target = _make_action_target(rows, epsilon_gain=50.0)

    assert target.tolist() == [1, 0, 0, 0]


def test_score_actions_supports_direct_positive_probability_modes() -> None:
    actions = pd.DataFrame({"feature": [1.0, 2.0]})

    value_scored = _score_actions(
        {"constant": 20.0},
        [],
        pd.Series(dtype=float),
        {"constant": 0.25},
        [],
        pd.Series(dtype=float),
        actions,
        action_score_mode="value",
    )
    prob_scored = _score_actions(
        {"constant": 20.0},
        [],
        pd.Series(dtype=float),
        {"constant": 0.25},
        [],
        pd.Series(dtype=float),
        actions,
        action_score_mode="positive_probability",
    )
    combo_scored = _score_actions(
        {"constant": 20.0},
        [],
        pd.Series(dtype=float),
        {"constant": 0.25},
        [],
        pd.Series(dtype=float),
        actions,
        action_score_mode="prob_x_value",
    )

    assert value_scored["pred_action_delta_J"].tolist() == [20.0, 20.0]
    assert prob_scored["pred_action_delta_J"].tolist() == [0.25, 0.25]
    assert combo_scored["pred_action_delta_J"].tolist() == [5.0, 5.0]
    assert combo_scored["pred_action_value_raw"].tolist() == [20.0, 20.0]
    assert combo_scored["pred_action_positive_prob"].tolist() == [0.25, 0.25]
