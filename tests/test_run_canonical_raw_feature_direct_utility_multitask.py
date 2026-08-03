"""Contracts for the raw-feature direct-utility multi-task runner.

These are deliberately unit-scale.  They exercise the experiment's immutable
data, time, target, task-loss, selection, and manifest contracts without
fitting a neural net or reading the full historical panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_canonical_raw_feature_direct_utility_multitask as runner


def _frame() -> pd.DataFrame:
    """Small, side-balanced fixture spanning the three frozen calendar blocks."""
    rows: list[dict[str, object]] = []
    starts = (
        pd.Timestamp("2025-02-10T00:00:00Z"),
        pd.Timestamp("2025-03-10T00:00:00Z"),
        pd.Timestamp("2025-04-10T00:00:00Z"),
    )
    for month, start in enumerate(starts, start=2):
        for side, symbol in (("long", "BTC"), ("short", "ETH")):
            decision = start + pd.Timedelta(hours=1)
            gross = 0.03 if side == "long" else -0.02
            cost = 0.01
            rows.append(
                {
                    "candidate_id": f"{month}-{side}",
                    "side_name": side,
                    "__symbol__": symbol,
                    "__ts__": start,
                    "__decision_ts__": decision,
                    "execution_label_end_utc": decision + pd.Timedelta(hours=12),
                    "execution_gross_ev_12h": gross,
                    "execution_cost_return": cost,
                    "execution_net_ev_12h": gross - cost,
                    "base_oof_score": 0.4 if side == "long" else 0.3,
                    "context__active_transition_probability": 0.2,
                    "context__health_score": 0.1,
                    "context__source_utc": decision,
                }
            )
    return pd.DataFrame(rows)


def test_population_contract_is_the_exact_path_context_intersection() -> None:
    contract = runner.population_contract()
    assert tuple(contract["identity"]) == runner.IDENTITY
    assert contract["expected_rows"] == 205_194
    assert contract["expected_rows_by_month"] == {
        "2025-02": 64_512,
        "2025-03": 71_424,
        "2025-04": 69_258,
    }
    assert contract["expected_rows_by_side"] == {"long": 102_597, "short": 102_597}
    assert contract["primary_target"] == "execution_net_ev_12h"
    assert contract["label_resolution_column"] == "execution_label_end_utc"
    assert contract["label_horizon_hours"] == 12


def test_explicit_identity_intersection_is_one_to_one_and_never_candidate_id_only() -> None:
    base = _frame().iloc[:4].copy()
    paths = base.loc[:, list(runner.IDENTITY)].copy()
    paths["__path_auxiliary_target_valid__"] = 1
    context = base.loc[:, list(runner.IDENTITY)].copy()
    context["base_oof_score"] = 0.1

    joined, audit = runner.intersect_exact_identities(
        {"base": base, "paths": paths, "context": context}, expected_rows=4
    )
    assert len(joined) == 4
    assert audit["mode"] == "explicit_common_identity_intersection_one_to_one"
    assert tuple(audit["keys"]) == runner.IDENTITY
    assert audit["common_rows"] == 4

    duplicate = paths.iloc[[0]].copy()
    duplicate["__symbol__"] = "NOT-THE-SAME-SYMBOL"
    # Candidate IDs alone are insufficient; identity collision must not be
    # silently collapsed or treated as a match.
    with pytest.raises(ValueError, match="duplicate|one-to-one|identity"):
        runner.intersect_exact_identities(
            {"base": base, "paths": pd.concat([paths, duplicate], ignore_index=True), "context": context},
            expected_rows=4,
        )


def test_calendar_split_freezes_february_then_march_then_april() -> None:
    feb, march, april = runner.split_february_march_april(_frame())
    assert feb["__ts__"].dt.strftime("%Y-%m").eq("2025-02").all()
    assert march["__ts__"].dt.strftime("%Y-%m").eq("2025-03").all()
    assert april["__ts__"].dt.strftime("%Y-%m").eq("2025-04").all()
    contract = runner.population_contract()
    assert contract["architecture_loss_development"] == ["2025-02"]
    assert contract["model_selection"] == ["2025-03"]
    assert contract["final_refit"] == ["2025-02", "2025-03"]
    assert contract["diagnostic_only"] == ["2025-04"]


def test_training_mask_uses_exact_resolved_12h_labels_and_purges_boundary_rows() -> None:
    frame = _frame().iloc[:4].copy()
    cutoff = pd.Timestamp("2025-03-01T00:00:00Z")
    poison = frame.iloc[[0]].copy()
    poison["candidate_id"] = "unresolved-at-boundary"
    poison["__ts__"] = pd.Timestamp("2025-02-28T11:00:00Z")
    poison["__decision_ts__"] = pd.Timestamp("2025-02-28T12:00:00Z")
    poison["execution_label_end_utc"] = cutoff
    poison["execution_net_ev_12h"] = 99.0
    poison["execution_gross_ev_12h"] = (
        poison["execution_net_ev_12h"] + poison["execution_cost_return"]
    )
    frame = pd.concat([frame, poison], ignore_index=True)

    runner.validate_exact_execution_targets(frame)
    mask = runner.resolved_training_mask(frame, cutoff)
    assert frame.loc[mask, "execution_label_end_utc"].lt(cutoff).all()
    assert not mask[frame.index[frame["candidate_id"].eq("unresolved-at-boundary")][0]]


def test_primary_target_must_be_exact_net_gross_less_cost_and_resolve_at_12h() -> None:
    frame = _frame()
    runner.validate_exact_execution_targets(frame)

    bad_net = frame.copy()
    bad_net.loc[0, "execution_net_ev_12h"] += 1e-6
    with pytest.raises(ValueError, match="gross.*cost|exact net"):
        runner.validate_exact_execution_targets(bad_net)

    bad_horizon = frame.copy()
    bad_horizon.loc[0, "execution_label_end_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="12-hour|12h|label"):
        runner.validate_exact_execution_targets(bad_horizon)


def test_target_path_and_action_fields_can_never_enter_raw_feature_matrix() -> None:
    frame = _frame()
    permitted = runner.select_raw_feature_columns(
        frame,
        ["base_oof_score", "context__active_transition_probability", "context__health_score"],
    )
    assert permitted == (
        "base_oof_score", "context__active_transition_probability", "context__health_score"
    )
    for forbidden in (
        "execution_net_ev_12h",
        "execution_mfe_return_12h",
        "exit_is_timeout",
        "opportunity_gross_above_cost",
        "__path_auxiliary_target_valid__",
        "__soft_tb_first_event__",
        "__meaningful_mfe_reached_12h__",
        "target_price_atr",
        "wait_action",
        "timing_prediction",
        "future_slope_atr_per_hour",
        "realized_timeout_outcome",
    ):
        with pytest.raises(ValueError, match="forbidden.*feature|target.*feature|leakage"):
            runner.select_raw_feature_columns(frame.assign(**{forbidden: 1.0}), ["base_oof_score", forbidden])


def test_context_sources_must_be_point_in_time_available_and_lineage_scoped() -> None:
    frame = _frame()
    runner.validate_context_availability(
        frame,
        source_columns={
            "context__active_transition_probability": "context__source_utc",
            "context__health_score": "context__source_utc",
        },
        health_lineage="historical_raw_alpha_v3",
    )

    future = frame.copy()
    future.loc[0, "context__source_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="point-in-time|future|source"):
        runner.validate_context_availability(
            future,
            source_columns={"context__active_transition_probability": "context__source_utc"},
            health_lineage="historical_raw_alpha_v3",
        )
    with pytest.raises(ValueError, match="lineage|health"):
        runner.validate_context_availability(
            frame,
            source_columns={"context__health_score": "context__source_utc"},
            health_lineage="current_execution_health",
        )


def test_auxiliary_arms_are_predeclared_low_weight_and_do_not_replace_direct_target() -> None:
    direct = runner.task_specs("direct_only")
    all_aux = runner.task_specs("all_aux_low_weight")
    assert tuple(direct) == ("direct_net",)
    assert "direct_net" in all_aux
    assert all_aux["direct_net"].weight == pytest.approx(1.0)
    for name, spec in all_aux.items():
        if name != "direct_net":
            assert 0.0 < spec.weight < all_aux["direct_net"].weight
    assert {"opportunity", "favorable_magnitude", "adverse_magnitude", "exit_conversion_loss", "timeout"}.issubset(all_aux)

    # Add-one-out arms are a fixed ablation, not an opportunity to tune weights.
    for name in runner.CORE_AUXILIARY_HEADS:
        arm = f"without_{name}"
        specs = runner.task_specs(arm)
        assert name not in specs
        assert set(all_aux).difference(specs) == {name}
        assert specs["direct_net"].weight == all_aux["direct_net"].weight


def test_exact_economic_auxiliary_targets_use_masks_and_policy_conversion_loss() -> None:
    frame = _frame().iloc[:2].copy()
    frame["execution_mfe_return_12h"] = [0.08, 0.01]
    frame["exit_is_timeout"] = [False, True]
    targets = runner.build_task_targets(frame, runner.ECONOMIC_TASKS)
    net = frame["execution_net_ev_12h"].to_numpy(float)
    np.testing.assert_array_equal(targets["opportunity"].values, net > 0.0)
    np.testing.assert_array_equal(targets["favorable_magnitude"].mask, net > 0.0)
    np.testing.assert_array_equal(targets["adverse_magnitude"].mask, net < 0.0)
    np.testing.assert_allclose(
        targets["exit_conversion_loss"].values,
        np.maximum(
            frame["execution_mfe_return_12h"].to_numpy(float)
            - frame["execution_gross_ev_12h"].to_numpy(float),
            0.0,
        ),
    )
    np.testing.assert_array_equal(targets["timeout"].values, [0.0, 1.0])


def test_grouped_hazard_is_not_an_eligible_transition_feature() -> None:
    assert not any("hazard" in name for name in runner.EXTERNAL_TRANSITION)


def test_masked_auxiliary_loss_is_regularization_not_an_unmasked_target_imputation() -> None:
    specs = runner.task_specs("all_aux_low_weight")
    predictions = {name: np.array([0.1, 0.9]) for name in specs}
    targets = {name: np.array([0.0, 1.0]) for name in specs}
    masks = {name: np.array([True, True]) for name in specs}
    baseline = runner.masked_multitask_loss(predictions, targets, masks, specs)

    changed = {key: value.copy() for key, value in targets.items()}
    changed["opportunity"][1] = 0.0
    with_opportunity = runner.masked_multitask_loss(predictions, changed, masks, specs)
    assert with_opportunity != pytest.approx(baseline)

    masks["opportunity"][:] = False
    masked_baseline = runner.masked_multitask_loss(predictions, targets, masks, specs)
    masked_changed = runner.masked_multitask_loss(predictions, changed, masks, specs)
    assert masked_changed == pytest.approx(masked_baseline)


def test_direct_net_is_the_only_eligible_ranking_score() -> None:
    assert runner.ranking_score_column() == "direct_net_score"
    for forbidden in (
        "opportunity_score",
        "favorable_magnitude_score",
        "exit_mixture_score",
        "probability_times_magnitude_score",
        "timing_prediction",
    ):
        with pytest.raises(ValueError, match="direct.*only|ranking.*direct|ineligible"):
            runner.validate_ranking_score_column(forbidden)

    frame = _frame().iloc[:4].copy()
    direct = np.array([0.9, 0.8, 0.7, 0.6])
    changed_auxiliary = np.array([-100.0, 100.0, 100.0, -100.0])
    first = runner.stable_global_top_mask(frame, direct, 0.50)
    second = runner.stable_global_top_mask(frame.assign(opportunity_score=changed_auxiliary), direct, 0.50)
    assert np.array_equal(first, second)


def test_global_top_k_is_pooled_deterministic_and_tie_breaks_candidate_id() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["z-long", "a-short", "b-long", "c-short"],
            "side_name": ["long", "short", "long", "short"],
            "__symbol__": ["A", "B", "C", "D"],
            "__ts__": pd.to_datetime(["2025-03-01T00:00Z"] * 4),
        }
    )
    first = runner.stable_global_top_mask(frame, [1.0, 1.0, 0.5, 0.4], 0.25)
    second = runner.stable_global_top_mask(frame, [1.0, 1.0, 0.5, 0.4], 0.25)
    assert np.array_equal(first, second)
    assert frame.loc[first, "candidate_id"].tolist() == ["a-short"]
    assert runner.ranking_scope() == "one_pooled_global_cross_timestamp_cross_side"


def test_causal_mapping_reference_excludes_current_and_unresolved_outcomes() -> None:
    frame = _frame().iloc[:4].copy()
    snapshot = pd.Timestamp("2025-03-10T14:00:00Z")
    current = frame.iloc[[2]].copy()
    current["__decision_ts__"] = snapshot
    current["execution_label_end_utc"] = snapshot + pd.Timedelta(hours=12)
    frame = pd.concat([frame, current], ignore_index=True)
    reference = runner.causal_mapping_reference_mask(frame, snapshot)
    assert frame.loc[reference, "execution_label_end_utc"].lt(snapshot).all()
    assert not reference[-1]


def test_manifest_declares_april_diagnostic_and_direct_only_selection_contract() -> None:
    manifest = runner.experiment_manifest_contract()
    assert manifest["selection"]["ranking_score"] == "direct_net_score"
    assert manifest["selection"]["scope"] == "pooled_global"
    assert manifest["selection"]["no_per_timestamp_quota"] is True
    assert manifest["selection"]["auxiliary_outputs_are_ranking_inputs"] is False
    assert manifest["validation"]["april_untouched_by_selection"] is True
    assert manifest["validation"]["april_status"] == "diagnostic_only_not_promotion_evidence"
    assert manifest["outputs"]["immutable"] is True
    assert manifest["outputs"]["sha256_manifest_sidecar"] is True
