from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.residual_event_archetypes import (
    EXECUTABLE_FAILURE_TARGETS,
    GlobalEVThresholdState,
    RESIDUAL_EVENT_PREFIX,
    RESIDUAL_EVENT_TARGET_PREFIX,
    ResidualEventArchetypeConfig,
    ResidualEventArchetypeState,
    ResidualEventBaselineState,
    ScoreExpectationState,
    _executable_quality_targets,
    add_residual_event_targets,
    add_residual_event_temporal_context,
    causal_eight_day_hit_rate_overlay,
    residual_event_distilled_feature_names,
    residual_event_feature_names,
    residual_event_quality_probability_feature_names,
    screen_local_residual_features,
)
from scripts.run_residual_event_archetype_discovery import (
    _load_candidate_columns,
    _load_shards,
    _normalise_candidate_contract,
    _parse_choice_csv,
    _parse_float_csv,
    _parse_int_csv,
    _project_source_columns,
    _state_target_probability_separation,
    _surprise_autocorrelation,
)
from scripts.run_meta_residual_event_balanced_error_overlay import (
    _parse_state_group_filter,
)
from scripts.run_meta_residual_head_stack_ablation import _load_handoff_with_labels
from scripts.run_meta_residual_head_feature_ablation import (
    ARM_FEATURES,
    DIRECT_RESIDUAL_MECHANISMS,
    RELIABILITY_GATED_MECHANISMS,
    RESIDUAL_STATE_FEATURES,
    _direct_residual_mechanism_target,
    _direct_mechanism_support_rows,
    _load_saved_full_feature_ledgers,
    _load_saved_direct_mechanism_support_ledgers,
    _require_month_coverage,
    _causal_residual_shortfall_target,
    _contrastive_executable_failure_training_set,
    _gate_auxiliary_by_oof_reliability,
    _select_oof_feature_contract_groups,
    _causal_phase_state_context,
    _residual_state_observable,
)
from extreme_price_movements.global_residual_latent_state import PHASE_STATE_FEATURES


def _frame(rows: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01", periods=rows // 6, freq="D", tz="UTC"
    ).repeat(6)
    score = np.tile(np.linspace(0.05, 0.95, 6, dtype=np.float32), rows // 6)
    hit = (score >= 0.55).astype(np.float32)
    # A persistent adverse local stream gives the event annotator non-trivial
    # outcomes while preserving a non-RangeIndex test surface.
    adverse = (np.arange(rows) >= rows // 2) & (np.arange(rows) % 6 < 2)
    hit[adverse] = 0.0
    return pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": np.where(np.arange(rows) % 2, "A", "B"),
            "side_name": np.where(np.arange(rows) % 2, "long", "short"),
            "archetype_policy_key": np.where(np.arange(rows) % 3, "arch_a", "arch_b"),
            "score_meta_base_soft_label": score,
            "clean_exec": hit,
            "ev_after_1pct": hit * 0.02 - (1.0 - hit) * 0.01,
            "dirty_positive": (1.0 - hit),
            "full_path_bad_mae_1r": (1.0 - hit),
            "timeout": 0.0,
            "stop_or_adverse": 0.0,
            "directional_feature": score + np.sin(np.arange(rows)) * 0.01,
        },
        index=pd.Index(np.arange(10_000, 10_000 + rows)),
    )


def _config() -> ResidualEventArchetypeConfig:
    return ResidualEventArchetypeConfig(
        min_global_threshold_rows=60,
        min_local_threshold_rows=30,
        min_local_state_rows=10_000,
        min_side_state_rows=10_000,
        min_event_class_rows=5,
        timestamp_min_peers=2,
    )


def test_global_ev_threshold_is_train_fitted_and_index_safe() -> None:
    train = _frame()
    cfg = _config()
    state = GlobalEVThresholdState(cfg).fit(train)
    out = state.transform(train.drop(columns=["clean_exec", "ev_after_1pct"]))

    assert out.index.equals(train.index)
    assert out["resid_event_top10_population"].sum() > 0
    assert (
        out["resid_event_top20_population"].sum()
        >= out["resid_event_top10_population"].sum()
    )
    assert state.global_targets["top10"] >= state.global_targets["top20"]


def test_projected_source_load_is_balanced_and_keeps_requested_oos_month_full(
    tmp_path,
) -> None:
    root = tmp_path / "candidate_shards"
    root.mkdir()
    for token, rows in (("202601", 18), ("202602", 15)):
        frame = pd.DataFrame(
            {
                "__ts__": pd.date_range(
                    f"{token[:4]}-{token[4:]}-01", periods=rows, freq="h", tz="UTC"
                ),
                "__symbol__": "TEST",
                "side_name": "long",
                "archetype_policy_key": "long_mixed",
                "score_meta_base_soft_label": np.linspace(0.1, 0.9, rows),
                "clean_exec": 1.0,
                "ev_after_1pct": 0.01,
                "dirty_positive": 0.0,
                "full_path_bad_mae_1r": 0.0,
                "timeout": 0.0,
                "ema20_slope": np.linspace(-1.0, 1.0, rows),
                "oi_value_log_z_30d": np.linspace(-2.0, 2.0, rows),
                "mkt_breadth_4h": np.linspace(-0.5, 0.5, rows),
                "gmm_ood_score": np.linspace(0.0, 1.0, rows),
                # Outcome-looking fields must not be projected as inputs.
                "expost__future_loss": 1.0,
                "resid_event_target_top_tail_false_positive": 0.0,
            }
        )
        frame.to_parquet(root / f"candidates_{token}.parquet", index=False)

    projected = _project_source_columns(
        root,
        end=pd.Timestamp("2026-03-01", tz="UTC"),
        max_features=12,
    )
    assert projected is not None
    assert 0 < len(projected) <= 12
    assert "expost__future_loss" not in projected
    assert "resid_event_target_top_tail_false_positive" not in projected
    assert {"ema20_slope", "oi_value_log_z_30d", "mkt_breadth_4h", "gmm_ood_score"}.intersection(projected)

    loaded, coverage = _load_shards(
        root,
        end=pd.Timestamp("2026-03-01", tz="UTC"),
        requested_columns=projected,
        max_train_rows_per_shard=6,
        full_months=("2026-02",),
    )
    assert len(loaded.loc[loaded["__ts__"].dt.month.eq(2)]) == 15
    assert len(loaded.loc[loaded["__ts__"].dt.month.eq(1)]) <= 6
    jan = next(row for row in coverage if row["path"].endswith("candidates_202601.parquet"))
    feb = next(row for row in coverage if row["path"].endswith("candidates_202602.parquet"))
    assert jan["sampled_train_shard"] is True
    assert feb["sampled_train_shard"] is False


def test_local_ev_thresholds_respect_the_frozen_top20_score_floor() -> None:
    train = _frame()
    cfg = _config()
    state = GlobalEVThresholdState(cfg).fit(train)

    for payload in state.local_thresholds.values():
        floor = float(payload["top10_local_score_floor"])
        assert float(payload["top10"]) >= floor
        assert float(payload["top20"]) >= floor


def test_local_top20_support_fallback_prevents_state_starvation() -> None:
    rows = 4_000
    score = np.concatenate(
        [
            np.linspace(0.5, 1.0, rows // 2, dtype=np.float32),
            np.linspace(0.0, 0.5, rows // 2, dtype=np.float32),
        ]
    )
    archetype = np.repeat(["strong", "weak"], rows // 2)
    ev = np.where(archetype == "strong", score * 0.04, score * 0.002).astype(
        np.float32
    )
    train = pd.DataFrame(
        {
            "score_meta_base_soft_label": score,
            "ev_after_1pct": ev,
            "side_name": "short",
            "archetype_policy_key": archetype,
        }
    )
    cfg = ResidualEventArchetypeConfig(
        min_global_threshold_rows=60,
        min_local_threshold_rows=30,
        min_local_state_rows=150,
    )
    state = GlobalEVThresholdState(cfg).fit(train)
    weak = state.local_thresholds["short|weak"]

    assert weak["top20_source"] == 2.0
    assert weak["top20"] == pytest.approx(weak["top20_local_score_floor"])
    assert weak["top20_support"] >= cfg.min_local_state_rows


def test_temporal_state_context_collapses_symbols_before_differencing() -> None:
    timestamps = pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC").repeat(2)
    observable = pd.DataFrame(
        {
            "__ts__": timestamps,
            "side_name": "short",
            "archetype_policy_key": "short_breakout_precision",
        }
    )
    generated = pd.DataFrame(index=observable.index)
    posterior = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.2, 0.8], [0.2, 0.8], [0.3, 0.7], [0.3, 0.7]],
        dtype=np.float32,
    )
    for index in range(7):
        generated[f"{RESIDUAL_EVENT_PREFIX}gmm_cluster_posterior_{index}"] = (
            posterior[:, index] if index < 2 else 0.0
        )
    generated[f"{RESIDUAL_EVENT_PREFIX}gmm_ood_score"] = [0, 0, 2, 2, 0, 0]
    generated[f"{RESIDUAL_EVENT_PREFIX}dae_reconstruction_error_zscore"] = 0.0

    result = add_residual_event_temporal_context(
        generated, observable, ResidualEventArchetypeConfig()
    )

    speed = result[f"{RESIDUAL_EVENT_PREFIX}posterior_speed"].to_numpy()
    assert speed[0] == speed[1] == 0.0
    assert speed[2] == pytest.approx(speed[3])
    assert speed[2] > speed[4] > 0.0
    recent = result[f"{RESIDUAL_EVENT_PREFIX}ood_recent_max_24h"].to_numpy()
    np.testing.assert_allclose(recent, [0, 0, 2, 2, 2, 2])


def test_transition_context_uses_causal_timestamp_trajectory_not_row_order() -> None:
    timestamps = pd.date_range("2026-07-01", periods=5, freq="h", tz="UTC").repeat(2)
    observable = pd.DataFrame(
        {
            "__ts__": timestamps,
            "side_name": "short",
            "archetype_policy_key": "short_breakout_precision",
            # Different symbols have different raw values, but the output is
            # the timestamp aggregate and must be identical within a bar.
            "mkt_systemic_deleveraging_score": np.repeat(
                [0.0, 0.1, 0.5, 0.4, 0.2], 2
            ),
            "market_breadth_recovery_from_24h_min": np.repeat(
                [0.0, 0.0, 0.1, 0.5, 0.8], 2
            ),
        }
    )
    generated = pd.DataFrame(index=observable.index)
    for index in range(7):
        generated[f"{RESIDUAL_EVENT_PREFIX}gmm_cluster_posterior_{index}"] = (
            1.0 if index == 0 else 0.0
        )
    generated[f"{RESIDUAL_EVENT_PREFIX}gmm_ood_score"] = 0.0
    generated[f"{RESIDUAL_EVENT_PREFIX}dae_reconstruction_error_zscore"] = 0.0

    result = add_residual_event_temporal_context(
        generated, observable, ResidualEventArchetypeConfig()
    )
    flush = result[f"{RESIDUAL_EVENT_PREFIX}oi_flush_impulse_1h"].to_numpy()
    breadth = result[
        f"{RESIDUAL_EVENT_PREFIX}breadth_recovery_impulse_4h"
    ].to_numpy()
    assert flush[0] == flush[1] == 0.0
    assert flush[4] == flush[5] > 0.0
    # The 4h trajectory cannot be populated before the causal 4h history.
    assert np.allclose(breadth[:8], 0.0)
    assert breadth[8] == breadth[9] > 0.0


def test_executable_quality_targets_keep_direction_and_path_damage_distinct() -> None:
    frame = pd.DataFrame(
        {
            "clean_exec": [1.0, 1.0, 0.0, 1.0],
            "ev_after_1pct": [0.02, -0.01, -0.02, 0.01],
            "resid_event_top10_population": [1.0, 1.0, 1.0, 0.0],
            "first_touch_bad_mae_1r": [0.0, 0.0, 1.0, 0.0],
            "full_path_bad_mae_1r": [0.0, 1.0, 1.0, 0.0],
            "timeout": [0.0, 0.0, 0.0, 1.0],
            "dirty_positive": [0.0, 1.0, 0.0, 1.0],
        }
    )
    targets = _executable_quality_targets(frame, ResidualEventArchetypeConfig())
    np.testing.assert_allclose(targets["correct_direction"], [1.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(targets["negative_executable_ev"], [0.0, 1.0, 1.0, 0.0])
    assert targets["correct_direction_bad_trade"][1] == 1.0
    assert targets["correct_direction_bad_trade"][2] == 0.0
    assert targets["adverse_path_damage"][1] > targets["adverse_path_damage"][0]
    assert targets["executable_adverse_path_event"][1] == 1.0
    assert targets["correct_direction_adverse_path_event"][1] == 1.0
    assert targets["correct_direction_adverse_path_event"][2] == 0.0


def test_executable_failure_targets_separate_top_tail_mechanisms() -> None:
    frame = pd.DataFrame(
        {
            # false positive, clean-cost-fragile, adverse loss, timeout,
            # dirty positive, clean executable, outside operational tail.
            "clean_exec": [0, 1, 1, 1, 1, 1, 0],
            "ev_after_1pct": [-0.02, -0.001, -0.02, 0.003, 0.004, 0.02, -0.02],
            "resid_event_top10_population": [1, 1, 1, 1, 1, 1, 0],
            "resid_event_top20_population": [1, 1, 1, 1, 1, 1, 1],
            "resid_event_class": [
                "negative_residual_event", "normal", "adverse_path_event",
                "negative_residual_event", "normal", "normal",
                "positive_residual_event",
            ],
            "first_touch_bad_mae_1r": [0, 0, 1, 0, 0, 0, 1],
            "full_path_bad_mae_1r": [0, 0, 1, 0, 0, 0, 1],
            "timeout": [0, 0, 0, 1, 0, 0, 0],
            "dirty_positive": [0, 0, 0, 0, 1, 0, 0],
            "stop_or_adverse": [0, 0, 1, 0, 0, 0, 1],
        }
    )
    targets = _executable_quality_targets(frame, ResidualEventArchetypeConfig())

    assert set(EXECUTABLE_FAILURE_TARGETS).issubset(targets)
    np.testing.assert_allclose(targets["top_tail_false_positive"], [1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(targets["top_tail_clean_cost_fragile"], [0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(targets["top_tail_adverse_loss"], [0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(targets["top_tail_timeout_failure"], [0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(targets["top_tail_dirty_positive"], [0, 0, 0, 0, 1, 0, 0])
    np.testing.assert_allclose(targets["top_tail_clean_executable"], [0, 0, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(targets["top_tail_residual_false_positive"], [1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(targets["top_tail_residual_adverse_loss"], [0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(targets["top_tail_residual_timeout_loss"], [0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(targets["near_tail_positive_residual_clean_executable"], [0, 0, 0, 0, 0, 0, 0])


def test_reversal_and_breakout_failure_targets_are_outcome_defined() -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short", "short", "long"],
            "archetype_policy_key": [
                "long_mixed_wideslow_tentative",
                "short_mixed_clean_path",
                "short_breakout_precision",
                "long_breakout_diagnostic_candidate",
            ],
            # The first two got direction right but became dirty/adverse
            # losses.  The latter two are an adverse short-breakout loss and
            # an ordinary wrong long-breakout call.
            "clean_exec": [1, 1, 0, 0],
            "ev_after_1pct": [-0.01, -0.02, -0.02, -0.01],
            "resid_event_top10_population": [1, 1, 1, 1],
            "resid_event_top20_population": [1, 1, 1, 1],
            "resid_event_class": ["negative_residual_event"] * 4,
            "full_path_bad_mae_1r": [1, 0, 1, 0],
            "first_touch_bad_mae_1r": [1, 0, 1, 0],
            "dirty_positive": [0, 1, 0, 0],
            "timeout": [0, 0, 0, 0],
            "stop_or_adverse": [1, 0, 1, 0],
            # These observable-looking values must not affect labels.
            "gmm_mahal_2": [0.0, 99.0, 5.0, 7.0],
            "dae_reconstruction_error_delta_1": [0.0, 88.0, 4.0, 6.0],
        }
    )
    targets = _executable_quality_targets(frame, ResidualEventArchetypeConfig())

    np.testing.assert_allclose(
        targets["top_tail_reversal_after_initial_success"], [1, 1, 0, 0]
    )
    np.testing.assert_allclose(
        targets["long_mixed_reversal_after_initial_success"], [1, 0, 0, 0]
    )
    np.testing.assert_allclose(
        targets["short_mixed_reversal_after_initial_success"], [0, 1, 0, 0]
    )
    np.testing.assert_allclose(
        targets["short_breakout_overconfident_path_loss"], [0, 0, 1, 0]
    )
    np.testing.assert_allclose(
        targets["long_breakout_overconfident_path_loss"], [0, 0, 0, 1]
    )


def test_executable_failure_priors_are_emitted_in_stable_contracts() -> None:
    full = set(residual_event_feature_names())
    distilled = set(residual_event_distilled_feature_names(include_market=False))
    probability_only = set(residual_event_quality_probability_feature_names())
    for target in EXECUTABLE_FAILURE_TARGETS:
        name = f"{RESIDUAL_EVENT_PREFIX}expected_{target}"
        assert name in full
        assert name in distilled
        assert name in probability_only
    assert probability_only == {
        f"{RESIDUAL_EVENT_PREFIX}expected_{target}"
        for target in EXECUTABLE_FAILURE_TARGETS
    }


def test_assessment_annotation_has_failure_labels_but_oos_transform_does_not() -> None:
    train = _frame()
    cfg = _config()
    state = ResidualEventArchetypeState(cfg).fit(
        train, candidate_features=["directional_feature"]
    )
    assessed = state.annotate_outcomes_for_assessment(train)
    for target in EXECUTABLE_FAILURE_TARGETS:
        assert f"{RESIDUAL_EVENT_TARGET_PREFIX}{target}" in assessed

    safe = train.drop(
        columns=[
            "clean_exec", "ev_after_1pct", "dirty_positive",
            "full_path_bad_mae_1r", "timeout", "stop_or_adverse",
        ]
    )
    transformed = state.transform_oos(safe)
    assert not any(name.startswith(RESIDUAL_EVENT_TARGET_PREFIX) for name in transformed)


def test_state_target_probability_report_uses_local_oos_lift() -> None:
    rows = 100
    probability = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "oos_month": "2026-06",
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "ev_after_1pct": np.where(probability >= 0.8, -0.01, 0.002),
            f"{RESIDUAL_EVENT_PREFIX}expected_top_tail_false_positive": probability,
            f"{RESIDUAL_EVENT_TARGET_PREFIX}top_tail_false_positive": (
                probability >= 0.8
            ).astype(np.float32),
        }
    )
    report = _state_target_probability_separation(frame)
    assert len(report) == 1
    assert report.loc[0, "state_target"] == "top_tail_false_positive"
    assert report.loc[0, "target_rate_lift"] > 4.0
    assert report.loc[0, "mean_ev_high_probability"] < 0.0


def test_m6_head_arm_uses_only_observable_residual_state_inputs() -> None:
    assert "M6_good_trade_plus_residual_event_states" in ARM_FEATURES
    assert "meta_aux_good_trade_oof" in ARM_FEATURES[
        "M6_good_trade_plus_residual_event_states"
    ]
    assert set(RESIDUAL_STATE_FEATURES).issubset(
        ARM_FEATURES["M6_good_trade_plus_residual_event_states"]
    )
    source = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=2, tz="UTC"),
            "score": [0.7, 0.8],
            "clean_exec": [1.0, 0.0],
            "ev_after_1pct": [0.01, -0.01],
            "meta_target_soft": [0.9, 0.1],
            "first_touch_bad_mae_1r": [0.0, 1.0],
        }
    )
    observable = _residual_state_observable(source)
    assert {"clean_exec", "ev_after_1pct", "meta_target_soft", "first_touch_bad_mae_1r"}.isdisjoint(observable.columns)
    assert {"__ts__", "score"}.issubset(observable.columns)


def test_m7_head_arm_uses_only_named_quality_probabilities() -> None:
    names = ARM_FEATURES["M7_good_trade_plus_residual_quality_probabilities"]
    assert names[0] == "meta_aux_good_trade_oof"
    assert set(names[1:]) == set(residual_event_quality_probability_feature_names())


def test_isolated_negative_residual_and_expected_surprise_arms_exclude_harmful_context() -> None:
    assert ARM_FEATURES["M12_good_trade_plus_negative_residual_probability"] == [
        "meta_aux_good_trade_oof",
        "meta_aux_negative_residual_oof",
    ]


def test_each_causal_phase_state_has_an_independent_meta_input_arm() -> None:
    for phase in PHASE_STATE_FEATURES:
        assert ARM_FEATURES[f"M17p_{phase}"] == [
            "meta_aux_good_trade_oof",
            f"meta_aux_{phase}",
        ]
    assert ARM_FEATURES["M13_good_trade_plus_expected_hit_surprise"] == [
        "meta_aux_good_trade_oof",
        "meta_resid_arch_expected_hit_surprise",
    ]
    assert ARM_FEATURES[
        "M14_good_trade_negative_residual_plus_expected_hit_surprise"
    ] == [
        "meta_aux_good_trade_oof",
        "meta_aux_negative_residual_oof",
        "meta_resid_arch_expected_hit_surprise",
    ]


def test_m8q_arms_test_each_quality_probability_incrementally() -> None:
    arms = {
        name: values
        for name, values in ARM_FEATURES.items()
        if name.startswith("M8q_")
    }
    assert set(arms) == {f"M8q_{target}" for target in EXECUTABLE_FAILURE_TARGETS}
    for target in EXECUTABLE_FAILURE_TARGETS:
        assert arms[f"M8q_{target}"] == [
            "meta_aux_good_trade_oof",
            f"{RESIDUAL_EVENT_PREFIX}expected_{target}",
        ]


def test_m9s_arms_test_direct_residual_mechanisms_incrementally() -> None:
    arms = {
        name: values
        for name, values in ARM_FEATURES.items()
        if name.startswith("M9s_")
    }
    assert set(arms) == {f"M9s_{mechanism}" for mechanism in DIRECT_RESIDUAL_MECHANISMS}
    for mechanism in DIRECT_RESIDUAL_MECHANISMS:
        assert arms[f"M9s_{mechanism}"] == [
            "meta_aux_good_trade_oof",
            f"meta_aux_{mechanism}_oof",
        ]


def test_m10g_arms_only_expose_oof_reliable_mechanisms() -> None:
    arms = {name: values for name, values in ARM_FEATURES.items() if name.startswith("M10g_")}
    assert set(arms) == {f"M10g_{mechanism}" for mechanism in RELIABILITY_GATED_MECHANISMS}
    for mechanism in RELIABILITY_GATED_MECHANISMS:
        assert arms[f"M10g_{mechanism}"] == [
            "meta_aux_good_trade_oof",
            f"meta_aux_{mechanism}_reliability_gated_oof",
        ]


def test_oof_reliability_gate_neutralizes_nonpredictive_side_archetype_streams() -> None:
    rows = 700
    side = np.where(np.arange(rows) < rows // 2, "short", "long")
    archetype = np.where(side == "short", "short_mixed", "long_breakout")
    probability = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    ev = np.where(side == "short", 0.005 - 0.04 * probability, 0.002).astype(np.float32)
    train = pd.DataFrame(
        {
            "side_name": side,
            "archetype_policy_key": archetype,
            "ev_after_1pct": ev,
        }
    )
    test = train.iloc[:10].copy()
    out_train, out_test, detail = _gate_auxiliary_by_oof_reliability(
        train, test, probability, probability[:10]
    )
    assert np.any(out_train[side == "short"] > 0.0)
    assert np.allclose(out_train[side == "long"], 0.0)
    assert np.any(out_test > 0.0)
    assert any(row["active"] for row in detail)


def test_causal_residual_shortfall_is_nonnegative_and_does_not_label_first_block() -> None:
    train = _frame(480).reset_index(drop=True)
    train["score"] = train["score_meta_base_soft_label"]
    local, target = _causal_residual_shortfall_target(train, clip=0.03)
    assert 0 < len(local) < len(train)
    assert len(local) == len(target)
    assert np.all(target >= 0.0)
    assert np.all(target <= 0.03)


def test_oof_input_contract_route_requires_ev_gain_and_path_stability() -> None:
    # The selector is top-k *within each timestamp*.  Build a realistic
    # cross-sectional auction where M1 puts weak names in the selected tail
    # and M12 puts the strong names there.
    bars = 24
    candidates_per_bar = 10
    rows = bars * candidates_per_bar
    rank = np.tile(np.arange(candidates_per_bar), bars)
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=bars, freq="h", tz="UTC").repeat(candidates_per_bar),
            "__symbol__": [f"S{i}" for i in rank],
            "side_name": ["short"] * rows,
            "archetype_policy_key": ["short_mixed"] * rows,
            "score": rank.astype(float),
            # The bottom half has weak EV and the top half strong EV.
            "ev_after_1pct": np.where(rank < 5, 0.001, 0.02),
            "full_path_bad_mae_1r": 0.0,
        }
    )
    m1 = (9 - rank).astype(np.float32)
    m12 = rank.astype(np.float32)
    active, details = _select_oof_feature_contract_groups(
        frame, m1, m12, min_selected_rows=20, min_ev_improvement=0.001
    )
    assert active == {("short", "short_mixed")}
    assert details[0]["active"] is True


def test_contrastive_failure_target_matches_score_bands_without_raw_score_bias() -> None:
    rows = 160
    score = np.tile(np.linspace(0.91, 0.99, 8, dtype=np.float32), rows // 8)
    # Every second score point has both a bad unexpected loss and a clean
    # executable control.  The contrast must retain both classes and balance
    # their training weight within the local score bands.
    target_loss = np.arange(rows) % 2 == 0
    frame = pd.DataFrame(
        {
            "side_name": "short",
            "archetype_policy_key": "short_mixed",
            "base_rank_pct_by_timestamp": score,
            "score": score,
            "ev_after_1pct": np.where(target_loss, -0.01, 0.02),
            "clean_exec": np.where(target_loss, 0.0, 1.0),
            "full_path_bad_mae_1r": 0.0,
            "timeout": 0.0,
            "dirty_positive": 0.0,
        }
    )
    local, target, weight = _contrastive_executable_failure_training_set(
        frame, target_loss.astype(np.float32)
    )
    assert len(local) == rows
    assert set(np.unique(target)) == {0.0, 1.0}
    assert np.isclose(weight[target > 0.5].sum(), weight[target <= 0.5].sum(), rtol=0.05)


def test_phase_context_is_outcome_free_and_timestamp_side_order_invariant() -> None:
    timestamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    rows = pd.DataFrame(
        {
            "__ts__": timestamps.repeat(2),
            "side_name": "short",
            "mkt_median_oi_chg_4h_rz": np.repeat([-1.0, -1.5, -0.4, 0.2], 2),
            "mkt_pct_oi_chg_4h_rz_lt_minus1": np.repeat([0.1, 0.4, 0.3, 0.1], 2),
            "mkt_pct_price_down_oi_down_4h": np.repeat([0.1, 0.3, 0.2, 0.0], 2),
            "market_breadth_chg_1h": np.repeat([-0.1, -0.3, 0.1, 0.2], 2),
            "mkt_systemic_deleveraging_score": np.repeat([0.1, 0.8, 0.5, 0.1], 2),
            "mkt_flush_exhaustion_score": np.repeat([0.0, 0.2, 0.8, 0.5], 2),
            "mkt_oi_flush_breadth_recovery_4h": np.repeat([0.0, 0.0, 0.3, 0.5], 2),
            "market_breadth_recovery_from_6h_min": np.repeat([0.0, 0.0, 0.2, 0.5], 2),
            "mkt_pct_price_up_oi_down_1h": np.repeat([0.0, 0.0, 0.3, 0.5], 2),
            "market_pc1_variance_share_12h": np.repeat([0.2, 0.7, 0.5, 0.2], 2),
            "ev_after_1pct": np.linspace(-0.2, 0.2, 8),
            "clean_exec": [0, 1] * 4,
        }
    )
    train, test = rows.iloc[:4].copy(), rows.iloc[4:].copy()
    first_train, first_test, detail = _causal_phase_state_context(train, test)
    mutated = rows.copy()
    mutated["ev_after_1pct"] *= -100.0
    mutated["clean_exec"] = 1 - mutated["clean_exec"]
    second_train, second_test, _ = _causal_phase_state_context(
        mutated.iloc[:4].copy(), mutated.iloc[4:].copy()
    )
    assert detail["status"] == "complete"
    assert first_train.columns.tolist() == second_train.columns.tolist()
    np.testing.assert_allclose(first_train, second_train)
    np.testing.assert_allclose(first_test, second_test)
    # Both symbols at one timestamp receive the same market-state context.
    np.testing.assert_allclose(first_test.iloc[0], first_test.iloc[1])


def test_direct_residual_mechanisms_keep_demotion_promotion_and_systemic_labels_distinct() -> None:
    local = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01T00:00:00Z"] * 4 + ["2026-01-01T01:00:00Z"] * 2
            ),
            "side_name": ["short"] * 6,
            "base_rank_pct_by_timestamp": [0.95, 0.94, 0.93, 0.92, 0.85, 0.84],
            "ev_after_1pct": [-0.02, -0.01, -0.03, -0.02, 0.03, -0.01],
            "clean_exec": [1, 0, 1, 0, 1, 1],
            "full_path_bad_mae_1r": [0, 1, 0, 1, 0, 0],
            "timeout": [0, 0, 1, 0, 0, 0],
        }
    )
    negative = np.array([1, 1, 1, 1, 0, 0], dtype=np.float32)
    unexpected_loss = _direct_residual_mechanism_target(
        local, negative, "top_tail_residual_negative_ev"
    )
    fragile = _direct_residual_mechanism_target(
        local, negative, "top_tail_residual_clean_cost_fragile"
    )
    systemic = _direct_residual_mechanism_target(
        local, negative, "top_tail_residual_systemic_loss"
    )
    promote = _direct_residual_mechanism_target(
        local, negative, "near_tail_clean_executable"
    )
    np.testing.assert_allclose(unexpected_loss, [1, 1, 1, 1, 0, 0])
    np.testing.assert_allclose(fragile, [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(systemic, [1, 1, 1, 1, 0, 0])
    np.testing.assert_allclose(promote, [0, 0, 0, 0, 1, 0])


def test_direct_residual_episode_labels_require_persistent_local_or_market_failures() -> None:
    rows = 12
    local = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC"),
            "side_name": ["short"] * rows,
            "archetype_policy_key": ["short_mixed"] * 6 + ["short_default"] * 6,
            "base_rank_pct_by_timestamp": [0.95] * rows,
            "ev_after_1pct": [-0.01] * 5 + [0.01] + [-0.01] + [0.01] * 5,
            "clean_exec": [0] * rows,
            "full_path_bad_mae_1r": [1] * rows,
            "timeout": [0] * rows,
        }
    )
    negative = np.ones(rows, dtype=np.float32)
    local_episode = _direct_residual_mechanism_target(
        local, negative, "top_tail_residual_local_loss_episode_6h"
    )
    market_episode = _direct_residual_mechanism_target(
        local, negative, "top_tail_residual_market_loss_episode_6h"
    )
    # The mixed archetype has five losses in its 6h bucket and qualifies; the
    # default archetype has one loss and remains an isolated row-level event.
    np.testing.assert_allclose(local_episode, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # The six-hour side-wide batch has six losses out of twelve and qualifies.
    np.testing.assert_allclose(market_episode, [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])


def test_direct_residual_episode_onsets_mark_only_the_start_of_a_persistent_run() -> None:
    rows = 24
    local = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01T00:00:00Z"] * 12
                + ["2026-01-01T06:00:00Z"] * 12
            ),
            "side_name": ["short"] * rows,
            "archetype_policy_key": ["short_mixed"] * 6
            + ["short_default"] * 6
            + ["short_mixed"] * 6
            + ["short_default"] * 6,
            "base_rank_pct_by_timestamp": [0.95] * rows,
            "ev_after_1pct": [-0.01] * rows,
            "clean_exec": [0] * rows,
            "full_path_bad_mae_1r": [1] * rows,
            "timeout": [0] * rows,
        }
    )
    negative = np.ones(rows, dtype=np.float32)
    local_onset = _direct_residual_mechanism_target(
        local, negative, "top_tail_residual_local_loss_episode_onset_6h"
    )
    market_onset = _direct_residual_mechanism_target(
        local, negative, "top_tail_residual_market_loss_episode_onset_6h"
    )
    np.testing.assert_allclose(local_onset, [1] * 12 + [0] * 12)
    np.testing.assert_allclose(market_onset, [1] * 12 + [0] * 12)


def test_candidate_transition_state_onsets_label_all_candidates_in_active_bucket() -> None:
    rows = 24
    local = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01T00:00:00Z"] * 12
                + ["2026-01-01T06:00:00Z"] * 12
            ),
            "side_name": ["short"] * rows,
            "archetype_policy_key": ["short_mixed"] * 6
            + ["short_default"] * 6
            + ["short_mixed"] * 6
            + ["short_default"] * 6,
            "base_rank_pct_by_timestamp": [0.95] * rows,
            # Both archetypes have an onset in the first bucket.  The second
            # bucket remains bad but is the same contiguous state, not a new
            # transition.
            "ev_after_1pct": [-0.01] * 10 + [0.01] * 2 + [-0.01] * 10 + [0.01] * 2,
            "clean_exec": [0] * rows,
            "full_path_bad_mae_1r": [1] * rows,
            "timeout": [0] * rows,
        }
    )
    negative = np.ones(rows, dtype=np.float32)
    local_state = _direct_residual_mechanism_target(
        local, negative, "candidate_top20_local_loss_state_onset_6h"
    )
    side_state = _direct_residual_mechanism_target(
        local, negative, "candidate_top20_side_loss_state_onset_6h"
    )
    contagion_state = _direct_residual_mechanism_target(
        local, negative, "candidate_top20_cross_archetype_loss_state_onset_6h"
    )

    # State labels cover every candidate in the onset bucket, including the
    # two rows that did not themselves lose.  This differs deliberately from
    # a first-touch or row-level false-positive target.
    np.testing.assert_allclose(local_state, [1] * 12 + [0] * 12)
    np.testing.assert_allclose(side_state, [1] * 12 + [0] * 12)
    np.testing.assert_allclose(contagion_state, [1] * 12 + [0] * 12)


def test_candidate_transition_activation_is_not_diluted_by_below_cutoff_rows() -> None:
    candidates = 8
    non_candidates = 24
    local = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * (candidates + non_candidates)),
            "side_name": ["long"] * (candidates + non_candidates),
            "archetype_policy_key": ["long_breakout"] * (candidates + non_candidates),
            "base_rank_pct_by_timestamp": [0.95] * candidates + [0.40] * non_candidates,
            "ev_after_1pct": [-0.01] * 4 + [0.01] * 4 + [0.01] * non_candidates,
            "clean_exec": [0] * 4 + [1] * (candidates - 4 + non_candidates),
            "full_path_bad_mae_1r": [0] * (candidates + non_candidates),
            "timeout": [0] * (candidates + non_candidates),
        }
    )
    residual = np.array([1] * 4 + [0] * (candidates - 4 + non_candidates), dtype=np.float32)
    target = _direct_residual_mechanism_target(
        local, residual, "candidate_top20_local_loss_state_onset_6h"
    )

    # Candidate loss rate is 4/8, above the 35% activation threshold.  The
    # 24 rows below the fixed handoff rank cannot dilute it to 4/32.
    np.testing.assert_allclose(target[:candidates], [1] * candidates)
    np.testing.assert_allclose(target[candidates:], [0] * non_candidates)


def test_direct_residual_path_survivor_and_idiosyncratic_labels_are_distinct() -> None:
    local = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 6),
            "side_name": ["long"] * 6,
            "archetype_policy_key": ["long_mixed"] * 6,
            "base_rank_pct_by_timestamp": [0.95] * 6,
            "ev_after_1pct": [0.01, 0.01, 0.01, 0.01, 0.01, -0.02],
            "clean_exec": [1, 1, 1, 1, 1, 0],
            "full_path_bad_mae_1r": [1, 0, 0, 0, 0, 1],
            "timeout": [0, 1, 0, 0, 0, 0],
            "dirty_positive": [0, 0, 1, 0, 0, 0],
        }
    )
    negative = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    adverse_survivor = _direct_residual_mechanism_target(
        local, negative, "top_tail_adverse_path_survivor"
    )
    timeout_survivor = _direct_residual_mechanism_target(
        local, negative, "top_tail_timeout_positive_survivor"
    )
    dirty_survivor = _direct_residual_mechanism_target(
        local, negative, "top_tail_dirty_positive_survivor"
    )
    idiosyncratic = _direct_residual_mechanism_target(
        local, negative, "top_tail_residual_idiosyncratic_loss"
    )
    # Timeout is an adverse executable path even without a bad-MAE breach.
    np.testing.assert_allclose(adverse_survivor, [1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(timeout_survivor, [0, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(dirty_survivor, [0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(idiosyncratic, [0, 0, 0, 0, 0, 1])


def test_candidate_top20_failure_labels_do_not_require_the_top10_cutoff() -> None:
    """Conditional heads label failures inside the candidate population."""

    local = pd.DataFrame(
        {
            "base_rank_pct_by_timestamp": [0.85, 0.95, 0.75],
            "ev_after_1pct": [-0.01, 0.01, -0.02],
            "clean_exec": [0.0, 1.0, 0.0],
            "timeout": [0.0, 0.0, 0.0],
            "full_path_bad_mae_1r": [1.0, 0.0, 1.0],
            "dirty_positive": [0.0, 0.0, 0.0],
            "side_name": ["short", "short", "short"],
            "archetype_policy_key": ["short_default"] * 3,
            "__ts__": pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC"),
        }
    )
    negative = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    candidate_negative = _direct_residual_mechanism_target(
        local, negative, "candidate_top20_residual_negative_ev"
    )
    candidate_adverse = _direct_residual_mechanism_target(
        local, negative, "candidate_top20_residual_adverse_loss"
    )

    np.testing.assert_allclose(candidate_negative, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(candidate_adverse, [1.0, 0.0, 0.0])


def test_candidate_cross_archetype_contagion_requires_multiple_active_cells() -> None:
    """Contagion must not relabel an isolated archetype failure as systemic."""

    rows = []
    for archetype, failures in (
        ("short_default", [1, 1, 0]),
        ("short_mixed", [1, 1, 0]),
        ("short_breakout", [0, 0, 0]),
    ):
        for index, failure in enumerate(failures):
            rows.append(
                {
                    "__ts__": pd.Timestamp("2026-07-01T00:00:00Z"),
                    "side_name": "short",
                    "archetype_policy_key": archetype,
                    "base_rank_pct_by_timestamp": 0.85,
                    "ev_after_1pct": -0.01 if failure else 0.01,
                    "clean_exec": 0.0 if failure else 1.0,
                    "full_path_bad_mae_1r": 1.0 if failure else 0.0,
                    "timeout": 0.0,
                    "dirty_positive": 0.0,
                }
            )
    local = pd.DataFrame(rows)
    negative = (local["ev_after_1pct"].to_numpy() <= 0.0).astype(np.float32)

    contagion = _direct_residual_mechanism_target(
        local, negative, "candidate_top20_cross_archetype_loss_contagion"
    )

    # Both supported failing archetypes are in the shared state; the clean
    # archetype is not, although it sees the same timestamp-side context.
    np.testing.assert_allclose(contagion, [1, 1, 0, 1, 1, 0, 0, 0, 0])


def test_candidate_first_touch_stop_is_distinct_from_full_path_damage() -> None:
    """The stop-state target must use the executable first touch, not MAE."""

    local = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-01T00:00:00Z"], utc=True
            ),
            "side_name": ["short", "short"],
            "archetype_policy_key": ["short_default", "short_default"],
            "base_rank_pct_by_timestamp": [0.85, 0.85],
            "ev_after_1pct": [-0.01, -0.01],
            "clean_exec": [0.0, 0.0],
            # The first row is stopped immediately but has no later full-path
            # MAE; the second has later path damage but no first-touch stop.
            "first_touch_bad_mae_1r": [1.0, 0.0],
            "full_path_bad_mae_1r": [0.0, 1.0],
            "timeout": [0.0, 0.0],
            "dirty_positive": [0.0, 0.0],
        }
    )
    negative = np.ones(len(local), dtype=np.float32)
    first_touch = _direct_residual_mechanism_target(
        local, negative, "candidate_top20_residual_first_touch_stop_loss"
    )
    adverse = _direct_residual_mechanism_target(
        local, negative, "candidate_top20_residual_adverse_loss"
    )
    np.testing.assert_allclose(first_touch, [1, 0])
    np.testing.assert_allclose(adverse, [0, 1])


def test_new_candidate_failure_states_are_oof_reliability_gated() -> None:
    """Rare stop/timeout/episode heads must not be global forced inputs."""

    expected = {
        "candidate_top20_residual_first_touch_stop_loss",
        "candidate_top20_residual_timeout_loss",
        "candidate_top20_residual_systemic_loss",
        "candidate_top20_residual_local_adverse_episode_onset_6h",
    }
    assert expected.issubset(DIRECT_RESIDUAL_MECHANISMS)
    assert expected.issubset(RELIABILITY_GATED_MECHANISMS)
    for mechanism in expected:
        arm = f"M10g_{mechanism}"
        assert ARM_FEATURES[arm] == [
            "meta_aux_good_trade_oof",
            f"meta_aux_{mechanism}_reliability_gated_oof",
        ]


def test_candidate_stop_contagion_requires_two_archetypes_not_three_rows_per_cell() -> None:
    """Shared stop states must be reachable in a top-20 candidate stream."""

    local = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-07-01T00:00:00Z"] * 3, utc=True),
            "side_name": ["short"] * 3,
            "archetype_policy_key": ["short_default", "short_mixed", "short_breakout"],
            "base_rank_pct_by_timestamp": [0.85, 0.85, 0.85],
            "ev_after_1pct": [-0.01, -0.01, 0.01],
            "clean_exec": [0.0, 0.0, 1.0],
            "first_touch_bad_mae_1r": [1.0, 1.0, 0.0],
            "full_path_bad_mae_1r": [1.0, 1.0, 0.0],
            "timeout": [0.0, 0.0, 0.0],
            "dirty_positive": [0.0, 0.0, 0.0],
        }
    )
    target = _direct_residual_mechanism_target(
        local,
        np.ones(len(local), dtype=np.float32),
        "candidate_top20_cross_archetype_stop_contagion",
    )
    np.testing.assert_allclose(target, [1.0, 1.0, 0.0])


def test_candidate_reversal_target_uses_candidate_tail_not_only_top10() -> None:
    local = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-07-01T00:00:00Z"], utc=True),
            "side_name": ["long"],
            "archetype_policy_key": ["long_mixed"],
            "base_rank_pct_by_timestamp": [0.85],
            "ev_after_1pct": [-0.01],
            "clean_exec": [1.0],
            "first_touch_bad_mae_1r": [0.0],
            "full_path_bad_mae_1r": [1.0],
            "timeout": [0.0],
            "dirty_positive": [1.0],
        }
    )
    target = _direct_residual_mechanism_target(
        local,
        np.ones(len(local), dtype=np.float32),
        "candidate_top20_reversal_after_initial_success",
    )
    np.testing.assert_allclose(target, [1.0])


def test_direct_mechanism_support_is_train_only_and_candidate_conditioned() -> None:
    rows = 480
    local = _frame(rows)
    local["score"] = local["score_meta_base_soft_label"].astype(np.float32)
    local["base_rank_pct_by_timestamp"] = np.where(
        np.arange(rows) % 3,
        0.85,
        0.70,
    ).astype(np.float32)
    # Make the candidate stream contain causal negative-EV outcomes without
    # adding any outcome-derived value to the eventual model input.
    candidate = local["base_rank_pct_by_timestamp"].ge(0.80)
    local.loc[candidate, "ev_after_1pct"] = -0.01
    local.loc[candidate, "clean_exec"] = 0.0
    local.loc[candidate, "first_touch_bad_mae_1r"] = 1.0

    rows_out = _direct_mechanism_support_rows(
        local,
        fold_month="2026-04",
        mechanism="candidate_top20_residual_first_touch_stop_loss",
    )

    report = pd.DataFrame(rows_out)
    assert {"global", "side", "side_archetype"}.issubset(report["level"])
    global_row = report.loc[report["level"].eq("global")].iloc[0]
    # The helper must apply its own top-20 condition rather than let the
    # caller's full frame define the fitting population.
    assert int(global_row["rows"]) < len(local)
    assert int(global_row["positive_rows"]) > 0
    assert global_row["fold_month"] == "2026-04"


def test_compact_mechanism_support_loader_reads_no_model_matrix(tmp_path) -> None:
    period = pd.Period("2026-03", freq="M")
    timestamps = pd.date_range("2026-03-01", periods=4, freq="h", tz="UTC")
    ledger = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["A/USD:USD", "B/USD:USD"] * 2,
            "side_name": ["long", "long", "short", "short"],
            "archetype_policy_key": ["long_default", "long_default", "short_default", "short_default"],
            "score": [0.1, 0.9, 0.2, 0.8],
            "ev_after_1pct": [0.01, -0.01, 0.01, -0.01],
            "clean_exec": [1.0, 0.0, 1.0, 0.0],
            "dirty_positive": [0.0] * 4,
            "full_path_bad_mae_1r": [0.0, 1.0, 0.0, 1.0],
            "timeout": [0.0] * 4,
            # This must not be read by the compact support loader.
            "very_wide_model_feature": [99.0] * 4,
        }
    )
    ledger_path = tmp_path / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    for side in ("long", "short"):
        label = ledger.loc[ledger["side_name"].eq(side), ["__ts__", "__symbol__", "side_name"]].copy()
        label["__first_touch_mae_to_sl__"] = [0.0, 1.0]
        label.to_parquet(
            labels_root / f"train_global_{side}_5_2026_03.parquet",
            index=False,
        )

    loaded, coverage = _load_saved_direct_mechanism_support_ledgers(
        [ledger_path], labels_root=labels_root, months=[period]
    )

    assert "very_wide_model_feature" not in loaded
    assert int(coverage["2026-03"]) == 4
    assert loaded["first_touch_bad_mae_1r"].sum() == 2.0
    assert loaded["base_rank_pct_by_timestamp"].between(0.0, 1.0).all()


def test_extended_residual_state_ablation_rejects_missing_history_months() -> None:
    with pytest.raises(RuntimeError, match="2025-12"):
        _require_month_coverage(
            {"2026-01": 10, "2026-02": 10},
            list(pd.period_range("2025-12", "2026-02", freq="M")),
            context="test_source",
        )


def test_saved_full_loader_samples_only_train_months_before_wide_feature_read(tmp_path) -> None:
    """Full ledgers retain all OOS top-tail rows but bound prior-month width."""

    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    ledger_rows: list[dict[str, object]] = []
    for period in pd.period_range("2026-01", "2026-02", freq="M"):
        side_labels = {"long": [], "short": []}
        for timestamp in pd.date_range(period.start_time, periods=2, freq="h", tz="UTC"):
            for index in range(5):
                side = "long" if index % 2 else "short"
                symbol = f"S{index}"
                ledger_rows.append(
                    {
                        "__ts__": timestamp,
                        "__symbol__": symbol,
                        "side_name": side,
                        "archetype_policy_key": f"{side}_default",
                        "score": float(index),
                        "ev_after_1pct": 0.01,
                        "clean_exec": 1.0,
                        "dirty_positive": 0.0,
                        "full_path_bad_mae_1r": 0.0,
                        "timeout": 0.0,
                        "base_margin_to_cutoff": 0.1,
                        "base_margin_to_cutoff_z": 0.1,
                        "base_signal_zscore_within_archetype": 0.1,
                        "wide_feature": float(index),
                    }
                )
                side_labels[side].append(
                    {
                        "__ts__": timestamp,
                        "__symbol__": symbol,
                        "side_name": side,
                        "__first_touch_target_soft__": 0.75,
                        "__first_touch_mae_to_sl__": 0.0,
                    }
                )
        for side, rows in side_labels.items():
            pd.DataFrame(rows).to_parquet(
                labels_root / f"train_global_{side}_5_{period.year}_{period.month:02d}.parquet",
                index=False,
            )
    ledger_path = tmp_path / "wide_ledger.parquet"
    pd.DataFrame(ledger_rows).to_parquet(ledger_path, index=False)

    loaded, coverage = _load_saved_full_feature_ledgers(
        [ledger_path],
        labels_root=labels_root,
        months=list(pd.period_range("2026-01", "2026-02", freq="M")),
        features_by_side={"long": ["wide_feature"], "short": ["wide_feature"]},
        full_months={"2026-02"},
        max_rows_per_train_month=2,
    )

    counts = loaded.groupby(loaded["__ts__"].dt.to_period("M"), observed=True).size()
    assert counts.loc[pd.Period("2026-01", freq="M")] == 2
    assert counts.loc[pd.Period("2026-02", freq="M")] == 4
    assert coverage == {"2026-01": 2, "2026-02": 4}
    # Rank is calculated before sampling.  A post-sampling recomputation would
    # assign every retained row a rank of one and invalidate top-tail targets.
    observed_ranks = loaded.loc[
        loaded["__ts__"].dt.to_period("M").eq(pd.Period("2026-02", freq="M")),
        "base_rank_pct_by_timestamp",
    ].to_numpy(dtype=float)
    assert np.all(np.isclose(np.sort(np.unique(observed_ranks)), np.array([0.8, 1.0])))


def test_static_handoff_streaming_keeps_full_oos_and_bounds_train_months(tmp_path) -> None:
    """Historical static hydration may sample train months, never OOS months."""

    handoff_rows = []
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    for period in pd.period_range("2026-01", "2026-03", freq="M"):
        ts = pd.date_range(period.start_time, periods=6, freq="h", tz="UTC")
        label_rows = {"long": [], "short": []}
        for side in ("long", "short"):
            for index, value in enumerate(ts):
                handoff_rows.append(
                    {
                        "__ts__": value,
                        "__symbol__": f"{side}_{index}",
                        "side_name": side,
                        "archetype_policy_key": f"{side}_default",
                        "score": float(index) / 10.0,
                        "selected_top30": True,
                    }
                )
                label_rows[side].append(
                    {
                        "__ts__": value,
                        "__symbol__": f"{side}_{index}",
                        "side_name": side,
                        "__u_policy_net__": 0.01,
                        "__first_touch_target_soft__": 0.8,
                        "__long_path_clean_exec_label__": 1.0,
                        "__long_path_dirty_positive_label__": 0.0,
                        "__path_full_bad_mae_1r__": 0.0,
                        "__first_touch_mae_to_sl__": 0.0,
                        "__first_touch_timeout__": 0.0,
                        "__is_timeout__": 0.0,
                        "__first_touch_stop__": 0.0,
                    }
                )
        for side, rows in label_rows.items():
            pd.DataFrame(rows).to_parquet(
                labels_root / f"train_global_{side}_5_{period.year}_{period.month:02d}.parquet",
                index=False,
            )
    handoff_path = tmp_path / "handoff.parquet"
    pd.DataFrame(handoff_rows).to_parquet(handoff_path, index=False)

    loaded, _, coverage = _load_handoff_with_labels(
        handoff_path,
        labels_root,
        list(pd.period_range("2026-01", "2026-03", freq="M")),
        full_months={"2026-03"},
        max_rows_per_train_month=3,
    )
    counts = loaded.groupby(loaded["__ts__"].dt.to_period("M"), observed=True).size()
    assert counts.loc[pd.Period("2026-01", freq="M")] <= 3
    assert counts.loc[pd.Period("2026-02", freq="M")] <= 3
    assert counts.loc[pd.Period("2026-03", freq="M")] == 12
    assert coverage["2026-03"] == 12


def test_static_handoff_normalizes_legacy_score_base_alias(tmp_path) -> None:
    """Older expanded-pool handoffs expose the base score as ``score_base``."""

    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    period = pd.Period("2026-03", freq="M")
    ts = pd.Timestamp("2026-03-12T00:00:00Z")
    handoff = pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "archetype_policy_key": ["long_default"],
            "score_base": [0.73],
            "selected_top30": [True],
        }
    )
    handoff_path = tmp_path / "legacy_handoff.parquet"
    handoff.to_parquet(handoff_path, index=False)
    label = pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "__u_policy_net__": [0.01],
            "__first_touch_target_soft__": [0.8],
            "__long_path_clean_exec_label__": [1.0],
            "__long_path_dirty_positive_label__": [0.0],
            "__path_full_bad_mae_1r__": [0.0],
            "__first_touch_mae_to_sl__": [0.0],
            "__first_touch_timeout__": [0.0],
            "__is_timeout__": [0.0],
            "__first_touch_stop__": [0.0],
        }
    )
    label.to_parquet(labels_root / "train_global_long_5_2026_03.parquet", index=False)
    label.to_parquet(labels_root / "train_global_short_5_2026_03.parquet", index=False)

    loaded, _, coverage = _load_handoff_with_labels(
        handoff_path,
        labels_root,
        [period],
        full_months={"2026-03"},
    )
    assert "score" in loaded
    assert "score_base" not in loaded
    assert float(loaded["score"].iloc[0]) == pytest.approx(0.73)
    assert coverage == {"2026-03": 1}


def test_named_side_archetype_failure_labels_do_not_depend_on_ood_features() -> None:
    local = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "side_name": ["long", "short", "short", "long"],
            "archetype_policy_key": [
                "long_mixed_wideslow_tentative",
                "short_mixed_clean_path",
                "short_default_clean_path",
                "long_breakout_diagnostic_candidate",
            ],
            "base_rank_pct_by_timestamp": [0.95] * 4,
            "ev_after_1pct": [-0.01] * 4,
            "clean_exec": [1, 0, 1, 0],
            "full_path_bad_mae_1r": [0, 1, 0, 1],
            "timeout": [0, 0, 1, 0],
            # Deliberately contradictory OOD values. These must not alter
            # outcome-label construction; they are classifier inputs only.
            "gmm_ood_score": [0.0, 99.0, 0.0, 99.0],
        }
    )
    negative = np.ones(len(local), dtype=np.float32)
    np.testing.assert_allclose(
        _direct_residual_mechanism_target(local, negative, "long_mixed_latent_misfire"),
        [1, 0, 0, 0],
    )
    np.testing.assert_allclose(
        _direct_residual_mechanism_target(local, negative, "short_mixed_off_manifold"),
        [0, 1, 0, 0],
    )
    np.testing.assert_allclose(
        _direct_residual_mechanism_target(local, negative, "short_default_latent_uncertainty"),
        [0, 0, 1, 0],
    )


def test_timestamp_neutral_surprise_uses_leave_one_out_peers() -> None:
    train = _frame()
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(train)
    expectation = ScoreExpectationState(cfg).fit(train)
    labelled = add_residual_event_targets(
        train, threshold_state=thresholds, expectation_state=expectation
    )

    first_ts = labelled["__ts__"].iloc[0]
    group = labelled.loc[labelled["__ts__"].eq(first_ts)]
    residual = group["resid_event_global_surprise"].to_numpy(dtype=float)
    neutral = group["resid_event_timestamp_neutral_surprise"].to_numpy(dtype=float)
    expected = residual - (residual.sum() - residual) / (len(residual) - 1)
    np.testing.assert_allclose(neutral, expected, atol=1e-6)
    np.testing.assert_allclose(
        group["resid_event_market_peer_surprise"].to_numpy(dtype=float),
        (residual.sum() - residual) / (len(residual) - 1),
        atol=1e-6,
    )


def test_frozen_hit_probability_is_the_residual_expectation() -> None:
    train = _frame()
    train["hit_probability"] = np.linspace(0.2, 0.8, len(train), dtype=np.float32)
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(train)
    expectation = ScoreExpectationState(cfg).fit(train)
    labelled = add_residual_event_targets(
        train, threshold_state=thresholds, expectation_state=expectation
    )
    np.testing.assert_allclose(
        labelled["resid_event_expected_hit"], train["hit_probability"], atol=1e-7
    )


def test_ev_expectation_is_train_fitted_and_not_replaced_by_hit_probability() -> None:
    train = _frame()
    train["hit_probability"] = np.linspace(0.95, 0.05, len(train), dtype=np.float32)
    cfg = _config()
    ev_state = ScoreExpectationState(
        cfg, target_col=cfg.ev_col, direct_col=""
    ).fit(train)
    expected = ev_state.transform(train)
    assert not np.allclose(expected.to_numpy(), train["hit_probability"].to_numpy())
    assert expected.min() < 0.0 < expected.max()


def test_oos_transform_rejects_outcomes_and_allows_preentry_rows() -> None:
    train = _frame()
    cfg = _config()
    state = ResidualEventArchetypeState(cfg).fit(
        train, candidate_features=["directional_feature"]
    )
    with pytest.raises(ValueError, match="outcome columns"):
        state.transform_oos(train)

    safe = train.drop(
        columns=[
            "clean_exec",
            "ev_after_1pct",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "stop_or_adverse",
        ]
    )
    out = state.transform_oos(safe)
    assert out.index.equals(train.index)
    assert out.columns.str.startswith(
        ("resid_event_aegmm_", "resid_event_market_aegmm_")
    ).all()


def test_assessment_smoother_excludes_current_day_outcomes() -> None:
    frame = _frame(120)
    cfg = _config()
    frame["resid_event_top10_population"] = 1
    overlay = causal_eight_day_hit_rate_overlay(frame, config=cfg, embargo_hours=0.0)
    first_day = frame["__ts__"].iloc[0].floor("D")
    assert (
        overlay.loc[
            frame["__ts__"].dt.floor("D").eq(first_day), "assessment_hr8_surprise"
        ]
        .isna()
        .all()
    )
    second_day = first_day + pd.Timedelta(days=1)
    assert (
        overlay.loc[
            frame["__ts__"].dt.floor("D").eq(second_day), "assessment_hr8_effective_n"
        ]
        .gt(0.0)
        .all()
    )


def test_event_baseline_marks_rows_without_cross_archetype_merge() -> None:
    train = _frame()
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(train)
    expectation = ScoreExpectationState(cfg).fit(train)
    raw = add_residual_event_targets(
        train, threshold_state=thresholds, expectation_state=expectation
    )
    baseline = ResidualEventBaselineState(cfg).fit(raw)
    labelled = add_residual_event_targets(
        train,
        threshold_state=thresholds,
        expectation_state=expectation,
        baseline_state=baseline,
    )
    assert len(labelled) == len(train)
    assert set(labelled["resid_event_class"].astype(str)).issubset(
        {
            "normal",
            "negative_residual_event",
            "adverse_path_event",
            "positive_residual_event",
            "favorable_near_miss_event",
            "high_variance_event",
        }
    )


def test_extreme_single_day_surprise_is_retained_without_prior_week() -> None:
    train = _frame(120)
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(train)
    expectation = ScoreExpectationState(cfg).fit(train)
    raw = add_residual_event_targets(
        train, threshold_state=thresholds, expectation_state=expectation
    )
    baseline = ResidualEventBaselineState(cfg).fit(raw.iloc[:60])
    # Force an acute local miss on a later day. The event must not require a
    # seven/eight-day outcome history to survive target construction.
    shocked = train.iloc[60:].copy()
    shocked["hit_probability"] = shocked["clean_exec"].astype(np.float32)
    local = shocked["side_name"].eq("short") & shocked["archetype_policy_key"].eq(
        "arch_a"
    )
    shocked.loc[local, "clean_exec"] = 0.0
    shocked.loc[local, "hit_probability"] = 1.0
    labelled = add_residual_event_targets(
        shocked,
        threshold_state=thresholds,
        expectation_state=expectation,
        baseline_state=baseline,
    )
    assert labelled["resid_event_large_event_strength"].max() > 0.0
    assert labelled["resid_event_persistent"].max() > 0.0


def test_positive_hit_surprise_with_negative_ev_is_adverse_event() -> None:
    train = _frame(600)
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(train)
    hit_expectation = ScoreExpectationState(cfg).fit(train)
    ev_expectation = ScoreExpectationState(
        cfg, target_col=cfg.ev_col, direct_col=""
    ).fit(train)
    raw = add_residual_event_targets(
        train,
        threshold_state=thresholds,
        expectation_state=hit_expectation,
        ev_expectation_state=ev_expectation,
    )
    baseline = ResidualEventBaselineState(cfg).fit(raw)
    shocked = train.iloc[-12:].copy()
    shocked["hit_probability"] = 0.0
    shocked["clean_exec"] = 1.0
    shocked["ev_after_1pct"] = -0.20
    labelled = add_residual_event_targets(
        shocked,
        threshold_state=thresholds,
        expectation_state=hit_expectation,
        ev_expectation_state=ev_expectation,
        baseline_state=baseline,
    )
    selected = labelled["resid_event_top10_population"].gt(0.5)
    assert labelled.loc[selected, "resid_event_global_surprise"].mean() > 0.0
    assert labelled.loc[selected, "resid_event_ev_global_surprise"].mean() < 0.0
    assert labelled.loc[selected, "resid_event_class"].astype(str).isin(
        ["negative_residual_event", "adverse_path_event"]
    ).any()


def test_lgbm_screen_can_be_explicitly_disabled() -> None:
    frame = _frame(1_800)
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(frame)
    expectation = ScoreExpectationState(cfg).fit(frame)
    raw = add_residual_event_targets(
        frame, threshold_state=thresholds, expectation_state=expectation
    )
    baseline = ResidualEventBaselineState(cfg).fit(raw)
    labelled = add_residual_event_targets(
        frame,
        threshold_state=thresholds,
        expectation_state=expectation,
        baseline_state=baseline,
    )
    labels = pd.Categorical(labelled["resid_event_class"]).codes.astype(np.int32)
    selected, metrics, meta = screen_local_residual_features(
        labelled,
        labels,
        ["directional_feature"],
        config=ResidualEventArchetypeConfig(
            **{
                **_config().__dict__,
                "lgbm_enabled": False,
                "max_features_after_mi": 1,
                "max_features_after_lgbm": 1,
            }
        ),
        seed=7,
    )
    assert selected == ["directional_feature"]
    assert float(metrics["lgbm_validation_gain"].max()) == 0.0
    assert meta["disabled"] == 1.0


def test_signed_autocorrelation_does_not_compress_calendar_gaps() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-04"], utc=True
            ),
            "side_name": "long",
            "archetype_policy_key": "mixed",
            "resid_event_timestamp_neutral_surprise": [-0.8, -0.6, -0.9],
        }
    )
    result = _surprise_autocorrelation(
        frame,
        ["side_name", "archetype_policy_key"],
        surprise_col="resid_event_timestamp_neutral_surprise",
        population="top10",
    )
    assert int(result.loc[0, "consecutive_pairs"]) == 1
    assert np.isclose(result.loc[0, "adverse_lag1_product_mean"], 0.48)
    assert np.isclose(result.loc[0, "favorable_lag1_product_mean"], 0.0)


def test_prediction_shard_contract_is_normalised_without_outcome_use() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-07-01", periods=3, freq="15min", tz="UTC"),
            "__symbol__": "BTC/USD:USD",
            "side_name": "long",
            "archetype_policy_key": ["existing", None, "missing"],
            "__archetype_policy_key__": "long_breakout",
            "score_meta_base_soft_label": [0.71, 0.82, 0.93],
            "clean_exec": [1.0, 0.0, 1.0],
        }
    )
    out, aliases = _normalise_candidate_contract(frame, score_col="score")
    np.testing.assert_allclose(out["score"], frame["score_meta_base_soft_label"])
    assert aliases["score"] == "score_meta_base_soft_label"
    assert aliases["archetype_policy_key"].startswith("coalesce:")
    assert out["archetype_policy_key"].tolist() == [
        "existing",
        "long_breakout",
        "long_breakout",
    ]
    assert int(out["selected_top30"].sum()) == len(out)


def test_candidate_column_contract_accepts_feature_names_json(tmp_path) -> None:
    path = tmp_path / "columns.json"
    path.write_text('{"feature_names": ["score", "market_state", "score"]}')
    assert _load_candidate_columns(path) == ["score", "market_state"]


def test_state_search_budget_parsers_are_explicit_and_validated() -> None:
    assert _parse_int_csv("3,4,3", name="clusters") == (3, 4)
    assert _parse_float_csv("0.0001,0.001", name="covars") == (0.0001, 0.001)
    assert _parse_choice_csv(
        "diag,tied,diag", name="covariance", allowed={"diag", "tied"}
    ) == ("diag", "tied")
    with pytest.raises(ValueError):
        _parse_int_csv("1", name="clusters")
    with pytest.raises(ValueError):
        _parse_choice_csv("full", name="covariance", allowed={"diag"})


def test_state_group_filter_is_strictly_side_by_archetype() -> None:
    assert _parse_state_group_filter("short::short_mixed") == (
        "short",
        "short_mixed",
    )
    assert _parse_state_group_filter("") is None
    with pytest.raises(ValueError):
        _parse_state_group_filter("short_only")
