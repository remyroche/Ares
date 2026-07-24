from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.report_meta_residual_sequential_calibration import sequential_calibrate
from scripts.run_train_meta_residual_archetype_enhancement import (
    FAILURE_PRESSURE_CONTEXT_FEATURES,
    HIT_SURPRISE_NUMERIC_FEATURES,
    OUTCOME_CONTEXT_FEATURES,
    _add_reference_fold_features,
    _append_meta_identity_features,
    _append_residual_state_v2_composites,
    _arm_candidate_features,
    _merge_residual_features,
    _parse_months,
    _residual_v2_sample_weight_multiplier,
    metrics_by_scope,
    surprise_calendar,
)


def test_metrics_decompose_one_global_timestamp_topk_selection() -> None:
    timestamps = pd.to_datetime(
        ["2026-04-01T00:00:00Z"] * 4 + ["2026-04-01T01:00:00Z"] * 4,
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": [f"S{idx}" for idx in range(8)],
            "calendar_month": "2026-04",
            "week_start": pd.Timestamp("2026-03-30", tz="UTC"),
            "side_name": ["long", "short", "short", "short"] * 2,
            "archetype_policy_key": ["long_a", "short_a", "short_b", "short_c"] * 2,
            # Alternative selects the one long globally at each timestamp.
            "score_alternative": [0.99, 0.80, 0.70, 0.60] * 2,
            # Reference selects short_a globally at each timestamp.
            "score_current_reference": [0.60, 0.99, 0.80, 0.70] * 2,
            "hit_prob_alternative": np.full(8, 0.5, dtype=np.float32),
            "hit_prob_current_reference": np.full(8, 0.5, dtype=np.float32),
            "ev_after_1pct": np.linspace(-0.01, 0.02, 8),
            "clean_exec": [1, 0, 0, 0] * 2,
            "dirty_positive": [0, 1, 1, 1] * 2,
            "first_touch_bad_mae_1r": [0, 1, 1, 1] * 2,
            "full_path_bad_mae_1r": [0, 1, 1, 1] * 2,
            "timeout": np.zeros(8, dtype=np.float32),
        }
    )
    metrics = metrics_by_scope(frame, "local_aegmm_all_three")
    overall = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq("local_aegmm_all_three")
    ].iloc[0]
    assert int(overall["selected_rows"]) == 2
    assert overall["selection_basis"] == "global_within_timestamp"
    side = metrics[
        metrics["scope"].eq("side")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq("local_aegmm_all_three")
    ].set_index("side_name")
    assert int(side.loc["long", "selected_rows"]) == 2
    assert int(side.loc["short", "selected_rows"]) == 0


def test_metrics_use_reachable_ev_policy_columns_for_top10_when_present() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-04-01"] * 4, utc=True),
            "__symbol__": ["A", "B", "C", "D"],
            "calendar_month": "2026-04",
            "week_start": pd.Timestamp("2026-03-30", tz="UTC"),
            "side_name": ["long", "long", "short", "short"],
            "archetype_policy_key": ["a", "a", "b", "b"],
            "score_alternative": [0.9, 0.8, 0.7, 0.6],
            "score_current_reference": [0.6, 0.7, 0.8, 0.9],
            "hit_prob_alternative": [0.5] * 4,
            "hit_prob_current_reference": [0.5] * 4,
            "ev_after_1pct": [0.01, 0.02, -0.01, 0.03],
            "clean_exec": [1, 1, 0, 1],
            "dirty_positive": [0, 0, 1, 0],
            "first_touch_bad_mae_1r": [0, 0, 1, 0],
            "full_path_bad_mae_1r": [0, 0, 1, 0],
            "timeout": [0, 0, 0, 0],
            "policy_selected_current_reference": [False, False, True, False],
            "policy_selected_alternative": [True, True, False, False],
        }
    )
    metrics = metrics_by_scope(frame, "alternative")
    top10 = metrics[
        metrics["scope"].eq("overall") & metrics["fraction"].eq(0.10)
    ]
    assert set(top10["selection_basis"]) == {
        "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1"
    }
    selected = top10.set_index("selector")["selected_rows"]
    assert int(selected["current_reference"]) == 1
    assert int(selected["alternative"]) == 2


def test_surprise_autocorrelation_reports_raw_and_policy_top10_separately() -> None:
    dates = pd.date_range("2026-04-01", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": dates.repeat(2),
            "side_name": ["long"] * 8,
            "archetype_policy_key": ["long_a"] * 8,
            "score_current_reference": [0.9, 0.1] * 4,
            "score_alternative": [0.9, 0.1] * 4,
            "hit_prob_current_reference": [0.5] * 8,
            "hit_prob_alternative": [0.5] * 8,
            "clean_exec": [1, 0, 0, 1, 1, 0, 0, 1],
            "ev_after_1pct": [0.01, -0.01] * 4,
            # Deliberately select the opposite row from raw top10.
            "policy_selected_current_reference": [False, True] * 4,
            "policy_selected_alternative": [False, True] * 4,
        }
    )
    calendar, autocorr, comparison = surprise_calendar(frame, "alternative")
    assert set(calendar["selection_basis"]) == {
        "raw_global_within_timestamp_top10",
        "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1",
    }
    assert set(autocorr["selection_basis"]) == set(calendar["selection_basis"])
    assert set(comparison["selection_basis"]) == set(calendar["selection_basis"])
    raw = calendar[
        calendar["selection_basis"].eq("raw_global_within_timestamp_top10")
        & calendar["selector"].eq("current_reference")
    ].sort_values("date")
    policy = calendar[
        calendar["selection_basis"].str.startswith("ev_target_")
        & calendar["selector"].eq("current_reference")
    ].sort_values("date")
    assert raw["hit_rate"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert policy["hit_rate"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_v2_residual_features_use_explicit_unknown_state_without_dropping_rows() -> None:
    data = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-04-01", "2026-04-02"], utc=True),
            "__symbol__": ["A", "B"],
            "side_name": ["long", "short"],
            "archetype_policy_key": ["a", "b"],
            "base_feature": [1.0, 2.0],
        }
    )
    residual = pd.DataFrame(
        {
            "__ts__": data["__ts__"].iloc[:1],
            "__symbol__": ["A"],
            "side_name": ["long"],
            "archetype_policy_key": ["a"],
            "meta_resid_arch_prob__base_clean_high_confidence": [0.8],
            "meta_resid_arch_prob__base_low_edge_noise": [0.2],
            "meta_resid_arch_expected_ev": [0.01],
            "meta_resid_arch_entropy": [0.3],
            "meta_resid_arch_local_model": [1.0],
            "meta_resid_market_prob__neutral": [0.2],
            "meta_resid_market_prob__synchronized_adverse": [0.7],
            "meta_resid_market_prob__synchronized_favorable": [0.1],
            "meta_resid_market_expected_ev": [-0.01],
            "meta_resid_market_arch_state_prob__bad_mae_path": [0.6],
            "meta_resid_market_arch_state_prob__slow_timeout": [0.1],
            "meta_resid_market_arch_prob_adverse__ensemble": [0.7],
            "meta_resid_market_arch_expected_bad_mae": [0.8],
        }
    )
    merged = _merge_residual_features(data, residual, fill_unknown=True)
    assert len(merged) == len(data)
    unknown = merged.iloc[1]
    assert unknown["meta_resid_arch_prob__base_low_edge_noise"] == 1.0
    assert unknown["meta_resid_arch_prob__base_clean_high_confidence"] == 0.0
    assert unknown["meta_resid_arch_entropy"] == 1.0
    assert unknown["meta_resid_arch_local_model"] == 0.0
    features = _arm_candidate_features(
        "residual_states_v2_probabilities_priors",
        merged,
        ["base_feature"],
        [],
    )
    assert "base_feature" in features
    assert "meta_resid_arch_prob__base_clean_high_confidence" in features
    assert "meta_resid_arch_expected_ev" in features
    enriched = _append_residual_state_v2_composites(merged)
    assert enriched.loc[0, "meta_resid_v2_favorable_probability"] == 0.8
    assert enriched.loc[1, "meta_resid_v2_uncertainty_probability"] == 1.0
    assert "meta_base_archetype__a" in enriched.columns
    assert "meta_base_archetype__b" in enriched.columns
    assert "meta_resid_v2_hard_state_posterior_max" in enriched.columns
    assert "meta_resid_v2_market_hard_state__synchronized_adverse" in enriched.columns
    assert enriched.loc[0, "meta_base_archetype__a"] == 1.0
    assert enriched.loc[1, "meta_base_archetype__a"] == 0.0
    local_features = _arm_candidate_features(
        "residual_states_v2_distilled_local_interactions",
        enriched,
        ["base_feature"],
        [],
    )
    assert "meta_resid_v2_expected_path_risk" in local_features
    assert any(name.startswith("meta_resid_v2_local__long__a") for name in local_features)
    full_context = _arm_candidate_features(
        "residual_states_v2_full_context",
        enriched,
        ["base_feature"],
        [],
    )
    assert "meta_base_archetype__a" in full_context
    assert "meta_resid_v2_hard_state_posterior_max" in full_context
    assert "meta_resid_market_prob__synchronized_adverse" in full_context
    market_features = _arm_candidate_features(
        "residual_states_v2_priors_market",
        enriched,
        ["base_feature"],
        [],
    )
    assert "meta_resid_market_expected_ev" in market_features
    market_failure_only = _arm_candidate_features(
        "residual_states_v2_archetype_market_failure_posteriors_only",
        enriched,
        ["base_feature"],
        [],
    )
    assert "meta_resid_market_arch_state_prob__bad_mae_path" in market_failure_only
    assert "meta_resid_market_arch_expected_bad_mae" in market_failure_only
    assert "meta_resid_market_expected_ev" not in market_failure_only
    assert "meta_resid_arch_expected_ev" not in market_failure_only
    market_failure_combined = _arm_candidate_features(
        "residual_states_v2_archetype_market_failure_posteriors",
        enriched,
        ["base_feature"],
        [],
    )
    assert "meta_resid_market_arch_state_prob__slow_timeout" in market_failure_combined
    assert "meta_resid_arch_expected_ev" in market_failure_combined


def test_residual_v2_weight_multiplier_is_normalized_and_local() -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short", "short"],
            "archetype_policy_key": ["long_mixed", "short_breakout", "short_default"],
            "meta_resid_v2_adverse_confident": [0.1, 0.9, 0.9],
            "meta_resid_v2_favorable_confident": [0.5, 0.1, 0.1],
            "meta_resid_v2_expected_path_risk": [0.1, 0.8, 0.8],
            "meta_resid_v2_uncertainty_probability": [0.2, 0.2, 0.2],
            "meta_resid_v2_expected_hit_surprise": [0.1, -0.8, -0.8],
        }
    )
    weights = _residual_v2_sample_weight_multiplier(
        frame, "residual_states_v2_distilled_weighted_short_adverse"
    )
    assert weights is not None
    assert np.isclose(float(weights.mean()), 1.0)
    assert weights.iloc[1] > weights.iloc[2] > weights.iloc[0]


def test_eval_month_parser_and_sequential_calibration_burnin() -> None:
    assert _parse_months("2026-03, 2026-04,2026-04") == ("2026-03", "2026-04")

    march_scores = np.tile(np.linspace(0.05, 0.95, 11, dtype=np.float32), 20)
    april_scores = np.array([0.2, 0.8], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "calendar_month": ["2026-03"] * len(march_scores)
            + ["2026-04"] * len(april_scores),
            "side_name": ["long"] * (len(march_scores) + len(april_scores)),
            "archetype_policy_key": ["long_a"]
            * (len(march_scores) + len(april_scores)),
            "score_alternative": np.r_[march_scores, april_scores],
            "clean_exec": np.r_[
                (march_scores >= 0.5).astype(np.float32),
                [0.0, 1.0],
            ],
        }
    )
    calibrated, contract = sequential_calibrate(
        frame,
        source_col="score_alternative",
        target_col="clean_exec",
        min_local_rows=10,
        min_side_rows=10,
    )
    assert contract.loc[contract["month"].eq("2026-03"), "fallback"].iloc[0] == (
        "raw_no_prior"
    )
    assert contract.loc[contract["month"].eq("2026-04"), "train_rows"].iloc[0] == 220
    assert np.allclose(calibrated[:220], march_scores)
    assert calibrated[-2] < 0.05
    assert calibrated[-1] > 0.95


def test_residual_v2_hitrate_context_arm_adds_causal_model_features() -> None:
    data = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05"],
                utc=True,
            ),
            "__symbol__": ["A", "A", "A", "A"],
            "side_name": ["long", "long", "long", "long"],
            "archetype_policy_key": ["long_a", "long_a", "long_a", "long_a"],
            "selected_top30": [True, True, True, True],
            "score": [0.1, 0.2, 0.3, 0.4],
            "clean_exec": [0.0, 1.0, 1.0, 0.0],
            "full_path_bad_mae_1r": [1.0, 0.0, 0.0, 1.0],
            "timeout": [0.0, 0.0, 0.0, 0.0],
            "dirty_positive": [1.0, 0.0, 0.0, 1.0],
            "exec_margin": [-0.1, 0.2, 0.3, -0.2],
            "base_feature": [1.0, 2.0, 3.0, 4.0],
            "meta_resid_arch_expected_ev": [0.0, 0.1, 0.2, -0.1],
            "meta_resid_arch_entropy": [0.1, 0.2, 0.3, 0.4],
            "meta_resid_arch_confidence": [0.9, 0.8, 0.7, 0.6],
            "meta_resid_arch_support_log1p": [3.0, 3.0, 3.0, 3.0],
            "meta_resid_arch_local_model": [1.0, 1.0, 1.0, 1.0],
        }
    )
    features = _arm_candidate_features(
        "residual_states_v2_priors_hitrate_context",
        data,
        ["base_feature"],
        [],
    )
    assert "base_feature" in features
    assert "meta_resid_arch_expected_ev" in features
    assert set(HIT_SURPRISE_NUMERIC_FEATURES).issubset(features)
    outcome_features = _arm_candidate_features(
        "residual_states_v2_priors_outcome_context",
        data,
        ["base_feature"],
        [],
    )
    assert set(HIT_SURPRISE_NUMERIC_FEATURES).issubset(outcome_features)
    assert set(OUTCOME_CONTEXT_FEATURES).issubset(outcome_features)
    weighted_features = _arm_candidate_features(
        "residual_states_v2_priors_outcome_context_weighted",
        data,
        ["base_feature"],
        [],
    )
    assert set(HIT_SURPRISE_NUMERIC_FEATURES).issubset(weighted_features)
    assert set(OUTCOME_CONTEXT_FEATURES).issubset(weighted_features)
    pressure_features = _arm_candidate_features(
        "residual_states_v2_priors_failure_pressure_context",
        data,
        ["base_feature"],
        [],
    )
    assert set(HIT_SURPRISE_NUMERIC_FEATURES).issubset(pressure_features)
    assert set(OUTCOME_CONTEXT_FEATURES).issubset(pressure_features)
    assert set(FAILURE_PRESSURE_CONTEXT_FEATURES).issubset(pressure_features)
    identity_data = _append_meta_identity_features(data)
    identity_features = _arm_candidate_features(
        "residual_states_v2_priors_outcome_context_identity_weighted",
        identity_data,
        ["base_feature"],
        [],
    )
    identity_cols = [name for name in identity_data.columns if name.startswith("meta_identity_")]
    assert identity_cols
    assert set(identity_cols).issubset(identity_features)
    assert set(OUTCOME_CONTEXT_FEATURES).issubset(identity_features)
    pressure_identity_features = _arm_candidate_features(
        "residual_states_v2_priors_failure_pressure_identity_context",
        identity_data,
        ["base_feature"],
        [],
    )
    assert set(identity_cols).issubset(pressure_identity_features)
    assert set(FAILURE_PRESSURE_CONTEXT_FEATURES).issubset(
        pressure_identity_features
    )
    interaction_features = _arm_candidate_features(
        "residual_states_v2_priors_outcome_context_interactions",
        identity_data,
        ["base_feature"],
        [],
    )
    interaction_cols = [name for name in interaction_features if "__x__meta_identity_" in name]
    assert interaction_cols
    assert set(OUTCOME_CONTEXT_FEATURES).issubset(interaction_features)

    train, valid = _add_reference_fold_features(
        identity_data.iloc[:3], identity_data.iloc[3:]
    )
    assert set(HIT_SURPRISE_NUMERIC_FEATURES).issubset(train.columns)
    assert set(HIT_SURPRISE_NUMERIC_FEATURES).issubset(valid.columns)
    assert set(OUTCOME_CONTEXT_FEATURES).issubset(train.columns)
    assert set(OUTCOME_CONTEXT_FEATURES).issubset(valid.columns)
    assert set(FAILURE_PRESSURE_CONTEXT_FEATURES).issubset(train.columns)
    assert set(FAILURE_PRESSURE_CONTEXT_FEATURES).issubset(valid.columns)
    # Validation features come from prior train rows, not the validation label.
    assert float(valid["base_arch_hit_support_log1p_hl3d"].iloc[0]) > 0.0
    assert float(valid["base_arch_hit_recent_rate_hl3d"].iloc[0]) > 0.0
    assert float(valid["base_arch_outcome_support_log1p_hl3d"].iloc[0]) > 0.0
    assert np.isfinite(float(valid["base_arch_ev_surprise_hl3d"].iloc[0]))
    assert np.isfinite(float(valid["base_arch_failure_pressure_hl3d"].iloc[0]))
    assert np.isfinite(float(valid["base_arch_quality_balance_hl3d"].iloc[0]))
    generated_interactions = [name for name in valid.columns if "__x__meta_identity_" in name]
    assert generated_interactions
    assert np.isfinite(float(valid[generated_interactions[0]].iloc[0]))


def test_outcome_context_weight_multiplier_is_prior_feature_based() -> None:
    frame = pd.DataFrame(
        {
            "meta_resid_v2_adverse_confident": [0.0, 0.0],
            "meta_resid_v2_favorable_confident": [0.0, 0.0],
            "meta_resid_v2_expected_path_risk": [0.0, 0.0],
            "meta_resid_v2_uncertainty_probability": [0.0, 0.0],
            "meta_resid_v2_expected_hit_surprise": [0.0, 0.0],
            "base_arch_bad_mae_surprise_hl3d": [0.0, 0.4],
            "base_arch_bad_mae_surprise_hl7d": [0.0, 0.2],
            "base_arch_timeout_surprise_hl3d": [0.0, 0.1],
            "base_arch_timeout_surprise_hl7d": [0.0, 0.0],
            "base_arch_dirty_surprise_hl3d": [0.0, 0.1],
            "base_arch_dirty_surprise_hl7d": [0.0, 0.1],
            "base_arch_ev_surprise_hl3d": [0.0, -0.01],
            "base_arch_ev_surprise_hl7d": [0.0, 0.0],
            "base_arch_hit_surprise_z_hl3d": [0.0, -2.0],
            "base_arch_hit_surprise_z_hl7d": [0.0, 0.0],
            "base_arch_outcome_effective_n_hl3d": [0.0, 20.0],
            "base_arch_outcome_effective_n_hl7d": [0.0, 0.0],
        }
    )
    weights = _residual_v2_sample_weight_multiplier(
        frame, "residual_states_v2_priors_outcome_context_weighted"
    )
    assert weights is not None
    assert np.isclose(float(weights.mean()), 1.0)
    assert float(weights.iloc[1]) > float(weights.iloc[0])
