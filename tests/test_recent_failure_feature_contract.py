import pandas as pd
import pytest

from scripts.diagnose_meta_recent_failures import (
    _candidate_feature_contract,
    _is_deployable_export_feature,
    _is_forbidden_feature_name,
    _known_export_features,
    _matched_baseline_weeks,
    _summarise_report as _summarise_failure_diagnostics_report,
)
from scripts.quantify_bad_regime_archetype_usefulness import _summary_report
from scripts.quantify_bad_regime_archetype_usefulness import (
    _archetype_alias_audit_rows,
    _evaluate_smooth_risk_scalers,
    _fit_lgbm_episode_transfer_binary,
    _safe_week_start,
    _training_intervention_recommendations,
)


def test_recent_failure_contract_blocks_target_like_rank_bin_exports():
    forbidden = [
        "oof_rank_bin_win_rate_oof",
        "export__oof_rank_bin_lift_oof",
        "export__oof_rank_bin_net_ret_oof",
        "oof_rank_bin_se_oof",
        "export__diag_mean_pred",
        "oof_leaf_target_mean_mean",
    ]
    for name in forbidden:
        assert _is_forbidden_feature_name(name)
        assert not _is_deployable_export_feature(name)


def test_recent_failure_contract_keeps_deployable_score_path_exports():
    allowed = [
        "oof_pred",
        "oof_rank_pct",
        "oof_score_early_10pct",
        "oof_rank_path_std",
        "oof_regime_centroid_similarity_train",
        "volatility_zscore",
        "asset_minus_mkt_oi_1d",
    ]
    for name in allowed:
        assert not _is_forbidden_feature_name(name)
        assert _is_deployable_export_feature(name)


def test_known_export_features_uses_clean_allowlist():
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "symbol": ["A", "A", "A"],
            "oof_pred": [0.1, 0.2, 0.3],
            "oof_rank_bin_win_rate_oof": [0.4, 0.5, 0.6],
            "diag_mean_pred": [0.7, 0.8, 0.9],
            "asset_minus_mkt_oi_1d": [1.0, 2.0, 3.0],
        }
    )
    export = _known_export_features(panel)
    assert "export__oof_pred" in export.columns
    assert "export__asset_minus_mkt_oi_1d" in export.columns
    assert "export__oof_rank_bin_win_rate_oof" not in export.columns
    assert "export__diag_mean_pred" not in export.columns


def test_candidate_feature_contract_carries_availability_flags():
    frame = pd.DataFrame(
        {
            "export__oof_pred": [0.1, 0.2],
            "export__oof_rank_bin_lift_oof": [0.0, 0.0],
            "url__regime_prob_max": [0.8, 0.9],
        }
    )
    contract = _candidate_feature_contract(frame).set_index("feature")
    assert bool(contract.loc["export__oof_pred", "available_before_trade"])
    assert bool(contract.loc["export__oof_pred", "outcome_independent"])
    assert not bool(contract.loc["export__oof_rank_bin_lift_oof", "allowed_by_clean_contract"])
    assert bool(contract.loc["url__regime_prob_max", "available_before_trade"])
    assert contract.loc["url__regime_prob_max", "causal_availability"] == "causal_latent_state_transform"


def test_archetype_alias_audit_tracks_partial_resolution_and_alias_parents():
    archetypes = pd.DataFrame(
        {
            "badregime__leverage_crowding_archetype_score": [0.1, 0.2, 0.3],
            "badregime__leverage_crowding_archetype_support": [1.0, 1.0, 1.0],
            "badregime__leverage_crowding_archetype_probability": [0.3, 0.4, 0.5],
            "archetype_1_score": [0.1, 0.2, 0.3],
            "leverage_funding_crowding_score": [0.1, 0.2, 0.3],
        }
    )
    diagnostics = {
        "archetypes": {
            "leverage_crowding_archetype": {
                "requested_features": 4,
                "resolved_features": 2,
                "active": True,
                "resolved_feature_map": {
                    "asset_minus_mkt_oi_1d": "asset_minus_mkt_oi_1d",
                    "funding_mom_4h": "funding_mom_4h",
                },
                "score_column": "badregime__leverage_crowding_archetype_score",
                "support_column": "badregime__leverage_crowding_archetype_support",
                "probability_column": "badregime__leverage_crowding_archetype_probability",
                "alias_columns": ["archetype_1_score", "leverage_funding_crowding_score"],
            }
        }
    }
    audit = pd.DataFrame(
        _archetype_alias_audit_rows(
            head="long_bars",
            archetypes=archetypes,
            diagnostics=diagnostics,
        )
    ).set_index("output_feature")
    assert audit.loc["badregime__leverage_crowding_archetype_score", "resolved_fraction"] == 0.5
    assert audit.loc["badregime__leverage_crowding_archetype_score", "fallback_fraction"] == 0.5
    assert audit.loc["leverage_funding_crowding_score", "resolved_parents"] == "asset_minus_mkt_oi_1d,funding_mom_4h"
    assert audit.loc["archetype_1_score", "source_archetype"] == "leverage_crowding_archetype"


def test_matched_baseline_weeks_prefers_observable_similarity_over_recency():
    weekly = pd.DataFrame(
        {
            "week": pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-12",
                    "2026-01-19",
                    "2026-01-26",
                    "2026-02-02",
                    "2026-02-09",
                ],
                utc=True,
            ),
            "rows": [1000, 130, 980, 150, 160, 990],
            "symbol_count": [100, 20, 98, 25, 30, 99],
            "timestamp_count": [168, 50, 168, 55, 60, 168],
            "rows_per_timestamp_mean": [6.0, 2.0, 5.8, 2.2, 2.3, 5.9],
            "asset_age_hours_mean": [2000, 100, 1950, 120, 140, 1980],
            "pred_mean": [0.70, 0.20, 0.68, 0.22, 0.23, 0.69],
            "pred_std": [0.10, 0.02, 0.11, 0.02, 0.02, 0.10],
            "usable_week": [True] * 6,
        }
    )
    baseline, meta = _matched_baseline_weeks(
        weekly,
        bad_week="2026-02-09",
        usable_week_set=set(pd.to_datetime(weekly["week"]).dt.strftime("%Y-%m-%d")),
        all_week_labels=set(pd.to_datetime(weekly["week"]).dt.strftime("%Y-%m-%d")),
        max_weeks=2,
    )
    assert baseline == ["2026-01-05", "2026-01-19"]
    assert meta["baseline_match_method"] == "prior_observable_similarity"
    assert meta["baseline_match_feature_count"] >= 3


def test_recent_failure_report_surfaces_local_and_leaf_interaction_diagnostics(tmp_path):
    _summarise_failure_diagnostics_report(
        tmp_path,
        failure_rows=[
            {
                "head": "long_bars",
                "auc_mean": 0.72,
                "rows": 1000,
                "positive_rate": 0.35,
            }
        ],
        adversarial_rows=[
            {
                "head": "long_bars",
                "diagnostic": "adversarial_global_bad_weeks",
                "auc_mean": 0.91,
                "bad_rows": 100,
                "normal_rows": 900,
            },
            {
                "head": "long_bars",
                "diagnostic": "adversarial_local_bad_week",
                "bad_week": "2026-06-01",
                "baseline_weeks": "2026-05-04,2026-05-11",
                "auc_mean": 0.88,
                "bad_rows": 100,
                "normal_rows": 200,
            },
        ],
        residualized_adversarial_rows=[
            {
                "head": "long_bars",
                "diagnostic": "residualized_adversarial_global_bad_weeks",
                "raw_auc": 0.94,
                "nuisance_auc": 0.80,
                "residualized_auc": 0.76,
                "raw_minus_nuisance_auc": 0.14,
                "incremental_auc_beyond_nuisance": 0.08,
            },
            {
                "head": "long_bars",
                "diagnostic": "residualized_adversarial_local_bad_week",
                "bad_week": "2026-06-01",
                "baseline_weeks": "2026-05-04,2026-05-11",
                "raw_auc": 0.92,
                "nuisance_auc": 0.78,
                "residualized_auc": 0.75,
                "incremental_auc_beyond_nuisance": 0.07,
                "bad_rows": 100,
                "normal_rows": 200,
            },
        ],
        leaf_rows=[
            {
                "head": "long_bars",
                "meta_leaf_rows": 12,
                "base_leaf_rows": 7,
                "meta_model_count": 3,
                "base_model_count": 2,
                "meta_top_instability_score": 0.6,
                "meta_top_occupancy_shift": 0.1,
                "meta_top_outcome_shift": -0.2,
                "meta_top_calibration_shift": -0.3,
                "meta_top_recent_support": 50,
                "base_top_instability_score": 0.4,
                "base_top_occupancy_shift": 0.05,
                "base_top_outcome_shift": -0.1,
                "base_top_calibration_shift": -0.15,
                "base_top_recent_support": 40,
                "meta_top_interaction_feature": "funding_pressure_score",
                "meta_top_interaction_score": 0.5,
                "meta_top_interaction_delta": -0.25,
                "meta_top_period_interaction_delta": -0.2,
                "meta_top_within_interaction_delta": -0.05,
                "meta_top_episode_sign_stability": 0.8,
                "meta_top_economic_effect": -0.01,
                "base_top_interaction_feature": "oi_contraction_score",
                "base_top_interaction_score": 0.3,
                "base_top_interaction_delta": -0.12,
                "base_top_period_interaction_delta": -0.08,
                "base_top_within_interaction_delta": -0.04,
                "base_top_episode_sign_stability": 0.6,
                "base_top_economic_effect": -0.005,
            }
        ],
        coverage_rows=[],
    )
    report = (tmp_path / "diagnostic_report.md").read_text()
    assert "Local adversarial validation" in report
    assert "Local Residualized Adversarial Validation" in report
    assert "Top Base/Meta Leaf Shifts" in report
    assert "Top Residual x Archetype Leaf Interactions" in report
    assert "funding_pressure_score" in report


def test_archetype_usefulness_report_includes_episode_transfer(tmp_path):
    transfer = pd.DataFrame(
        {
            "target": ["high_conf_miss", "high_conf_miss"],
            "head": ["long_bars", "long_bars"],
            "model": ["prediction_plus_archetype", "prediction_plus_archetype"],
            "heldout_episode": ["2026-05-25", "2026-06-01"],
            "transfer_reason": [float("nan"), ""],
            "transfer_roc_auc": [0.62, 0.58],
            "transfer_failure_capture_at_10pct_abstain": [0.30, 0.25],
            "transfer_retained_return_mean_at_10pct_abstain": [0.01, 0.02],
            "transfer_rejected_winner_cost_at_10pct_abstain": [0.001, 0.002],
        }
    )
    shadow_policy = pd.DataFrame(
        {
            "head": ["long_bars"],
            "target": ["high_conf_miss"],
            "model": ["support_x_market_interactions"],
            "policy": ["linear_floor25_alpha75"],
            "avg_size": [0.7],
            "success_minus_failure_exposure": [0.2],
            "return_mean_delta": [0.001],
            "tail_loss_delta_q05": [0.01],
            "winner_haircut_mean": [0.001],
            "loser_loss_reduction_mean": [0.005],
            "risk_sizing_score": [0.006],
        }
    )
    _summary_report(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        [],
        pd.DataFrame(),
        transfer,
        shadow_policy,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    report = (tmp_path / "bad_regime_archetype_usefulness_report.md").read_text()
    assert "Leave-One-Episode-Out Transfer" in report
    assert "prediction_plus_archetype" in report
    assert "Shadow Failure-Risk Sizing" in report
    assert "support_x_market_interactions" in report


def test_episode_transfer_binary_smoke():
    pytest.importorskip("lightgbm")
    n = 500
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"))
    idx = pd.Series(range(n), dtype="float32")
    y = ((idx.astype("int32") % 48) < 24).astype("float32").to_numpy()
    seasonal = ((idx % 24) / 24.0).astype("float32")
    x = pd.DataFrame(
        {
            "signal": (0.75 * y + 0.25 * seasonal.to_numpy(dtype="float32")).astype("float32"),
            "seasonal": seasonal,
        }
    )
    returns = pd.Series((2.0 * y - 1.0) * 0.01, dtype="float32")
    episodes = _safe_week_start(timestamps)
    heldout = episodes.drop_duplicates().iloc[2]
    result = _fit_lgbm_episode_transfer_binary(
        x=x,
        y=y,
        timestamps=timestamps,
        realized_return=returns,
        episode_labels=episodes,
        heldout_episode=heldout,
        max_train_rows=300,
        seed=11,
    )
    assert result["transfer_reason"] == ""
    assert result["transfer_feature_count"] >= 1
    assert result["transfer_rows_test"] >= 50
    assert result["transfer_roc_auc"] >= 0.5


def test_smooth_shadow_risk_scalers_reduce_failure_exposure():
    frame = pd.DataFrame(
        {
            "head": ["long_bars"] * 100,
            "target": ["high_conf_miss"] * 100,
            "model": ["support_x_market_interactions"] * 100,
            "shadow_failure_risk": [i / 99.0 for i in range(100)],
            "realized_return": [0.01] * 50 + [-0.02] * 50,
            "failure_target": [0] * 50 + [1] * 50,
        }
    )
    eval_df = _evaluate_smooth_risk_scalers(frame)
    assert not eval_df.empty
    assert (eval_df["success_minus_failure_exposure"] > 0.0).any()
    assert (eval_df["tail_loss_delta_q05"] >= 0.0).any()


def test_training_intervention_recommendations_gate_retrain_and_sizing():
    model_rows = pd.DataFrame(
        {
            "head": ["long_bars", "long_bars", "long_bars"],
            "target": ["high_conf_miss", "high_conf_miss", "high_conf_miss"],
            "target_kind": ["binary", "binary", "binary"],
            "model": [
                "prediction_controls_only",
                "nuisance_controls_only",
                "prediction_plus_archetype",
            ],
            "auc_mean": [0.60, 0.59, 0.625],
        }
    )
    transfer = pd.DataFrame(
        {
            "head": ["long_bars", "long_bars"],
            "target": ["high_conf_miss", "high_conf_miss"],
            "model": ["prediction_plus_archetype", "prediction_plus_archetype"],
            "heldout_episode": ["2026-05-25", "2026-06-01"],
            "transfer_reason": [float("nan"), ""],
            "transfer_roc_auc": [0.57, 0.58],
            "transfer_failure_capture_at_10pct_abstain": [0.25, 0.30],
            "transfer_retained_return_mean_at_10pct_abstain": [0.01, 0.02],
            "transfer_rejected_winner_cost_at_10pct_abstain": [0.001, 0.001],
        }
    )
    shadow_policy = pd.DataFrame(
        {
            "head": ["long_bars"],
            "target": ["high_conf_miss"],
            "model": ["support_x_market_interactions"],
            "policy": ["linear_floor25_alpha75"],
            "risk_sizing_score": [0.01],
            "tail_loss_delta_q05": [0.02],
            "success_minus_failure_exposure": [0.15],
            "winner_haircut_mean": [0.001],
            "loser_loss_reduction_mean": [0.003],
            "avg_size": [0.72],
        }
    )
    recs = _training_intervention_recommendations(
        model_rows=model_rows,
        transfer=transfer,
        shadow_policy=shadow_policy,
        decomposition=pd.DataFrame(),
    )
    retrain = recs.loc[recs["action"].eq("archetype_aware_meta_retrain")].iloc[0]
    sizing = recs.loc[recs["action"].eq("shadow_smooth_risk_sizing")].iloc[0]
    assert retrain["recommendation"] == "candidate"
    assert bool(retrain["recurrence_pass"])
    assert bool(retrain["incremental_lift_pass"])
    assert sizing["recommendation"] == "candidate"
