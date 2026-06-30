from pathlib import Path
import hashlib
import json
import subprocess
import sys

import joblib
import pandas as pd

from scripts import audit_market_state_controller_contract as audit


def _valid_manifest() -> dict:
    return {
        "rank_contract": "short_boll_timestamp_rank",
        "disabled_heads": ["long_bars", "long_dist"],
        "active_heads": ["short_asset", "short_boll"],
        "selected_controller_candidate": {"selected_arm": None},
        "controller": {
            "penalty_only": True,
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
            "controller_enabled_heads": ["short_asset", "short_boll"],
            "controller_enabled_scope": "all_active_heads",
            "allow_candidate_state_fallback": False,
            "include_latent_shadow_arms": False,
        },
        "source_contract_audit": {
            "audit_version": "market_state_source_contract_audit_v1",
            "overall_passed": True,
            "actual_order_book_features_allowed": False,
            "candidate_population_fallback_allowed_for_production": False,
            "splits": {
                "train": {
                    "passed": True,
                    "production_safe": True,
                    "candidate_fallback_enabled": False,
                    "validation_forbidden_column_count": 0,
                    "timestamp_unique": True,
                    "market_wide_one_row_per_timestamp": True,
                },
                "eval": {
                    "passed": True,
                    "production_safe": True,
                    "candidate_fallback_enabled": False,
                    "validation_forbidden_column_count": 0,
                    "timestamp_unique": True,
                    "market_wide_one_row_per_timestamp": True,
                },
            },
        },
    }


def test_target_definition_audit_accepts_declared_current_axis_fallback() -> None:
    payload = {
        "contract_version": "market_state_target_definitions_v1",
        "target_type": "training_cdf_normalized_future_market_geometry_soft_severity",
        "forecast_targets": {
            "forecast_h6_deleveraging": {
                "fold_count": 1,
                "folds": [
                    {
                        "fold": 1,
                        "mode": "current_axis_fallback",
                        "train_prediction_mode": "bounded_current_axis_fallback",
                        "fallback_axis": "state_shock",
                    }
                ],
            }
        },
    }

    assert audit._audit_target_definitions(payload, {1}) == []


def test_target_definition_audit_rejects_fallback_without_axis() -> None:
    payload = {
        "contract_version": "market_state_target_definitions_v1",
        "target_type": "training_cdf_normalized_future_market_geometry_soft_severity",
        "forecast_targets": {
            "forecast_h6_deleveraging": {
                "fold_count": 1,
                "folds": [
                    {
                        "fold": 1,
                        "mode": "current_axis_fallback",
                        "train_prediction_mode": "bounded_current_axis_fallback",
                    }
                ],
            }
        },
    }

    failures = audit._audit_target_definitions(payload, {1})

    assert "market_state_target_definitions.forecast_h6_deleveraging.fallback_axis is missing" in failures


def test_market_state_controller_contract_audit_accepts_t1_contract() -> None:
    assert audit.audit_manifest(_valid_manifest(), require_null_selection=True) == []


def test_market_state_controller_contract_audit_accepts_explicit_global_rank_contract() -> None:
    manifest = _valid_manifest()
    manifest["rank_contract"] = "anchor_global_policy_rank_reference"

    default_failures = audit.audit_manifest(manifest, require_null_selection=True)
    override_failures = audit.audit_manifest(
        manifest,
        require_null_selection=True,
        expected_rank_contract="anchor_global_policy_rank_reference",
    )

    assert "rank_contract != short_boll_timestamp_rank" in default_failures
    assert override_failures == []


def test_feature_and_controller_contract_audits_accept_explicit_global_rank_contract() -> None:
    feature_contract = {
        "rank_contract": "anchor_global_policy_rank_reference",
        "active_heads": ["short_asset", "short_boll"],
        "disabled_heads": ["long_bars", "long_dist"],
        "invariants": {
            key: True for key in audit.FEATURE_CONTRACT_TRUE_INVARIANTS
        }
        | {
            key: False for key in audit.FEATURE_CONTRACT_FALSE_INVARIANTS
        },
        "validation": {
            "passed": True,
            "failures": [],
            "fold_count": 1,
            "state_head_registry_rows": 1,
        },
    }
    controller_config = {
        "baseline_contract": {
            "rank_contract": "anchor_global_policy_rank_reference",
            "active_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "q_fail_enabled": False,
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
        },
        "controller": {"penalty_only": True},
        "validation": {
            "chronological_complete_timestamp_folds": True,
            "embargo_hours": 12,
            "selected_controller_is_null": True,
        },
    }

    default_feature_failures = audit._audit_feature_contract(feature_contract)
    override_feature_failures = audit._audit_feature_contract(
        feature_contract,
        expected_rank_contract="anchor_global_policy_rank_reference",
    )
    assert audit._audit_controller_config(
        controller_config,
        expected_rank_contract="anchor_global_policy_rank_reference",
    ) == []
    assert (
        "market_state_feature_contract.rank_contract != short_boll_timestamp_rank"
        in default_feature_failures
    )
    assert not any("rank_contract" in failure for failure in override_feature_failures)


def test_controller_config_audit_accepts_no_backfill_overlay_selection() -> None:
    controller_config = {
        "baseline_contract": {
            "rank_contract": "anchor_global_policy_rank_reference",
            "active_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "q_fail_enabled": False,
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
        },
        "controller": {
            "penalty_only": True,
            "include_post_selection_overlay_arms": True,
            "select_no_backfill_overlay_only": True,
            "post_selection_overlay_contract": (
                "baseline accepted decision keys only; no freed-capacity backfill"
            ),
        },
        "validation": {
            "chronological_complete_timestamp_folds": True,
            "embargo_hours": 96,
            "selected_controller_is_null": False,
        },
    }

    assert audit._audit_controller_config(
        controller_config,
        expected_rank_contract="anchor_global_policy_rank_reference",
    ) == []


def test_selected_controller_audit_accepts_no_backfill_overlay_selection() -> None:
    payload = {
        "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
        "selection_policy": {"select_no_backfill_overlay_only": True},
        "selected_metrics": {
            "action_entrants": 0.0,
            "passed_selection_gates": True,
        },
    }

    assert audit._audit_selected_controller(payload) == []


def test_selected_controller_audit_rejects_no_backfill_overlay_with_entrants() -> None:
    payload = {
        "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
        "selection_policy": {"select_no_backfill_overlay_only": True},
        "selected_metrics": {
            "action_entrants": 1.0,
            "passed_selection_gates": True,
        },
    }

    failures = audit._audit_selected_controller(payload)

    assert "walkforward_selected_controller_candidate.selected_arm is not null" in failures


def test_market_state_controller_contract_audit_rejects_mutating_controller() -> None:
    manifest = _valid_manifest()
    manifest["controller"]["changes_scores_or_ranks"] = True
    failures = audit.audit_manifest(manifest)
    assert "controller.changes_scores_or_ranks is not false" in failures


def test_market_state_controller_contract_audit_rejects_implicit_scope() -> None:
    manifest = _valid_manifest()
    manifest["active_heads"] = None
    manifest["controller"]["controller_enabled_heads"] = "all_active_heads"
    failures = audit.audit_manifest(manifest)
    assert "active_heads != ['short_asset', 'short_boll']" in failures
    assert "controller.controller_enabled_heads != ['short_asset', 'short_boll']" in failures


def test_market_state_controller_contract_audit_accepts_bundle_top_level_scope() -> None:
    manifest = _valid_manifest()
    manifest["controller"].pop("controller_enabled_heads")
    manifest["controller"].pop("controller_enabled_scope")
    manifest["controller"].pop("allow_candidate_state_fallback")
    manifest["controller"].pop("include_latent_shadow_arms")
    manifest["controller_enabled_heads"] = ["short_asset", "short_boll"]
    manifest["controller_enabled_scope"] = "all_active_heads"
    manifest["latent_report"] = {"mode": "shadow_disabled_by_default"}

    assert audit.audit_manifest(manifest, require_null_selection=True) == []


def test_market_state_controller_contract_audit_accepts_activation_registry_disabled_scope_when_explicit() -> None:
    manifest = _valid_manifest()
    manifest["controller"].pop("controller_enabled_heads")
    manifest["controller"].pop("controller_enabled_scope")
    manifest["controller"]["execution_enabled"] = False
    manifest["controller_enabled_heads"] = []
    manifest["controller_enabled_scope"] = "disabled_by_activation_registry"
    manifest["controller_execution_enabled"] = False

    assert audit.audit_manifest(
        manifest,
        allow_disabled_by_activation_registry=True,
    ) == []


def test_market_state_controller_contract_audit_rejects_activation_registry_disabled_scope_by_default() -> None:
    manifest = _valid_manifest()
    manifest["controller"].pop("controller_enabled_heads")
    manifest["controller"].pop("controller_enabled_scope")
    manifest["controller"]["execution_enabled"] = False
    manifest["controller_enabled_heads"] = []
    manifest["controller_enabled_scope"] = "disabled_by_activation_registry"
    manifest["controller_execution_enabled"] = False

    failures = audit.audit_manifest(manifest)

    assert "controller.controller_enabled_heads != ['short_asset', 'short_boll']" in failures
    assert "controller.controller_enabled_scope != all_active_heads" in failures


def test_market_state_controller_contract_audit_accepts_shadow_only_explicit_scope() -> None:
    manifest = _valid_manifest()
    manifest["controller"].pop("controller_enabled_heads")
    manifest["controller"].pop("controller_enabled_scope")
    manifest["controller"]["execution_enabled"] = False
    manifest["controller"]["shadow_controller_only"] = True
    manifest["controller_enabled_heads"] = []
    manifest["controller_enabled_scope"] = "disabled_by_activation_registry"
    manifest["controller_execution_enabled"] = False
    manifest["shadow_controller_only"] = True
    manifest["shadow_controller_enabled_heads"] = ["short_asset", "short_boll"]
    manifest["shadow_controller_enabled_scope"] = "explicit"

    assert audit.audit_manifest(manifest) == []


def test_market_state_controller_contract_audit_rejects_missing_source_audit() -> None:
    manifest = _valid_manifest()
    manifest.pop("source_contract_audit")

    failures = audit.audit_manifest(manifest)

    assert "source_contract_audit is missing" in failures


def test_market_state_controller_contract_audit_rejects_unsafe_source_audit() -> None:
    manifest = _valid_manifest()
    train = manifest["source_contract_audit"]["splits"]["train"]
    train["passed"] = False
    train["production_safe"] = False
    train["candidate_fallback_enabled"] = True
    train["validation_forbidden_column_count"] = 2

    failures = audit.audit_manifest(manifest)

    assert "source_contract_audit.overall_passed is not true" not in failures
    assert "source_contract_audit.train.passed is not true" in failures
    assert "source_contract_audit.train.production_safe is not true" in failures
    assert "source_contract_audit.train.candidate_fallback_enabled is not false" in failures
    assert "source_contract_audit.train.validation_forbidden_column_count != 0" in failures


def _write_minimal_artifacts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    joblib.dump(
        {
            "generated_by": "run_market_state_threshold_controller_walkforward",
            "reference_version": "market_state_training_reference_bundle_v1",
            "fold_references": {
                "fold_1": {
                    "feature_store_columns": ["mkt_ret_eq_1h", "amihud_illiq"],
                    "feature_store_tail_reference_quantiles": {
                        "mkt_ret_eq_1h": {"q10": -0.01, "q90": 0.01},
                        "amihud_illiq": {"q10": 0.1, "q90": 0.9},
                    },
                    "feature_store_reports": {
                        "train": {
                            "tail_reference_source": "self_window_reference",
                            "tail_reference_role": "fit_on_training_timestamps",
                            "tail_reference_quantiles": {
                                "mkt_ret_eq_1h": {"q10": -0.01, "q90": 0.01},
                                "amihud_illiq": {"q10": 0.1, "q90": 0.9},
                            },
                        },
                        "valid": {
                            "tail_reference_source": "provided_train_reference",
                            "tail_reference_role": "transformed_with_training_timestamp_reference",
                        },
                    },
                    "observed_axis_encoder": {
                        "mode": "observed_axis_robust_z_v1",
                        "minimum_input_coverage": 0.80,
                        "axes": {
                            "state_shock": ["fs__mkt_ret_eq_1h__mean"],
                            "state_liquidity_stress_proxy": ["fs__amihud_illiq__mean"],
                        },
                        "column_refs": {
                            "fs__mkt_ret_eq_1h__mean": {
                                "median": 0.0,
                                "scale": 1.0,
                                "q05": -1.0,
                                "q95": 1.0,
                            },
                            "fs__amihud_illiq__mean": {
                                "median": 0.0,
                                "scale": 1.0,
                                "q05": -1.0,
                                "q95": 1.0,
                            },
                        },
                        "axis_sources": {
                            "state_shock": ["fs__mkt_ret_eq_1h__mean"],
                            "state_liquidity_stress_proxy": ["fs__amihud_illiq__mean"],
                            "state_input_coverage": ["fs__mkt_ret_eq_1h__mean", "fs__amihud_illiq__mean"],
                            "state_uncertainty": [
                                "state_novelty",
                                "state_drift_score",
                                "state_input_coverage",
                                "state_extreme_value_share",
                            ],
                            "state_low_input_coverage": [
                                "state_input_coverage",
                                "minimum_input_coverage=0.8000",
                            ],
                        },
                        "source_validation": {
                            "train": {
                                "timestamp_unique": True,
                                "market_wide_one_row_per_timestamp": True,
                                "forbidden_column_count": 0,
                            },
                            "eval": {
                                "timestamp_unique": True,
                                "market_wide_one_row_per_timestamp": True,
                                "forbidden_column_count": 0,
                            },
                        },
                    },
                },
            },
        },
        root / "market_state_training_reference.joblib",
    )
    (root / "market_state_target_definitions.json").write_text(
        json.dumps(
            {
                "contract_version": "market_state_target_definitions_v1",
                "generated_by": "run_market_state_threshold_controller_walkforward",
                "target_type": "training_cdf_normalized_future_market_geometry_soft_severity",
                "forecast_targets": {
                    "forecast_h6_shock_down": {
                        "fold_count": 1,
                        "folds": [
                            {
                                "fold": 1,
                                "train_prediction_mode": "chronological_expanding_oof_or_fallback",
                                "oof_coverage": 1.0,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    joblib.dump(
        {
            "artifact_version": "market_state_target_cdfs_v1",
            "generated_by": "run_market_state_threshold_controller_walkforward",
            "normalization": "training_fold_empirical_cdf_raw_future_market_geometry_targets",
            "target_count": 1,
            "missing_reference_count": 0,
            "folds": {"fold_1": {"forecast_h6_shock_down": {"q": [0.0, 1.0]}}},
        },
        root / "market_state_target_cdfs.joblib",
    )
    joblib.dump(
        {
            "generated_by": "run_market_state_threshold_controller_walkforward",
            "artifact_version": "market_state_forecast_models_v1",
            "forecast_model_kind": "lightgbm",
            "fold_forecast_artifacts": {"fold_1": {"model_count": 1}},
        },
        root / "market_state_lgbm_models.joblib",
    )
    joblib.dump(
        {
            "generated_by": "run_market_state_threshold_controller_walkforward",
            "artifact_version": "strategy_rank_outcome_curves_v1",
            "rank_curve_table": pd.DataFrame({"strategy_id": ["s1"], "rank": [0.8], "mu": [0.01]}),
        },
        root / "strategy_rank_outcome_curves.joblib",
    )
    joblib.dump(
        {
            "generated_by": "run_market_state_threshold_controller_walkforward",
            "model_type": "rank_curve_plus_additive_ebm_response",
            "response_model_kind": "additive_ebm",
            "fold_models": {
                "fold_1__S1_observed_axes_shared_response": {
                    "fold": 1,
                    "arm": "S1_observed_axes_shared_response",
                    "model_type": "rank_curve_plus_additive_ebm_response",
                    "state_columns": ["state_shock"],
                    "response_feature_columns": ["state_shock", "normalized_rank_score"],
                    "model_report": {
                        "train_rows": 10,
                        "state_training_input_contract": "fold_fitted_descriptive_state_axes_no_learned_state_oof_required",
                        "response_training_state_prediction_contract": "fold_fitted_descriptive_state_axes_no_learned_state_oof_required",
                        "response_training_uses_oof_state_scores": True,
                        "response_training_state_contract_passed": True,
                        "learned_state_non_oof_columns": [],
                    },
                    "models": {"response_model_kind": "additive_ebm"},
                }
            },
        },
        root / "strategy_response_models.joblib",
    )
    (root / "market_state_feature_contract.json").write_text(
        json.dumps(
            {
                "rank_contract": "short_boll_timestamp_rank",
                "active_heads": ["short_asset", "short_boll"],
                "disabled_heads": ["long_bars", "long_dist"],
                "invariants": {
                    "one_market_state_row_per_timestamp": True,
                    "state_join_timestamp_constant": True,
                    "market_state_uses_strategy_ids": False,
                    "market_state_uses_model_predictions": False,
                    "market_state_uses_ranks": False,
                    "market_state_uses_candidate_counts": False,
                    "market_state_uses_portfolio_pnl": False,
                    "market_state_uses_realized_strategy_outcomes": False,
                    "actual_order_book_features_allowed": False,
                    "candidate_population_fallback_enabled": False,
                    "candidate_population_fallback_is_production_safe": False,
                    "controller_changes_scores_or_ranks": False,
                    "controller_changes_auction_ordering": False,
                    "controller_can_lower_thresholds": False,
                    "latent_gmm_active_controller_input": False,
                },
                "validation": {
                    "passed": True,
                    "failures": [],
                    "fold_count": 1,
                    "state_head_registry_rows": 1,
                    "training_outcome_maturity_contract_passed": True,
                    "training_immature_outcome_rows_dropped": 0,
                    "training_outcome_maturity_failures": [],
                },
                "fold_definition": {
                    "n_folds_requested": 1,
                    "folds_built": [
                        {
                            "fold": 1,
                            "train_start": "2026-05-01T00:00:00+00:00",
                            "train_end": "2026-05-10T00:00:00+00:00",
                            "valid_start": "2026-05-14T00:00:00+00:00",
                            "valid_end": "2026-05-21T00:00:00+00:00",
                            "valid_rows_available": 25,
                            "valid_timestamps_available": 4,
                        }
                    ],
                    "embargo_hours": 96,
                    "min_valid_rows": 25,
                    "min_valid_timestamps": 4,
                },
                "source_schema": {
                    "source": "feature_store_market_aggregates",
                    "feature_store_columns": ["mkt_ret_eq_1h", "amihud_illiq"],
                    "observed_axis_columns": ["state_shock", "state_liquidity_stress_proxy"],
                },
                "source_contract_audit": _valid_manifest()["source_contract_audit"],
                "universe_contract": {
                    "contract_version": "market_state_universe_contract_v1",
                    "validation": {
                        "passed": True,
                        "failures": [],
                        "fold_split_count": 2,
                        "eligible_symbol_list_constant": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "market_state_universe_contract.json").write_text(
        json.dumps(
            {
                "contract_version": "market_state_universe_contract_v1",
                "generated_by": "run_market_state_threshold_controller_walkforward",
                "required_source": "feature_store_market_aggregates",
                "universe_definition_versions": ["feature_store_timestamp_market_state_v1"],
                "strategy_independent": True,
                "candidate_independent": True,
                "actual_order_book_features_allowed": False,
                "candidate_population_fallback_enabled": False,
                "feature_dirs": ["data_perp/features/test"],
                "symbol_caps": [2],
                "available_symbol_counts": [3],
                "eligible_symbol_counts": [2],
                "eligible_symbols": ["BTC_USD:BTC", "ETH_USD:ETH"],
                "eligible_symbol_count": 2,
                "minimum_history": ["upstream_feature_store_history_available_at_requested_timestamps"],
                "minimum_volume": ["upstream_feature_store_volume_filters_or_none"],
                "oi_coverage_requirements": ["optional_oi_features_are_used_when_present_and_finite"],
                "funding_coverage_requirements": ["optional_funding_features_are_used_when_present_and_finite"],
                "excluded_symbols_and_reasons": {"XRP_USD:USD": "symbol_cap_subsample"},
                "fold_split_contracts": {
                    "fold_1_train": {
                        "fold": 1,
                        "split": "train",
                        "source": "feature_store_market_aggregates",
                        "production_safe": True,
                        "candidate_fallback_enabled": False,
                        "strategy_independent": True,
                        "candidate_independent": True,
                        "actual_order_book_features_allowed": False,
                        "universe_definition_version": "feature_store_timestamp_market_state_v1",
                        "universe_source": "feature_store_symbol_parquet_files",
                        "feature_dir": "data_perp/features/test",
                        "minimum_history": "upstream_feature_store_history_available_at_requested_timestamps",
                        "minimum_volume": "upstream_feature_store_volume_filters_or_none",
                        "oi_coverage_requirements": "optional_oi_features_are_used_when_present_and_finite",
                        "funding_coverage_requirements": "optional_funding_features_are_used_when_present_and_finite",
                        "symbol_cap": 2,
                        "available_symbol_count": 3,
                        "eligible_symbol_count": 2,
                        "eligible_symbols": ["BTC_USD:BTC", "ETH_USD:ETH"],
                        "excluded_symbols": ["XRP_USD:USD"],
                        "excluded_symbols_and_reasons": {"XRP_USD:USD": "symbol_cap_subsample"},
                        "selection_reason": "symbol_cap_subsample",
                        "feature_store_timestamp_coverage": 1.0,
                        "feature_store_symbols_read": 2,
                    },
                    "fold_1_valid": {
                        "fold": 1,
                        "split": "valid",
                        "source": "feature_store_market_aggregates",
                        "production_safe": True,
                        "candidate_fallback_enabled": False,
                        "strategy_independent": True,
                        "candidate_independent": True,
                        "actual_order_book_features_allowed": False,
                        "universe_definition_version": "feature_store_timestamp_market_state_v1",
                        "universe_source": "feature_store_symbol_parquet_files",
                        "feature_dir": "data_perp/features/test",
                        "minimum_history": "upstream_feature_store_history_available_at_requested_timestamps",
                        "minimum_volume": "upstream_feature_store_volume_filters_or_none",
                        "oi_coverage_requirements": "optional_oi_features_are_used_when_present_and_finite",
                        "funding_coverage_requirements": "optional_funding_features_are_used_when_present_and_finite",
                        "symbol_cap": 2,
                        "available_symbol_count": 3,
                        "eligible_symbol_count": 2,
                        "eligible_symbols": ["BTC_USD:BTC", "ETH_USD:ETH"],
                        "excluded_symbols": ["XRP_USD:USD"],
                        "excluded_symbols_and_reasons": {"XRP_USD:USD": "symbol_cap_subsample"},
                        "selection_reason": "symbol_cap_subsample",
                        "feature_store_timestamp_coverage": 1.0,
                        "feature_store_symbols_read": 2,
                    },
                },
                "validation": {
                    "passed": True,
                    "failures": [],
                    "fold_split_count": 2,
                    "eligible_symbol_list_constant": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "strategy_threshold_controller_config.json").write_text(
        json.dumps(
            {
                "baseline_contract": {
                    "rank_contract": "short_boll_timestamp_rank",
                    "active_heads": ["short_asset", "short_boll"],
                    "disabled_heads": ["long_bars", "long_dist"],
                    "q_fail_enabled": False,
                    "changes_scores_or_ranks": False,
                    "changes_auction_ordering": False,
                },
                "controller": {"penalty_only": True},
                "validation": {
                    "chronological_complete_timestamp_folds": True,
                    "embargo_hours": 96,
                    "selected_controller_is_null": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "walkforward_selected_controller_candidate.json").write_text(
        json.dumps(
            {
                "selected_arm": None,
                "reason": "no_arm_passed_selection_gates",
                "selection_policy": {"min_positive_delta_share": 0.5},
            }
        ),
        encoding="utf-8",
    )

    market_rows = pd.DataFrame(
        {
            "fold": [1, 1],
            "split": ["train", "valid"],
            "state_arm": ["S1_observed_axes_shared_response", "S1_observed_axes_shared_response"],
            "timestamp": pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z"]),
            "state_shock": [0.1, 0.2],
            "state_input_coverage": [1.0, 1.0],
            "state_drift_score": [0.05, 0.06],
        }
    )
    market_rows.to_parquet(root / "market_state_timestamp_panel.parquet", index=False)

    state_oof = pd.DataFrame(
        {
            "fold": [1],
            "split": ["valid"],
            "state_arm": ["S1_observed_axes_shared_response"],
            "timestamp": pd.to_datetime(["2026-06-01T01:00:00Z"]),
            "state_shock": [0.2],
            "state_input_coverage": [1.0],
            "state_drift_score": [0.06],
            "prediction_contract": ["outer_fold_validation_state_scores"],
        }
    )
    state_oof.to_parquet(root / "market_state_oof_predictions.parquet", index=False)

    response_oof = pd.DataFrame(
        {
            "fold": [1, 1],
            "arm": ["S1_observed_axes_shared_response", "S1_observed_axes_shared_response"],
            "timestamp": pd.to_datetime(["2026-06-01T01:00:00Z", "2026-06-01T01:00:00Z"]),
            "strategy_id": ["s1", "s2"],
            "head": ["short_asset", "short_boll"],
            "side": ["short", "short"],
            "symbol": ["BTC_USD", "ETH_USD"],
            "state_feature_coverage": [1.0, 1.0],
            "response_feature_coverage": [1.0, 1.0],
            "state_ood_score": [0.1, 0.1],
            "state_drift_score": [0.06, 0.06],
            "state_ood_cutoff": [0.8, 0.8],
            "state_ood_flag": [False, False],
            "base_mu": [0.01, 0.02],
            "base_psl": [0.1, 0.2],
            "base_pto": [0.05, 0.1],
            "pred_eu_mean": [0.012, 0.018],
            "pred_eu_q10": [0.0, -0.002],
            "pred_excess_full_sl": [0.01, 0.02],
            "pred_excess_timeout": [0.0, 0.01],
            "pred_mean_utility": [0.012, 0.018],
            "pred_lcb_utility": [0.0, -0.002],
            "pred_full_sl": [0.11, 0.22],
            "pred_timeout": [0.05, 0.11],
            "actual_resid_utility": [0.002, -0.003],
            "actual_resid_full_sl": [0.0, 0.1],
            "actual_resid_timeout": [0.0, -0.1],
            "pred_resid_utility": [0.01, -0.01],
            "pred_resid_utility_lcb": [0.0, -0.02],
            "pred_resid_full_sl": [0.01, 0.02],
            "pred_resid_timeout": [0.0, 0.01],
            "state_prediction_contract": ["outer_fold_validation_state_scores", "outer_fold_validation_state_scores"],
        }
    )
    response_oof.to_parquet(root / "strategy_response_oof_predictions.parquet", index=False)

    schedule = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-01T01:00:00Z", "2026-06-01T01:00:00Z"]),
            "strategy_id": ["s1", "s2"],
            "head": ["short_asset", "short_boll"],
            "fold": [1, 1],
            "arm": ["S1_observed_axes_shared_response", "S1_observed_axes_shared_response"],
            "base_threshold": [0.71, 0.71],
            "raw_state_threshold": [0.72, 0.71],
            "state_threshold": [0.72, 0.71],
            "controller_mode": ["penalty_only", "penalty_only"],
            "threshold_action_enabled": [True, False],
            "force_base_threshold": [False, True],
            "sl_cap": [0.1, 0.1],
            "timeout_cap": [0.1, 0.1],
            "risk_severity": [0.2, 0.0],
            "controller_reason": ["threshold_raised", "force_base"],
            "prediction_coverage": [1.0, 1.0],
            "min_prediction_coverage": [0.8, 0.8],
            "state_ood_score_mean": [0.1, 0.1],
            "state_ood_score_max": [0.1, 0.1],
            "state_ood_cutoff": [0.8, 0.8],
            "state_ood_share": [0.0, 0.0],
            "mean_pred_utility": [0.01, 0.0],
            "mean_pred_lcb": [0.0, 0.0],
            "mean_pred_full_sl": [0.1, 0.1],
            "mean_pred_timeout": [0.05, 0.05],
            "base_candidate_count": [3, 3],
            "frontier_candidate_count": [2, 2],
            "frontier_upper_rank": [0.8, 0.8],
            "tail_candidate_count": [2, 2],
            "suppressed_candidate_count": [1, 0],
            "tail_lcb_q25": [0.0, 0.0],
            "tail_pred_full_sl": [0.1, 0.1],
            "tail_pred_timeout": [0.05, 0.05],
            "predicted_removed_loss_avoided": [1.0, 0.0],
            "predicted_removed_winner_sacrificed": [0.2, 0.0],
            "predicted_action_edge": [0.8, 0.0],
            "action_edge_per_suppressed": [0.8, 0.0],
            "threshold_delta": [0.01, 0.0],
            "threshold_raised": [True, False],
        }
    )
    schedule.to_parquet(root / "strategy_threshold_schedule.parquet", index=False)

    pd.DataFrame({"feature": ["state_shock"], "coverage": [1.0]}).to_csv(
        root / "market_state_feature_coverage.csv", index=False
    )
    state_diag = pd.DataFrame(
        {
            "state_level": ["observed_axis", "forecast"],
            "state_head": ["state_shock", "forecast_h6_shock_down"],
            "component_group": ["return_shock", "return_shock"],
            "aggregate_status": ["active", "active"],
            "folds_seen": [1, 1],
            "trained_folds": [1, 1],
            "fallback_folds": [0, 0],
            "shadow_disabled_folds": [0, 0],
            "active_fold_share": [1.0, 1.0],
            "fallback_fold_share": [0.0, 0.0],
            "mean_source_count": [2.0, 2.0],
            "mean_validation_rows": [10.0, 10.0],
            "mean_validation_top_decile_lift": [None, 0.1],
            "mean_tail_average_precision": [None, 0.2],
            "mean_tail_ap_lift_p90": [None, 0.1],
            "mean_tail_brier_p90": [None, 0.2],
            "mean_tail_ece_5bin": [None, 0.1],
            "mean_tail_false_alarm_rate_p90": [None, 0.1],
            "mean_tail_recall_p90": [None, 0.3],
            "collapsed_folds": [0, 0],
            "positive_validation_lift_share": [None, 1.0],
            "mean_oof_coverage": [None, 1.0],
            "min_oof_coverage": [None, 1.0],
            "mean_target_rows": [None, 10.0],
            "mean_target_std": [None, 0.2],
            "status_counts": ['{"active": 1}', '{"active": 1}'],
            "disable_reasons": [None, None],
        }
    )
    state_diag.to_csv(root / "market_state_head_diagnostics.csv", index=False)
    registry = pd.DataFrame(
        {
            **state_diag.to_dict(orient="list"),
            "recommended_status": ["shadow", "shadow"],
            "activation_registry_version": [
                "market_state_activation_registry_v1",
                "market_state_activation_registry_v1",
            ],
        }
    )
    registry.to_csv(root / "market_state_activation_registry.csv", index=False)
    registry.drop(columns=["recommended_status", "activation_registry_version"]).to_csv(
        root / "walkforward_state_head_registry.csv", index=False
    )
    pd.DataFrame({"strategy_id": ["s1"], "rank": [0.8], "mu": [0.01]}).to_csv(
        root / "strategy_rank_outcome_curves.csv", index=False
    )
    pd.DataFrame({"timestamp": pd.to_datetime(["2026-06-01T01:00:00Z"]), "resid": [0.0]}).to_parquet(
        root / "strategy_residual_target_ledger.parquet", index=False
    )
    pd.DataFrame(
        {
            "fold": [1, 1, 1, 1],
            "arm": ["S1_observed_axes_shared_response"] * 4,
            "scope": ["all"] * 4,
            "scope_value": ["all"] * 4,
            "state_feature": ["state_shock"] * 4,
            "target": [
                "pred_resid_utility",
                "pred_resid_utility_lcb",
                "pred_resid_full_sl",
                "pred_resid_timeout",
            ],
            "rows": [10, 10, 10, 10],
            "state_q10": [0.1, 0.1, 0.1, 0.1],
            "state_q90": [0.9, 0.9, 0.9, 0.9],
            "target_mean_state_q10": [0.0, -0.01, 0.01, 0.02],
            "target_mean_state_q90": [0.01, 0.0, 0.02, 0.03],
            "target_q90_minus_q10": [0.01, 0.01, 0.01, 0.01],
            "pearson": [0.1, 0.1, 0.1, 0.1],
            "spearman": [0.1, 0.1, 0.1, 0.1],
        }
    ).to_csv(root / "strategy_state_effect_matrix.csv", index=False)

    action_audit = schedule.copy()
    action_audit["baseline_accepted"] = [1, 1]
    action_audit["current_accepted"] = [0, 1]
    action_audit["overlap"] = [0, 1]
    action_audit["entrants"] = [0, 0]
    action_audit["removed"] = [1, 0]
    action_audit["entrant_net_pnl"] = [0.0, 0.0]
    action_audit["removed_net_pnl"] = [-1.0, 0.0]
    action_audit["net_replacement_pnl"] = [1.0, 0.0]
    action_audit["same_key_net_pnl_delta"] = [0.1, 0.0]
    action_audit["net_action_pnl_delta"] = [1.1, 0.0]
    action_audit["removed_loss_avoided"] = [1.0, 0.0]
    action_audit["removed_winner_pnl_sacrificed"] = [0.0, 0.0]
    action_audit["defensive_success"] = [1.0, 0.0]
    action_audit.to_csv(root / "strategy_threshold_action_audit.csv", index=False)
    action_audit.to_csv(root / "walkforward_threshold_action_edge_validation.csv", index=False)

    pd.DataFrame(
        {
            "arm": ["S0_baseline_static_thresholds", "S1_observed_axes_shared_response"],
            "head": ["short_asset", "short_asset"],
            "threshold_raised": [0, 1],
            "trade_count": [1, 1],
            "net_pnl": [1.0, 1.1],
            "gross_pnl": [1.2, 1.3],
            "cost_pnl": [0.2, 0.2],
            "mean_net_return": [0.01, 0.011],
            "win_rate": [1.0, 1.0],
            "full_sl_rate": [0.0, 0.0],
            "timeout_rate": [0.0, 0.0],
            "mean_threshold_delta": [0.0, 0.01],
            "mean_risk_severity": [None, 0.2],
            "force_base_share": [None, 0.0],
            "mean_prediction_coverage": [None, 1.0],
            "mean_min_prediction_coverage": [None, 0.8],
            "mean_state_ood_score": [None, 0.1],
            "max_state_ood_score": [None, 0.1],
            "mean_state_ood_cutoff": [None, 0.8],
            "mean_state_ood_share": [None, 0.0],
            "mean_pred_utility": [None, 0.01],
            "mean_pred_lcb": [None, 0.0],
            "mean_pred_full_sl": [None, 0.1],
            "mean_pred_timeout": [None, 0.05],
            "mean_tail_candidate_count": [None, 2.0],
            "mean_suppressed_candidate_count": [None, 1.0],
            "mean_tail_lcb_q25": [None, 0.0],
            "mean_tail_pred_full_sl": [None, 0.1],
            "mean_tail_pred_timeout": [None, 0.05],
            "mean_predicted_removed_loss_avoided": [None, 1.0],
            "mean_predicted_removed_winner_sacrificed": [None, 0.2],
            "mean_predicted_action_edge": [None, 0.8],
            "mean_action_edge_per_suppressed": [None, 0.8],
            "fold": [1, 1],
        }
    ).to_csv(root / "walkforward_controller_state_diagnostics.csv", index=False)

    action_utility = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "scope": ["all"],
            "scope_value": ["all"],
            "baseline_accepted": [2],
            "current_accepted": [1],
            "overlap": [1],
            "entrants": [0],
            "removed": [1],
            "entrant_net_pnl": [0.0],
            "removed_net_pnl": [-1.0],
            "net_replacement_pnl": [1.0],
            "same_key_net_pnl_delta": [0.1],
            "net_action_pnl_delta": [1.1],
            "removed_loss_avoided": [1.0],
            "removed_winner_pnl_sacrificed": [0.0],
            "defensive_success": [1.0],
            "fold": [1],
        }
    )
    action_utility.to_csv(root / "walkforward_threshold_action_utility.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "fold": [1],
            "predicted_action_edge_bucket": ["high"],
            "threshold_raised": [1],
            "schedule_rows": [2],
            "baseline_accepted": [2],
            "current_accepted": [1],
            "entrants": [0],
            "removed": [1],
            "mean_threshold_delta": [0.01],
            "mean_predicted_action_edge": [0.8],
            "sum_predicted_action_edge": [0.8],
            "net_replacement_pnl": [1.0],
            "same_key_net_pnl_delta": [0.1],
            "net_action_pnl_delta": [1.1],
            "removed_loss_avoided": [1.0],
            "removed_winner_pnl_sacrificed": [0.0],
            "defensive_success": [1.0],
            "realized_minus_predicted_action_edge": [0.3],
        }
    ).to_csv(root / "walkforward_threshold_action_edge_bucket_performance.csv", index=False)

    suppression_utility = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "scope": ["all"],
            "scope_value": ["all"],
            "suppressed_candidates": [1],
            "raised_schedule_count": [1],
            "mean_suppressed_per_raised_schedule": [1.0],
            "mean_threshold_delta": [0.01],
            "mean_risk_severity": [0.2],
            "suppressed_net_return_sum": [-1.0],
            "mean_suppressed_net_return": [-1.0],
            "suppressed_loss_avoided": [1.0],
            "suppressed_winner_pnl_sacrificed": [0.0],
            "realized_defensive_success": [1.0],
            "realized_defensive_success_per_candidate": [1.0],
            "suppressed_win_rate": [0.0],
            "suppressed_full_sl_rate": [1.0],
            "suppressed_timeout_rate": [0.0],
            "mean_predicted_action_edge": [0.8],
            "sum_predicted_action_edge": [0.8],
            "fold": [1],
        }
    )
    suppression_utility.to_csv(root / "walkforward_threshold_candidate_suppression_utility.csv", index=False)
    suppression_utility.to_csv(root / "walkforward_threshold_baseline_accepted_suppression_utility.csv", index=False)
    suppression_aggregate = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "scope": ["all"],
            "scope_value": ["all"],
            "folds_with_suppression": [1],
            "suppressed_candidates": [1],
            "suppressed_net_return_sum": [-1.0],
            "suppressed_loss_avoided": [1.0],
            "suppressed_winner_pnl_sacrificed": [0.0],
            "realized_defensive_success": [1.0],
            "positive_suppression_fold_share": [1.0],
            "mean_suppressed_full_sl_rate": [1.0],
            "mean_suppressed_timeout_rate": [0.0],
        }
    )
    suppression_aggregate.to_csv(root / "walkforward_threshold_candidate_suppression_aggregate.csv", index=False)
    suppression_aggregate.to_csv(root / "walkforward_threshold_baseline_accepted_suppression_aggregate.csv", index=False)
    pd.DataFrame(
        {
            "state_head": ["state_shock"],
            "action_arm_hint": ["S1_observed_axes_shared_response"],
            "loo_replay_folds": [1],
            "loo_mode": ["neutralized_state"],
            "loo_median_increment_net_pnl": [0.1],
            "loo_mean_increment_net_pnl": [0.1],
            "loo_q25_increment_net_pnl": [0.1],
            "loo_positive_increment_share": [1.0],
            "loo_mean_accepted_jaccard": [0.9],
            "loo_mean_delta_trade_count": [0.0],
            "loo_mean_threshold_raise_delta": [0.0],
            "loo_state_head_defensive_success": [0.1],
            "loo_state_head_median_defensive_success": [0.1],
            "loo_state_head_positive_defensive_share": [1.0],
            "loo_state_head_loss_avoided": [0.2],
            "loo_state_head_winner_pnl_sacrificed": [0.1],
            "loo_state_head_net_action_pnl_delta": [0.1],
        }
    ).to_csv(root / "market_state_leave_one_head_out_aggregate.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S0_baseline_static_thresholds", "S1_observed_axes_shared_response"],
            "trade_count": [2, 1],
            "net_pnl": [1.0, 1.1],
            "gross_pnl": [1.4, 1.3],
            "cost_pnl": [0.4, 0.2],
            "cost_to_abs_gross": [0.285, 0.154],
            "compounded_return": [0.01, 0.011],
            "max_drawdown": [-0.1, -0.09],
            "worst_24h_net_pnl": [-0.5, -0.4],
            "full_sl_rate": [0.5, 0.0],
            "timeout_rate": [0.0, 0.0],
            "avg_open_positions": [1.0, 1.0],
            "mean_threshold_delta": [0.0, 0.01],
            "p75_threshold_delta": [0.0, 0.01],
            "max_threshold_delta": [0.0, 0.01],
            "share_threshold_raised": [0.0, 0.5],
            "fold": [1, 1],
        }
    ).to_csv(root / "portfolio_replay_summary.csv", index=False)
    pd.DataFrame(
        {
            "arm": [
                "S0_baseline_static_thresholds",
                "S0_baseline_static_thresholds",
                "S1_observed_axes_shared_response",
            ],
            "head": ["short_asset", "short_boll", "short_asset"],
            "trade_count": [1, 1, 1],
            "win_rate": [1.0, 0.0, 1.0],
            "net_pnl": [1.0, 0.0, 1.1],
            "gross_pnl": [1.2, 0.2, 1.3],
            "cost_pnl": [0.2, 0.2, 0.2],
            "mean_net_return": [0.01, 0.0, 0.011],
            "q05_net_return": [0.01, -0.01, 0.011],
            "full_sl_rate": [0.0, 1.0, 0.0],
            "timeout_rate": [0.0, 0.0, 0.0],
            "fold": [1, 1, 1],
        }
    ).to_csv(root / "portfolio_replay_by_head.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S0_baseline_static_thresholds", "S1_observed_axes_shared_response"],
            "accepted": [2, 1],
            "overlap_with_baseline": [2, 1],
            "new_vs_baseline": [0, 0],
            "removed_vs_baseline": [0, 1],
            "jaccard_vs_baseline": [1.0, 0.5],
            "position_size_sum": [2.0, 1.0],
            "position_size_mean": [1.0, 1.0],
            "entrant_net_pnl": [0.0, 0.0],
            "removed_net_pnl": [0.0, -1.0],
            "net_replacement_pnl": [0.0, 1.0],
            "removed_loss_avoided": [0.0, 1.0],
            "removed_winner_pnl_sacrificed": [0.0, 0.0],
            "defensive_success": [0.0, 1.0],
            "fold": [1, 1],
        }
    ).to_csv(root / "walkforward_overlap.csv", index=False)
    pd.DataFrame(
        {
            "fold": [1],
            "arm": ["S0_baseline_static_thresholds"],
            "timestamp": pd.to_datetime(["2026-06-01T01:00:00Z"]),
            "symbol": ["BTC_USD"],
            "side": ["short"],
            "strategy_id": ["s1"],
            "net_pnl": [1.0],
        }
    ).to_parquet(root / "accepted_trades.parquet", index=False)
    _write_artifact_hashes(root)


def _write_artifact_hashes(root: Path) -> None:
    artifacts = {}
    names = [name for name in audit.REQUIRED_ARTIFACTS if name != "artifact_hashes.json"]
    names.append("market_state_lgbm_models.joblib")
    names.extend(audit.OPTIONAL_HASHED_ARTIFACTS)
    for name in names:
        path = root / name
        artifacts[path.stem] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
        }
    (root / "artifact_hashes.json").write_text(
        json.dumps({"hash_version": "sha256_artifact_hashes_v1", "artifacts": artifacts}),
        encoding="utf-8",
    )


def _source_audit_payload() -> dict:
    return _valid_manifest()["source_contract_audit"]


def _bundle_feature_contract_payload() -> dict:
    return {
        "contract_version": "market_state_feature_contract_v1",
        "rank_contract": "short_boll_timestamp_rank",
        "disabled_heads": ["long_bars", "long_dist"],
        "active_heads": ["short_asset", "short_boll"],
        "source_contract_audit": _source_audit_payload(),
        "state_join_validation": {
            "score": {
                "state_join_timestamp_constant": True,
                "max_state_values_per_timestamp": 1,
            }
        },
        "invariants": {
            "one_market_state_row_per_timestamp": True,
            "state_join_timestamp_constant": True,
            "market_state_uses_strategy_ids": False,
            "market_state_uses_model_predictions": False,
            "market_state_uses_ranks": False,
            "market_state_uses_candidate_counts": False,
            "market_state_uses_portfolio_pnl": False,
            "market_state_uses_realized_strategy_outcomes": False,
            "actual_order_book_features_allowed": False,
            "candidate_population_fallback_enabled": False,
            "candidate_population_fallback_is_production_safe": False,
            "controller_changes_scores_or_ranks": False,
            "controller_changes_auction_ordering": False,
            "controller_can_lower_thresholds": False,
            "latent_gmm_active_controller_input": False,
        },
        "source_schema": {
            "feature_store_columns": ["mkt_ret_eq_1h"],
            "observed_axis_columns": ["state_shock"],
            "state_feature_columns": ["state_shock"],
            "response_feature_columns": ["normalized_rank_score", "state_shock"],
        },
    }


def _write_minimal_scored_shadow_bundle(root: Path) -> None:
    ts = pd.to_datetime(["2026-06-15T00:00:00Z", "2026-06-15T01:00:00Z"])
    pd.DataFrame(
        {
            "split": ["score", "score"],
            "state_level": ["forecast", "forecast"],
            "timestamp": ts,
            "state_shock": [0.1, 0.2],
            "forecast_h6_shock_down": [0.3, 0.4],
        }
    ).to_parquet(root / "market_state_timestamp_panel.parquet", index=False)
    pd.DataFrame({"column": ["state_shock"], "finite_share": [1.0]}).to_csv(
        root / "market_state_feature_coverage.csv",
        index=False,
    )
    (root / "market_state_feature_contract.json").write_text(
        json.dumps(_bundle_feature_contract_payload()),
        encoding="utf-8",
    )
    schedule = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy_id": ["s_asset", "s_boll"],
            "head": ["short_asset", "short_boll"],
            "base_threshold": [0.70, 0.70],
            "raw_state_threshold": [0.70, 0.70],
            "state_threshold": [0.70, 0.70],
            "threshold_action_enabled": [False, False],
            "force_base_threshold": [True, True],
            "controller_reason": ["head_not_enabled_for_threshold_action"] * 2,
        }
    )
    schedule.to_parquet(root / "strategy_threshold_schedule.parquet", index=False)
    schedule.to_csv(root / "controller_schedule.csv", index=False)
    shadow_schedule = schedule.copy()
    shadow_schedule["raw_state_threshold"] = [0.74, 0.76]
    shadow_schedule["state_threshold"] = [0.74, 0.76]
    shadow_schedule["threshold_action_enabled"] = [True, True]
    shadow_schedule["force_base_threshold"] = [False, False]
    shadow_schedule["controller_reason"] = ["rank_grid_penalty", "rank_grid_penalty"]
    shadow_schedule["arm"] = "S2_observed_forecast_shared_response__shadow_proposed"
    shadow_schedule.to_parquet(root / "shadow_controller_proposed_schedule.parquet", index=False)
    shadow_schedule.to_csv(root / "shadow_controller_proposed_schedule.csv", index=False)
    action_audit = pd.DataFrame(
        {
            "scope": ["all"],
            "scope_value": ["all"],
            "schedule_rows": [2],
            "threshold_raised_count": [0],
            "threshold_raised_share": [0.0],
            "force_base_count": [2],
            "force_base_share": [1.0],
            "mean_base_threshold": [0.70],
            "mean_state_threshold": [0.70],
            "mean_threshold_delta": [0.0],
            "max_threshold_delta": [0.0],
        }
    )
    action_audit.to_csv(root / "strategy_threshold_action_audit.csv", index=False)
    shadow_audit = action_audit.copy()
    shadow_audit["threshold_raised_count"] = [2]
    shadow_audit["threshold_raised_share"] = [1.0]
    shadow_audit["force_base_count"] = [0]
    shadow_audit["force_base_share"] = [0.0]
    shadow_audit["mean_state_threshold"] = [0.75]
    shadow_audit["mean_threshold_delta"] = [0.05]
    shadow_audit["max_threshold_delta"] = [0.06]
    shadow_audit.to_csv(root / "shadow_threshold_action_audit.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S2_observed_forecast_shared_response__shadow_proposed"],
            "scope": ["all"],
            "scope_value": ["all"],
            "suppressed_candidates": [3],
            "raised_schedule_count": [2],
            "mean_threshold_delta": [0.05],
            "suppressed_loss_avoided": [1.2],
            "suppressed_winner_pnl_sacrificed": [0.4],
            "realized_defensive_success": [0.8],
            "suppressed_full_sl_rate": [0.33],
            "suppressed_timeout_rate": [0.0],
        }
    ).to_csv(root / "shadow_threshold_candidate_suppression_utility.csv", index=False)
    pd.DataFrame({"prediction": [0.1, 0.2]}).to_parquet(root / "controller_predictions.parquet", index=False)
    pd.DataFrame({"timestamp": ts, "strategy_id": ["s_asset", "s_boll"]}).to_parquet(
        root / "controller_scored_candidates.parquet",
        index=False,
    )
    accepted = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTC_USD", "ETH_USD"],
            "side": ["short", "short"],
            "strategy_id": ["s_asset", "s_boll"],
            "head": ["short_asset", "short_boll"],
            "net_pnl": [1.0, -0.2],
        }
    )
    accepted.to_parquet(root / "accepted_trades.parquet", index=False)
    pd.DataFrame({"timestamp": ts, "strategy_id": ["s_asset", "s_boll"], "accepted": [True, True]}).to_parquet(
        root / "decisions.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "arm": ["S2_observed_forecast_shared_response"],
            "trade_count": [2],
            "net_pnl": [0.8],
            "gross_pnl": [1.0],
            "cost_pnl": [0.2],
            "full_sl_rate": [0.5],
            "timeout_rate": [0.0],
            "mean_threshold_delta": [0.0],
            "max_threshold_delta": [0.0],
            "share_threshold_raised": [0.0],
        }
    ).to_csv(root / "controller_replay_summary.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S2_observed_forecast_shared_response"],
            "head": ["short_asset"],
            "trade_count": [1],
            "win_rate": [1.0],
            "net_pnl": [1.0],
        }
    ).to_csv(root / "controller_replay_by_head.csv", index=False)
    (root / "strategy_threshold_controller_config.json").write_text(
        json.dumps({"config_version": "strategy_threshold_controller_config_v1"}),
        encoding="utf-8",
    )
    outputs = {
        path.stem: str(path)
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = _valid_manifest()
    manifest.update(
        {
            "generated_by": "score_market_state_controller_bundle",
            "selected_arm": "S2_observed_forecast_shared_response",
            "controller_execution_enabled": False,
            "controller_enabled_heads": [],
            "controller_enabled_scope": "disabled_by_activation_registry",
            "shadow_controller_only": True,
            "shadow_controller_enabled_heads": ["short_asset", "short_boll"],
            "shadow_controller_enabled_scope": "all_active_heads",
            "controller": {
                "penalty_only": True,
                "execution_enabled": False,
                "controller_execution_enabled": False,
                "controller_enabled_heads": [],
                "controller_enabled_scope": "disabled_by_activation_registry",
                "shadow_controller_only": True,
                "changes_scores_or_ranks": False,
                "changes_auction_ordering": False,
            },
            "outputs": outputs,
            "output_sha256": {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in outputs.items()
            },
        }
    )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_market_state_artifact_audit_accepts_complete_artifacts(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)

    assert audit.audit_artifacts(tmp_path) == []


def test_market_state_artifact_audit_accepts_scored_shadow_bundle(tmp_path: Path) -> None:
    _write_minimal_scored_shadow_bundle(tmp_path)

    assert audit.audit_manifest(json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))) == []
    assert audit.audit_artifacts(tmp_path) == []


def test_market_state_artifact_audit_rejects_shadow_bundle_with_applied_threshold_raise(tmp_path: Path) -> None:
    _write_minimal_scored_shadow_bundle(tmp_path)
    schedule = pd.read_parquet(tmp_path / "strategy_threshold_schedule.parquet")
    schedule.loc[0, "state_threshold"] = 0.74
    schedule.to_parquet(tmp_path / "strategy_threshold_schedule.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any("expected to be no-op but raises thresholds" in failure for failure in failures)


def test_market_state_artifact_audit_reader_accepts_empty_csv(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty_metric.csv"
    empty_csv.write_text("", encoding="utf-8")

    frame = audit._read_frame(empty_csv)

    assert frame.empty


def test_market_state_artifact_audit_rejects_missing_state_metric_column(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_csv(tmp_path / "market_state_head_diagnostics.csv")
    frame = frame.drop(columns=["mean_tail_brier_p90"])
    frame.to_csv(tmp_path / "market_state_head_diagnostics.csv", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any("market_state_head_diagnostics missing numeric metric columns" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_missing_response_effect_target(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_csv(tmp_path / "strategy_state_effect_matrix.csv")
    frame = frame.loc[frame["target"].ne("pred_resid_timeout")].copy()
    frame.to_csv(tmp_path / "strategy_state_effect_matrix.csv", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert "strategy_state_effect_matrix missing response targets: ['pred_resid_timeout']" in failures


def test_market_state_artifact_audit_rejects_missing_leave_one_defensive_metric(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_csv(tmp_path / "market_state_leave_one_head_out_aggregate.csv")
    frame = frame.drop(columns=["loo_state_head_winner_pnl_sacrificed"])
    frame.to_csv(tmp_path / "market_state_leave_one_head_out_aggregate.csv", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any(
        "market_state_leave_one_head_out_aggregate missing columns" in failure
        and "loo_state_head_winner_pnl_sacrificed" in failure
        for failure in failures
    )


def test_market_state_artifact_audit_rejects_nonfinite_portfolio_metric(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_csv(tmp_path / "portfolio_replay_summary.csv")
    frame.loc[0, "net_pnl"] = None
    frame.to_csv(tmp_path / "portfolio_replay_summary.csv", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert "portfolio_replay_summary.net_pnl has non-finite metric values: 1" in failures


def test_market_state_artifact_audit_rejects_duplicate_state_rows(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "market_state_oof_predictions.parquet")
    pd.concat([frame, frame], ignore_index=True).to_parquet(
        tmp_path / "market_state_oof_predictions.parquet", index=False
    )

    failures = audit.audit_artifacts(tmp_path)

    assert any("market_state_oof_predictions has duplicate market-state rows" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_oof_state_value_mismatch(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "market_state_oof_predictions.parquet")
    frame.loc[0, "state_shock"] = 0.9
    frame.to_parquet(tmp_path / "market_state_oof_predictions.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any("market_state_oof_predictions values differ from timestamp panel" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_response_oof_without_state_oof_key(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "strategy_response_oof_predictions.parquet")
    frame.loc[:, "arm"] = "missing_state_arm"
    frame.to_parquet(tmp_path / "strategy_response_oof_predictions.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any("strategy_response_oof_predictions keys missing from market_state_oof_predictions" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_response_oof_without_outer_fold_state_contract(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "strategy_response_oof_predictions.parquet")
    frame.loc[:, "state_prediction_contract"] = "in_sample_train_state_scores"
    frame.to_parquet(tmp_path / "strategy_response_oof_predictions.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any(
        "strategy_response_oof_predictions state_prediction_contract is not outer_fold_validation_state_scores"
        in failure
        for failure in failures
    )


def test_market_state_artifact_audit_rejects_state_join_variance(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "strategy_response_oof_predictions.parquet")
    frame.loc[1, "state_ood_score"] = 0.9
    frame.to_parquet(tmp_path / "strategy_response_oof_predictions.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any("state_ood_score varies within fold/arm/timestamp groups" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_threshold_lowering(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "strategy_threshold_schedule.parquet")
    frame.loc[0, "state_threshold"] = 0.5
    frame.to_parquet(tmp_path / "strategy_threshold_schedule.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert "strategy_threshold_schedule lowers thresholds below base: 1" in failures


def test_market_state_artifact_audit_rejects_low_coverage_without_base_fallback(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "strategy_threshold_schedule.parquet")
    frame.loc[0, "prediction_coverage"] = 0.2
    frame.loc[0, "min_prediction_coverage"] = 0.8
    frame.loc[0, "force_base_threshold"] = False
    frame.loc[0, "state_threshold"] = 0.72
    frame.loc[0, "controller_reason"] = "threshold_raised"
    frame.to_parquet(tmp_path / "strategy_threshold_schedule.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert "missing/OOD fallback rows are not force_base_threshold: 1" in failures
    assert "missing/OOD fallback rows do not equal base threshold: 1" in failures
    assert "missing/OOD fallback rows have unexpected controller_reason: 1" in failures


def test_market_state_artifact_audit_rejects_ood_without_base_fallback(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "strategy_threshold_schedule.parquet")
    frame.loc[0, "state_ood_share"] = 1.0
    frame.loc[0, "state_ood_score_max"] = 2.0
    frame.loc[0, "state_ood_cutoff"] = 1.0
    frame.loc[0, "force_base_threshold"] = False
    frame.loc[0, "state_threshold"] = 0.72
    frame.loc[0, "controller_reason"] = "threshold_raised"
    frame.to_parquet(tmp_path / "strategy_threshold_schedule.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert "missing/OOD fallback rows are not force_base_threshold: 1" in failures
    assert "missing/OOD fallback rows do not equal base threshold: 1" in failures
    assert "missing/OOD fallback rows have unexpected controller_reason: 1" in failures


def test_market_state_artifact_audit_rejects_static_baseline_action(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_csv(tmp_path / "portfolio_replay_summary.csv")
    frame.loc[0, "mean_threshold_delta"] = 0.01
    frame.to_csv(tmp_path / "portfolio_replay_summary.csv", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert "portfolio_replay_summary.mean_threshold_delta is nonzero for static baseline" in failures


def test_market_state_artifact_audit_rejects_static_baseline_overlap_change(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_csv(tmp_path / "walkforward_overlap.csv")
    frame.loc[0, "jaccard_vs_baseline"] = 0.5
    frame.to_csv(tmp_path / "walkforward_overlap.csv", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert "walkforward_overlap.jaccard_vs_baseline != 1 for static baseline" in failures


def test_market_state_artifact_audit_rejects_duplicate_accepted_trade_keys(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "accepted_trades.parquet")
    pd.concat([frame, frame], ignore_index=True).to_parquet(tmp_path / "accepted_trades.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert "accepted_trades duplicate decision keys by fold/arm: 1" in failures


def test_market_state_artifact_audit_allows_empty_suppression_reports_with_schema(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    suppression_columns = [
        "fold",
        "arm",
        "scope",
        "scope_value",
        "suppressed_candidates",
        "raised_schedule_count",
        "mean_suppressed_per_raised_schedule",
        "mean_threshold_delta",
        "mean_risk_severity",
        "suppressed_net_return_sum",
        "mean_suppressed_net_return",
        "suppressed_loss_avoided",
        "suppressed_winner_pnl_sacrificed",
        "realized_defensive_success",
        "realized_defensive_success_per_candidate",
        "suppressed_win_rate",
        "suppressed_full_sl_rate",
        "suppressed_timeout_rate",
        "mean_predicted_action_edge",
        "sum_predicted_action_edge",
    ]
    suppression_aggregate_columns = [
        "arm",
        "scope",
        "scope_value",
        "folds_with_suppression",
        "suppressed_candidates",
        "suppressed_net_return_sum",
        "suppressed_loss_avoided",
        "suppressed_winner_pnl_sacrificed",
        "realized_defensive_success",
        "positive_suppression_fold_share",
        "mean_suppressed_full_sl_rate",
        "mean_suppressed_timeout_rate",
    ]
    for name in (
        "walkforward_threshold_candidate_suppression_utility.csv",
        "walkforward_threshold_baseline_accepted_suppression_utility.csv",
    ):
        pd.DataFrame(columns=suppression_columns).to_csv(tmp_path / name, index=False)
    for name in (
        "walkforward_threshold_candidate_suppression_aggregate.csv",
        "walkforward_threshold_baseline_accepted_suppression_aggregate.csv",
    ):
        pd.DataFrame(columns=suppression_aggregate_columns).to_csv(tmp_path / name, index=False)
    _write_artifact_hashes(tmp_path)

    failures = audit.audit_artifacts(tmp_path)

    assert not [failure for failure in failures if "suppression" in failure]


def test_market_state_artifact_audit_rejects_forbidden_state_columns(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "market_state_timestamp_panel.parquet")
    frame["strategy_id"] = "leaky_strategy"
    frame.to_parquet(tmp_path / "market_state_timestamp_panel.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any("contains forbidden market-state columns" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_order_book_state_columns(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    frame = pd.read_parquet(tmp_path / "market_state_oof_predictions.parquet")
    frame["ask_depth"] = 1.0
    frame.to_parquet(tmp_path / "market_state_oof_predictions.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any("contains actual order-book-like columns" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_semantic_model_state_columns(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    panel = pd.read_parquet(tmp_path / "market_state_timestamp_panel.parquet")
    panel["state_policy_rank_pct"] = 0.8
    panel["forecast_net_pnl"] = 1.0
    panel.to_parquet(tmp_path / "market_state_timestamp_panel.parquet", index=False)
    oof = pd.read_parquet(tmp_path / "market_state_oof_predictions.parquet")
    oof["state_model_score"] = 0.9
    oof.to_parquet(tmp_path / "market_state_oof_predictions.parquet", index=False)

    failures = audit.audit_artifacts(tmp_path)

    assert any(
        "market_state_timestamp_panel contains strategy/model/performance-like" in failure
        for failure in failures
    )
    assert any(
        "market_state_oof_predictions contains strategy/model/performance-like" in failure
        for failure in failures
    )


def test_market_state_artifact_audit_rejects_unsafe_feature_contract_invariant(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "market_state_feature_contract.json").read_text(encoding="utf-8"))
    payload["invariants"]["market_state_uses_ranks"] = True
    (tmp_path / "market_state_feature_contract.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert "market_state_feature_contract.invariants.market_state_uses_ranks is not false" in failures


def test_market_state_artifact_audit_rejects_activation_registry_dropped_state_leak(
    tmp_path: Path,
) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "market_state_feature_contract.json").read_text(encoding="utf-8"))
    payload["state_activation_filter"] = {
        "enforced": True,
        "reason": "activation_registry_active_candidate_filter",
        "input_state_feature_count": 2,
        "active_state_feature_count": 1,
        "dropped_state_feature_count": 1,
        "active_state_feature_columns": ["state_shock"],
        "dropped_state_feature_columns": ["forecast_h6_shock_down"],
    }
    payload["source_schema"]["state_feature_columns"] = [
        "state_shock",
        "forecast_h6_shock_down",
    ]
    payload["source_schema"]["response_feature_columns"] = [
        "normalized_rank_score",
        "state_shock",
        "forecast_h6_shock_down",
    ]
    (tmp_path / "market_state_feature_contract.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert any("outside activation filter" in failure for failure in failures)
    assert any("contains dropped activation-registry state features" in failure for failure in failures)


def test_market_state_artifact_audit_accepts_activation_registry_filtered_state_contract(
    tmp_path: Path,
) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "market_state_feature_contract.json").read_text(encoding="utf-8"))
    payload["state_activation_filter"] = {
        "enforced": True,
        "reason": "activation_registry_active_candidate_filter",
        "input_state_feature_count": 2,
        "active_state_feature_count": 1,
        "dropped_state_feature_count": 1,
        "active_state_feature_columns": ["state_shock"],
        "dropped_state_feature_columns": ["forecast_h6_shock_down"],
    }
    payload["source_schema"]["state_feature_columns"] = ["state_shock"]
    payload["source_schema"]["response_feature_columns"] = [
        "normalized_rank_score",
        "state_shock",
    ]
    (tmp_path / "market_state_feature_contract.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert not any("activation filter" in failure for failure in failures)
    assert not any("dropped activation-registry" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_missing_training_outcome_maturity_contract(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "market_state_feature_contract.json").read_text(encoding="utf-8"))
    payload["validation"].pop("training_outcome_maturity_contract_passed")
    payload["validation"].pop("training_immature_outcome_rows_dropped")
    (tmp_path / "market_state_feature_contract.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert (
        "market_state_feature_contract.validation.training_outcome_maturity_contract_passed is not true"
        in failures
    )
    assert (
        "market_state_feature_contract.validation.training_immature_outcome_rows_dropped is missing"
        in failures
    )


def test_market_state_artifact_audit_rejects_short_fold_embargo(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "market_state_feature_contract.json").read_text(encoding="utf-8"))
    payload["fold_definition"]["folds_built"][0]["valid_start"] = "2026-05-11T00:00:00+00:00"
    (tmp_path / "market_state_feature_contract.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert any("embargo 24h < required 96h" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_missing_target_cdf_reference(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = joblib.load(tmp_path / "market_state_target_cdfs.joblib")
    payload["missing_reference_count"] = 1
    joblib.dump(payload, tmp_path / "market_state_target_cdfs.joblib")

    failures = audit.audit_artifacts(tmp_path)

    assert "market_state_target_cdfs.missing_reference_count != 0" in failures


def test_market_state_artifact_audit_rejects_training_reference_without_observed_encoder(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = joblib.load(tmp_path / "market_state_training_reference.joblib")
    payload["fold_references"]["fold_1"].pop("observed_axis_encoder")
    joblib.dump(payload, tmp_path / "market_state_training_reference.joblib")

    failures = audit.audit_artifacts(tmp_path)

    assert (
        "market_state_training_reference.fold_references.fold_1.observed_axis_encoder is missing"
        in failures
    )


def test_market_state_artifact_audit_rejects_training_reference_without_low_coverage_fail_closed_source(
    tmp_path: Path,
) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = joblib.load(tmp_path / "market_state_training_reference.joblib")
    encoder = payload["fold_references"]["fold_1"]["observed_axis_encoder"]
    encoder["axis_sources"].pop("state_low_input_coverage")
    joblib.dump(payload, tmp_path / "market_state_training_reference.joblib")

    failures = audit.audit_artifacts(tmp_path)

    assert (
        "market_state_training_reference.fold_references.fold_1.observed_axis_encoder.axis_sources "
        "missing state_low_input_coverage"
        in failures
    )


def test_market_state_artifact_audit_rejects_training_reference_without_train_tail_reference(
    tmp_path: Path,
) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = joblib.load(tmp_path / "market_state_training_reference.joblib")
    fold = payload["fold_references"]["fold_1"]
    fold.pop("feature_store_tail_reference_quantiles")
    fold["feature_store_reports"]["train"].pop("tail_reference_quantiles")
    fold["feature_store_reports"]["valid"]["tail_reference_source"] = "self_window_reference"
    joblib.dump(payload, tmp_path / "market_state_training_reference.joblib")

    failures = audit.audit_artifacts(tmp_path)

    assert (
        "market_state_training_reference.fold_references.fold_1.feature_store_tail_reference_quantiles is missing"
        in failures
    )
    assert (
        "market_state_training_reference.fold_references.fold_1.feature_store_reports.valid."
        "tail_reference_source is not provided_train_reference"
        in failures
    )


def test_market_state_artifact_audit_rejects_training_reference_with_invalid_reference_scale(
    tmp_path: Path,
) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = joblib.load(tmp_path / "market_state_training_reference.joblib")
    encoder = payload["fold_references"]["fold_1"]["observed_axis_encoder"]
    encoder["column_refs"]["fs__mkt_ret_eq_1h__mean"]["scale"] = 0.0
    joblib.dump(payload, tmp_path / "market_state_training_reference.joblib")

    failures = audit.audit_artifacts(tmp_path)

    assert (
        "market_state_training_reference.fold_references.fold_1.observed_axis_encoder."
        "column_refs.fs__mkt_ret_eq_1h__mean.scale <= 0"
        in failures
    )


def test_market_state_artifact_audit_rejects_response_model_without_feature_order(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = joblib.load(tmp_path / "strategy_response_models.joblib")
    bundle = payload["fold_models"]["fold_1__S1_observed_axes_shared_response"]
    bundle["response_feature_columns"] = []
    joblib.dump(payload, tmp_path / "strategy_response_models.joblib")

    failures = audit.audit_artifacts(tmp_path)

    assert (
        "strategy_response_models.fold_1__S1_observed_axes_shared_response.response_feature_columns is missing"
        in failures
    )


def test_market_state_artifact_audit_rejects_response_model_without_oof_state_training_contract(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = joblib.load(tmp_path / "strategy_response_models.joblib")
    bundle = payload["fold_models"]["fold_1__S1_observed_axes_shared_response"]
    bundle["model_report"]["response_training_uses_oof_state_scores"] = False
    bundle["model_report"]["response_training_state_contract_passed"] = False
    bundle["model_report"]["learned_state_non_oof_columns"] = ["forecast_h6_shock_down"]
    joblib.dump(payload, tmp_path / "strategy_response_models.joblib")

    failures = audit.audit_artifacts(tmp_path)

    assert (
        "strategy_response_models.fold_1__S1_observed_axes_shared_response."
        "response_training_uses_oof_state_scores is not true"
        in failures
    )
    assert (
        "strategy_response_models.fold_1__S1_observed_axes_shared_response."
        "learned_state_non_oof_columns is not empty"
        in failures
    )


def test_market_state_artifact_audit_rejects_non_chronological_controller_validation(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "strategy_threshold_controller_config.json").read_text(encoding="utf-8"))
    payload["validation"]["chronological_complete_timestamp_folds"] = False
    (tmp_path / "strategy_threshold_controller_config.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert (
        "strategy_threshold_controller_config.validation.chronological_complete_timestamp_folds is not true"
        in failures
    )


def test_market_state_artifact_audit_rejects_unsafe_feature_contract_source_audit(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "market_state_feature_contract.json").read_text(encoding="utf-8"))
    payload["source_contract_audit"]["splits"]["train"]["candidate_fallback_enabled"] = True
    (tmp_path / "market_state_feature_contract.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert (
        "market_state_feature_contract.source_contract_audit.train.candidate_fallback_enabled is not false"
        in failures
    )


def test_market_state_artifact_audit_rejects_incomplete_universe_contract(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "market_state_universe_contract.json").read_text(encoding="utf-8"))
    payload["eligible_symbols"] = []
    payload["eligible_symbol_count"] = 0
    payload["fold_split_contracts"]["fold_1_train"]["candidate_independent"] = False
    (tmp_path / "market_state_universe_contract.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert "market_state_universe_contract.eligible_symbols is missing" in failures
    assert (
        "market_state_universe_contract.fold_split_contracts.fold_1_train.candidate_independent is not true"
        in failures
    )


def test_market_state_artifact_audit_rejects_actual_order_book_feature_name(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "market_state_feature_contract.json").read_text(encoding="utf-8"))
    payload["source_schema"]["feature_store_columns"].append("bid_ask_spread")
    (tmp_path / "market_state_feature_contract.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert any("source_schema contains unsafe feature names" in failure for failure in failures)


def test_market_state_artifact_audit_rejects_stale_artifact_hash(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    payload = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    payload["artifacts"]["manifest"]["sha256"] = "f" * 64
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(payload), encoding="utf-8")

    failures = audit.audit_artifacts(tmp_path)

    assert "artifact_hashes mismatched files: ['manifest:sha256']" in failures


def test_market_state_contract_audit_cli_writes_output_json(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    manifest = _valid_manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifact_hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    artifact_hashes["artifacts"]["manifest"]["bytes"] = (tmp_path / "manifest.json").stat().st_size
    artifact_hashes["artifacts"]["manifest"]["sha256"] = hashlib.sha256(
        (tmp_path / "manifest.json").read_bytes()
    ).hexdigest()
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(artifact_hashes), encoding="utf-8")
    output_path = tmp_path / "audit_result.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_market_state_controller_contract.py",
            str(tmp_path),
            "--require-null-selection",
            "--audit-artifacts",
            "--output-json",
            str(output_path),
        ],
        check=True,
        cwd=Path.cwd(),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["manifest_audit_enabled"] is True
    assert payload["artifact_audit_enabled"] is True
    assert "artifact_hashes_present_complete_and_verified" in payload["artifact_audit_checks"]
    assert "state_head_registries_present_and_versioned" in payload["artifact_audit_checks"]
    assert payload["failures"] == []


def test_market_state_contract_audit_cli_marks_manifest_only_as_not_completion_grade(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "manifest_only_audit_result.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_market_state_controller_contract.py",
            str(tmp_path),
            "--require-null-selection",
            "--output-json",
            str(output_path),
        ],
        check=True,
        cwd=Path.cwd(),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["audit_scope"] == "manifest_only"
    assert payload["artifact_audit_enabled"] is False
    assert payload["artifact_audit_required_for_completion"] is True
    assert payload["completion_grade_audit"] is False
    assert payload["completion_grade_passed"] is False
    assert payload["artifact_audit_checks"] == []
    assert payload["warnings"] == [
        "artifact audit not run; pass --audit-artifacts for completion-grade market-state controller audit"
    ]
