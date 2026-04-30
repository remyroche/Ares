import json

import pandas as pd
import pytest

from extreme_price_movements.inference.feature_generator import (
    get_inference_required_feature_keys,
)
from extreme_price_movements.inference.parity import (
    apply_strategy_acceptance_filter,
    calibrated_score_and_threshold,
    calibration_size_multiplier,
    load_profitable_sizer_strategy_filter,
    load_strategy_acceptance_filter,
    load_strategy_asset_exclusion_filter,
    passes_rank_filter,
    resolve_deployment_strategy_filter,
    strategy_core_id,
    strategy_id_matches,
    validate_calibration_artifacts,
    validate_deployment_model_coverage,
    validate_live_feature_contract,
    validate_meta_feature_contract_artifact,
    validate_required_feature_frames,
)


def test_load_strategy_acceptance_filter_reads_policy_artifact(tmp_path):
    run_id = "20260101_000000"
    p = tmp_path / "artifacts" / run_id
    p.mkdir(parents=True)
    payload = {
        "strategies": [
            {"strategy_id": "long_mr"},
            {"strategy_id": "short_tf"},
        ]
    }
    (p / "strategy_final_acceptation.json").write_text(json.dumps(payload))

    accepted = load_strategy_acceptance_filter(str(tmp_path), run_id)
    assert accepted == {"long_mr", "short_tf"}


def test_deployment_strategy_filter_intersects_holdout_profitable_and_policy(tmp_path):
    run_id = "20260101_000000"
    art = tmp_path / "artifacts" / run_id
    (art / "ridge_sizer").mkdir(parents=True)
    (art / "policy_params").mkdir(parents=True)

    (art / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {"strategy_id": "long_profitable"},
                    {"strategy_id": "long_unprofitable"},
                ]
            }
        )
    )
    (art / "ridge_sizer" / "strategy_params.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "profitable",
                        "wallet_pnl": 1.0,
                        "net_pnl": 2.0,
                    },
                    {
                        "strategy_id": "unprofitable",
                        "wallet_pnl": -1.0,
                        "net_pnl": 2.0,
                    },
                ]
            }
        )
    )
    (art / "policy_params" / "best_policy_params.json").write_text(
        json.dumps({"strategies": [{"strategy_id": "profitable"}]})
    )

    profitable = load_profitable_sizer_strategy_filter(str(tmp_path), run_id)
    assert "profitable" in profitable
    assert "unprofitable" not in profitable

    selected = resolve_deployment_strategy_filter(str(tmp_path), run_id)
    assert selected is not None
    assert "long_profitable" in selected
    assert "unprofitable" not in selected
    assert "long_unprofitable" not in selected


def test_deployment_strategy_filter_policy_selection_suffices_without_holdout(
    tmp_path,
):
    run_id = "20260101_000000"
    art = tmp_path / "artifacts" / run_id
    (art / "ridge_sizer").mkdir(parents=True)
    (art / "policy_params").mkdir(parents=True)

    (art / "policy_params" / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "generated_by": "policy_optimiser",
                "strategies": [
                    {
                        "strategy_id": "rule_a",
                        "strategy_for_inference": "rule_a",
                        "side": "short",
                        "selected": True,
                    }
                ],
            }
        )
    )
    (art / "ridge_sizer" / "strategy_params.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "rule_a",
                        "wallet_pnl": 1.0,
                        "net_pnl": 2.0,
                    }
                ]
            }
        )
    )
    (art / "policy_params" / "best_policy_params.json").write_text(
        json.dumps({"strategies": [{"strategy_id": "rule_a"}]})
    )

    selected = resolve_deployment_strategy_filter(str(tmp_path), run_id)

    assert selected == {"short_rule_a"}


def test_deployment_strategy_filter_uses_policy_when_sizer_allowlist_is_stale(
    tmp_path,
):
    run_id = "20260101_000000"
    art = tmp_path / "artifacts" / run_id
    (art / "ridge_sizer").mkdir(parents=True)
    (art / "policy_params").mkdir(parents=True)

    (art / "policy_params" / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "generated_by": "policy_optimiser",
                "strategies": [
                    {
                        "strategy_id": "rule_a",
                        "strategy_for_inference": "rule_a",
                        "side": "long",
                        "selected": True,
                    }
                ],
            }
        )
    )
    (art / "ridge_sizer" / "strategy_params.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "old_rule",
                        "wallet_pnl": 1.0,
                        "net_pnl": 2.0,
                    }
                ]
            }
        )
    )
    (art / "policy_params" / "best_policy_params.json").write_text(
        json.dumps({"strategies": [{"strategy_id": "rule_a"}]})
    )

    selected = resolve_deployment_strategy_filter(str(tmp_path), run_id)

    assert selected == {"long_rule_a"}


def test_strategy_asset_exclusion_filter_reads_strategy_for_inference(tmp_path):
    run_id = "20260101_000000"
    art = tmp_path / "artifacts" / run_id / "policy_params"
    art.mkdir(parents=True)
    (art / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_rule",
                        "selected": True,
                        "excluded_symbols": ["btc_usdt", "ETH/USDT"],
                    },
                    {
                        "strategy_id": "short_ignored",
                        "selected": False,
                        "excluded_symbols": ["SOL/USDT"],
                    },
                ]
            }
        )
    )

    exclusions = load_strategy_asset_exclusion_filter(str(tmp_path), run_id)

    assert exclusions["long_rule"] == {"BTC/USDT", "ETH/USDT"}
    assert exclusions["rule"] == {"BTC/USDT", "ETH/USDT"}
    assert "short_ignored" not in exclusions


def test_strategy_asset_exclusion_filter_skips_empty_root_artifact(tmp_path):
    run_id = "20260101_000000"
    art = tmp_path / "artifacts" / run_id
    (art / "policy_params").mkdir(parents=True)
    (art / "strategy_for_inference.json").write_text(
        json.dumps({"asset_exclusions": {"long_rule": []}})
    )
    (art / "policy_params" / "strategy_for_inference.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "long_rule",
                        "selected": True,
                        "excluded_symbols": ["DEXE/USDT"],
                    }
                ]
            }
        )
    )

    exclusions = load_strategy_asset_exclusion_filter(str(tmp_path), run_id)

    assert exclusions["long_rule"] == {"DEXE/USDT"}


def test_strategy_matching_normalizes_model_horizon_suffixes():
    accepted = {"complex_rule"}

    assert strategy_core_id("short_complex_rule_H10") == "complex_rule"
    assert strategy_id_matches("short_complex_rule_H10", accepted)


def test_strategy_matching_treats_side_aware_and_core_aliases_as_equivalent():
    assert strategy_id_matches("rule_a", {"long_rule_a"})
    assert strategy_id_matches("long_rule_a", {"rule_a"})


def test_apply_strategy_acceptance_filter_blocks_non_accepted():
    df = pd.DataFrame(
        {
            "strategy": ["long_mr", "long_tf", "short_mr"],
            "x": [1, 2, 3],
        }
    )
    out = apply_strategy_acceptance_filter(df, {"long_mr", "short_mr"}, "strategy")
    assert out["strategy"].tolist() == ["long_mr", "short_mr"]


def test_rank_filter_uses_p75_threshold_when_available():
    calibration_data = {
        "long_mr": {
            "p75_threshold": 0.60,
            "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
        }
    }
    calibrated, threshold = calibrated_score_and_threshold(
        raw_score=0.62,
        strategy_id="long_mr",
        calibration_data=calibration_data,
        default_threshold=0.5,
    )
    assert calibrated >= 0.62 - 1e-12
    assert threshold == 0.60
    assert passes_rank_filter(0.62, "long_mr", calibration_data)
    assert not passes_rank_filter(0.58, "long_mr", calibration_data)
    mult = calibration_size_multiplier(0.62, "long_mr", calibration_data)
    assert mult >= 1.0


def test_validate_calibration_artifacts_requires_contract(tmp_path):
    run_id = "20260101_000000"
    (tmp_path / "artifacts" / run_id / "ridge_sizer").mkdir(parents=True)
    calibration_data = {"long_mr": {"p75_threshold": 0.6, "calibration_curve": []}}
    # strict=False tolerates missing contract
    assert not validate_calibration_artifacts(
        str(tmp_path), run_id, calibration_data, strict=False
    )


def test_deployment_model_coverage_rejects_missing_meta_model():
    bundle = {
        "ridge_sizer": object(),
        "bucket_params": {"buckets": {"short_rule": {"sl_mult": 1.0}}},
        "bundle": {
            "alpha_models": {
                "short_rule": {"model": object(), "feat_cols": ["ret24h"]}
            },
            "meta_models": {"long_other": object()},
        },
    }

    with pytest.raises(ValueError, match="matching meta model missing"):
        validate_deployment_model_coverage(bundle, {"short_rule"})


def test_meta_feature_contract_required_when_meta_models_loaded(tmp_path):
    run_id = "20260101_000000"
    (tmp_path / "artifacts" / run_id).mkdir(parents=True)
    bundle = {"bundle": {"meta_models": {"short_rule": object()}}}

    with pytest.raises(ValueError, match="meta_feature_contract.json"):
        validate_meta_feature_contract_artifact(
            str(tmp_path),
            run_id,
            bundle,
            {"short_rule"},
            strict=True,
        )


def test_meta_feature_contract_validates_positional_mapping(tmp_path):
    class _Best:
        raw_selected_features = ["f0", "f1"]
        meta_feature_columns_ = ["ret24h", "base_probability_short_rule"]

    class _Meta:
        best_model = _Best()
        meta_feature_columns_ = ["ret24h", "base_probability_short_rule"]

    run_id = "20260101_000000"
    meta_dir = tmp_path / "artifacts" / run_id / "meta_oof"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta_feature_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "meta_models": {
                    "short_rule": {
                        "feature_columns": ["ret24h", "base_probability_short_rule"],
                        "positional_feature_mapping": {
                            "f0": "ret24h",
                            "f1": "base_probability_short_rule",
                        },
                        "n_features": 2,
                    }
                },
            }
        )
    )
    bundle = {"bundle": {"meta_models": {"short_rule": _Meta()}}}

    assert validate_meta_feature_contract_artifact(
        str(tmp_path),
        run_id,
        bundle,
        {"short_rule"},
        strict=True,
    )


def test_meta_feature_contract_rejects_live_unavailable_features(tmp_path):
    run_id = "20260101_000000"
    meta_dir = tmp_path / "artifacts" / run_id / "meta_oof"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta_feature_contract.json").write_text(
        json.dumps(
            {
                "meta_models": {
                    "long_rule": {
                        "feature_columns": ["ret24h", "reg_gate_target"],
                        "positional_feature_mapping": {
                            "f0": "ret24h",
                            "f1": "reg_gate_target",
                        },
                        "n_features": 2,
                    }
                }
            }
        )
    )
    bundle = {"bundle": {"meta_models": {"long_rule": object()}}}

    with pytest.raises(ValueError, match="live-unavailable"):
        validate_meta_feature_contract_artifact(
            str(tmp_path),
            run_id,
            bundle,
            {"long_rule"},
            strict=True,
        )


def test_required_feature_frames_reject_missing_keys_and_symbols():
    feats = {
        "ret24h": pd.DataFrame(
            {"BTC/USDT": [0.01]},
            index=pd.date_range("2026-01-01", periods=1, tz="UTC"),
        )
    }

    with pytest.raises(ValueError, match="Required inference features unavailable"):
        validate_required_feature_frames(
            feats,
            {"ret24h", "volatility_zscore"},
            symbols=["BTC/USDT", "ETH/USDT"],
        )


def test_live_required_features_exclude_target_derived_sizer_fields():
    class _Sizer:
        model_names_ = ["meta_pred", "reg_gate_target"]
        model_names_ridge_ = ["meta_pred", "reg_gate_target"]
        limit_offset_features_ = ["sizer_score_oof", "reg_gate_target"]

    bundle = {"ridge_sizer": _Sizer(), "bundle": {"alpha_models": {}}}

    required = get_inference_required_feature_keys(bundle)
    assert "reg_gate_target" not in required
    assert validate_live_feature_contract(bundle, strict=True)


def test_required_feature_keys_can_be_limited_to_deployment_strategies():
    bundle = {
        "bundle": {
            "alpha_models": {
                "long_kept": {"model": object(), "feat_cols": ["ret24h"]},
                "short_rejected": {
                    "model": object(),
                    "feat_cols": ["vov_mad_20_G_VOL_0"],
                },
            },
            "meta_models": {},
        }
    }

    required = get_inference_required_feature_keys(bundle, {"long_kept"})

    assert "ret24h" in required
    assert "vov_mad_20_G_VOL_0" not in required


def test_required_feature_keys_use_meta_contract_columns_not_positional_names():
    class _Meta:
        selected_features = ["f0", "f1"]
        meta_feature_columns_ = ["ret24h", "base_probability_long_rule"]

    bundle = {
        "bundle": {
            "alpha_models": {},
            "meta_models": {"long_rule": _Meta()},
        }
    }

    required = get_inference_required_feature_keys(bundle, {"long_rule"})

    assert "ret24h" in required
    assert "base_probability_long_rule" in required
    assert "f0" not in required


def test_live_contract_rejects_target_derived_active_alpha_features():
    bundle = {
        "bundle": {
            "alpha_models": {
                "long_rule": {
                    "model": object(),
                    "feat_cols": ["ret24h", "reg_gate_target"],
                }
            }
        }
    }
    with pytest.raises(ValueError, match="target-derived/unavailable"):
        validate_live_feature_contract(bundle, strict=True)
