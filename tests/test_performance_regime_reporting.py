import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.performance_regimes.labels import build_strategy_performance_labels
from extreme_price_movements.performance_regimes.archetypes import MarketStateArchetype
from extreme_price_movements.performance_regimes.diagnostics import PipelineStageReporter
from extreme_price_movements.performance_regimes.gatekeeping import (
    StageGateError,
    evaluate_stage_gate,
    gate_config_for_profile,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_performance_market_state_modulator.py"
SPEC = importlib.util.spec_from_file_location("run_performance_market_state_modulator", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)

REPORT_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "report_performance_market_state_modulator_runs.py"
)
REPORT_SPEC = importlib.util.spec_from_file_location(
    "report_performance_market_state_modulator_runs",
    REPORT_SCRIPT,
)
REPORT = importlib.util.module_from_spec(REPORT_SPEC)
assert REPORT_SPEC.loader is not None
REPORT_SPEC.loader.exec_module(REPORT)


def test_pipeline_stage_reporter_records_success_and_failure_rows():
    reporter = PipelineStageReporter()

    with reporter.stage("build_labels", fold=1, train_rows=20) as metrics:
        metrics["label_rows"] = 40

    with pytest.raises(RuntimeError):
        with reporter.stage("train_first_stage", fold=1) as metrics:
            metrics["model_count"] = 2
            raise RuntimeError("training failed")

    frame = reporter.to_frame()
    assert {"start", "end", "fail"}.issubset(set(frame["status"]))

    end_row = frame.loc[(frame["stage"] == "build_labels") & (frame["status"] == "end")].iloc[0]
    assert end_row["fold"] == 1
    assert end_row["train_rows"] == 20
    assert end_row["label_rows"] == 40
    assert end_row["duration_seconds"] >= 0.0

    fail_row = frame.loc[(frame["stage"] == "train_first_stage") & (frame["status"] == "fail")].iloc[0]
    assert fail_row["error_type"] == "RuntimeError"
    assert fail_row["error_message"] == "training failed"
    assert fail_row["model_count"] == 2

    summary = reporter.summary_frame()
    assert set(summary["stage"]) == {"build_labels", "train_first_stage"}
    assert set(summary["status"]) == {"end", "fail"}


def test_runner_defaults_to_per_head_pipeline_scope():
    parser = RUNNER.build_arg_parser()
    args = parser.parse_args(["--input", "in.parquet", "--output-dir", "out"])

    assert args.pipeline_scope == "per_head"
    assert args.first_stage_max_depth == 4


def test_first_stage_lgbm_config_is_cli_configurable():
    parser = RUNNER.build_arg_parser()
    args = parser.parse_args(
        [
            "--input",
            "in.parquet",
            "--output-dir",
            "out",
            "--first-stage-max-depth",
            "3",
            "--first-stage-num-leaves",
            "8",
            "--first-stage-min-child-samples-fraction",
            "0.05",
            "--first-stage-learning-rate",
            "0.02",
        ]
    )

    cfg = RUNNER._first_stage_lgbm_config(args)

    assert cfg.max_depth == 3
    assert cfg.num_leaves == 8
    assert cfg.min_child_samples_fraction == 0.05
    assert cfg.learning_rate == 0.02


def test_lagged_head_streak_risk_features_are_causal():
    ts = pd.date_range("2026-01-01", periods=6, freq="24h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy": ["s1"] * len(ts),
            "performance": [-0.01, -0.02, -0.03, -0.04, 0.01, -0.02],
        }
    )
    labels = build_strategy_performance_labels(
        frame,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        loss_streak_target_min_hours=48.0,
        loss_streak_target_full_hours=96.0,
    )

    features = RUNNER._lagged_head_streak_risk_features(
        labels,
        ts,
        strategies=["s1"],
        full_pressure_hours=96.0,
    )

    assert features.loc[ts[0], "s1__head_streak_hours_lag1"] == 0.0
    assert features.loc[ts[1], "s1__head_streak_hours_lag1"] == 24.0
    assert features.loc[ts[4], "s1__head_streak_pressure_lag1"] > 0.0
    assert features.loc[ts[5], "s1__head_streak_hours_lag1"] == 0.0
    assert features.loc[ts[0], "s1__head_streak_return_lag1"] == 0.0
    assert np.isclose(features.loc[ts[1], "s1__head_streak_return_lag1"], -0.01)
    assert features.loc[ts[4], "s1__head_streak_neg_share_24h_lag1"] == 1.0
    assert np.isclose(features.loc[ts[4], "s1__head_streak_loss_sum_72h_lag1"], 0.09)
    assert features.loc[ts[5], "s1__head_streak_neg_share_24h_lag1"] == 0.0


def test_per_head_orchestrator_runs_one_scope_per_strategy(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "strategy": ["s2", "s1", "s2", "s1"],
            "performance": [0.1, -0.1, 0.2, 0.3],
        }
    )
    calls = []

    monkeypatch.setattr(RUNNER, "_load_frame", lambda _path: frame)

    def fake_run_single_scope(args):
        calls.append((args.strategies, args.output_dir, args.pipeline_scope))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return {"folds": [{"fold": 1, "archetype_count": 1}]}

    monkeypatch.setattr(RUNNER, "_run_single_scope", fake_run_single_scope)
    args = RUNNER.build_arg_parser().parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    summary = RUNNER._run_per_head(args)

    assert summary["pipeline_scope"] == "per_head"
    assert [call[0] for call in calls] == ["s1", "s2"]
    assert all(call[2] == "global" for call in calls)
    assert calls[0][1].name == "head_s1"
    assert calls[1][1].name == "head_s2"
    assert (tmp_path / "out" / "performance_market_state_modulator_manifest.json").exists()


def test_quant_stage_gate_rejects_weak_leaf_stage_and_explains_failure():
    config = gate_config_for_profile("standard")
    decision = evaluate_stage_gate(
        "extract_score_prune_leaves",
        {
            "extracted_leaf_count": 10,
            "pruned_leaf_count": 0,
            "mean_pruned_leaf_stability": 0.0,
        },
        config,
    )

    assert not decision.passed
    assert "pruned_leaf_count" in decision.failures[0]
    with pytest.raises(StageGateError):
        raise StageGateError(decision)


def test_archetype_compression_merges_instead_of_discarding_source_leaf_ids():
    archetypes = tuple(
        MarketStateArchetype(
            archetype_id=f"strategy_s1_bad_archetype_{i}",
            strategy="s1",
            direction="bad",
            leaf_ids=(f"leaf_{i}",),
            dominant_features=(f"feature_{i % 2}",),
            dominant_feature_families=("feature",),
            total_weighted_coverage=0.1,
            mean_edge_mass=0.01 + i * 0.001,
            mean_contribution_share=0.1 + i,
            mean_stability=0.8,
            activation_timestamps=np.array([i % 2 == 0, i % 2 == 1, True, False]),
            diagnostics={},
        )
        for i in range(6)
    )

    compressed, report = RUNNER._select_archetypes_for_experts(archetypes, max_count=2)
    retained_leaf_ids = {leaf_id for archetype in compressed for leaf_id in archetype.leaf_ids}

    assert len(compressed) == 2
    assert retained_leaf_ids == {f"leaf_{i}" for i in range(6)}
    assert report["compressed_archetype_id"].nunique() == 2
    assert report["selected"].all()


def test_archetype_compression_preserves_strategy_direction_groups_when_cap_is_too_low():
    archetypes = tuple(
        MarketStateArchetype(
            archetype_id=f"strategy_{strategy}_{direction}_archetype_0",
            strategy=strategy,
            direction=direction,
            leaf_ids=(f"leaf_{strategy}_{direction}",),
            dominant_features=("feature",),
            dominant_feature_families=("feature",),
            total_weighted_coverage=0.1,
            mean_edge_mass=0.01,
            mean_contribution_share=0.1,
            mean_stability=0.8,
            activation_timestamps=np.array([True, False, True, False]),
            diagnostics={},
        )
        for strategy, direction in [
            ("s1", "bad"),
            ("s1", "good"),
            ("s2", "bad"),
        ]
    )

    compressed, report = RUNNER._select_archetypes_for_experts(archetypes, max_count=1)

    assert {(a.strategy, a.direction) for a in compressed} == {
        ("s1", "bad"),
        ("s1", "good"),
        ("s2", "bad"),
    }
    assert len(compressed) == 3
    assert report["compressed_archetype_id"].nunique() == 3


def test_archetype_compression_diagnostics_reports_silhouette_cov_and_coverage():
    archetypes = tuple(
        MarketStateArchetype(
            archetype_id=f"strategy_s1_bad_archetype_{i}",
            strategy="s1",
            direction="bad",
            leaf_ids=(f"leaf_{i}",),
            dominant_features=(f"feature_{i // 2}",),
            dominant_feature_families=("feature",),
            total_weighted_coverage=0.1 + 0.01 * i,
            mean_edge_mass=0.01 + i * 0.001,
            mean_contribution_share=0.1 + i,
            mean_stability=0.8,
            activation_timestamps=np.array([i < 2, i >= 2, True, False]),
            diagnostics={},
        )
        for i in range(4)
    )

    compressed, report = RUNNER._select_archetypes_for_experts(archetypes, max_count=2)
    report, diagnostics, metrics = RUNNER._build_archetype_compression_diagnostics(archetypes, report)

    assert len(compressed) == 2
    assert "compression_silhouette" in report.columns
    assert "member_count_cov" in diagnostics.columns
    assert metrics["compression_source_coverage_min"] == 1.0
    assert metrics["compression_group_count"] == 1.0
    assert np.isfinite(metrics["compression_silhouette_mean"])
    assert np.isfinite(metrics["compression_member_count_cov_max"])
    assert np.isfinite(metrics["compression_distance_to_seed_p95"])


def test_cluster_stage_gate_rejects_bad_compression_quality_metrics():
    config = gate_config_for_profile("standard")
    decision = evaluate_stage_gate(
        "cluster_archetypes",
        {
            "raw_archetype_count": 100,
            "archetype_count": 8,
            "compression_source_coverage_min": 1.0,
            "compression_silhouette_mean": -0.9,
            "compression_member_count_cov_max": 10.0,
            "compression_distance_to_seed_p95": 5.0,
        },
        config,
    )

    assert not decision.passed
    assert any("compression_silhouette_mean" in failure for failure in decision.failures)
    assert any("compression_member_count_cov_max" in failure for failure in decision.failures)


def test_market_state_feature_resolver_excludes_model_prediction_features():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "strategy": ["s1", "s1", "s1"],
            "performance": [0.1, -0.1, 0.2],
            "pred_base_model_score": [0.2, 0.3, 0.4],
            "pred_H5": [0.2, 0.3, 0.4],
            "pred_logit_H5": [0.1, 0.2, 0.3],
            "base_H5": [0.6, 0.5, 0.4],
            "base_prob_x_vol_regime": [0.7, 0.6, 0.5],
            "base_pred_delta": [0.1, 0.0, -0.1],
            "blend_B3_new_period_soft_qfail_score": [0.2, 0.1, 0.3],
            "oof_pred__head": [0.4, 0.5, 0.6],
            "base_lgbm_feature_drift_psi_core": [0.1, 0.2, 0.3],
            "regime_centroid_similarity_train_pc0": [0.7, 0.8, 0.9],
        }
    )

    defaults = RUNNER._default_feature_families(
        frame,
        timestamp_col="timestamp",
        strategy_col="strategy",
        performance_col="performance",
    )
    defaults, default_removed = RUNNER._sanitize_feature_families(defaults)
    assert defaults == {
        "general_market_state": [
            "base_lgbm_feature_drift_psi_core",
            "regime_centroid_similarity_train_pc0",
        ]
    }
    assert default_removed == [
        "base_H5",
        "base_pred_delta",
        "base_prob_x_vol_regime",
        "blend_B3_new_period_soft_qfail_score",
        "oof_pred__head",
        "pred_H5",
        "pred_base_model_score",
        "pred_logit_H5",
    ]

    sanitized, removed = RUNNER._sanitize_feature_families(
        {
            "explicit": [
                "pred_base_model_score",
                "base_prob_x_vol_regime",
                "blend_B3_new_period_soft_qfail_score",
                "base_lgbm_feature_drift_psi_core",
                "base_pred_delta",
                "oof_pred__head",
            ]
        }
    )
    assert sanitized == {"explicit": ["base_lgbm_feature_drift_psi_core"]}
    assert removed == [
        "base_pred_delta",
        "base_prob_x_vol_regime",
        "blend_B3_new_period_soft_qfail_score",
        "oof_pred__head",
        "pred_base_model_score",
    ]


def test_resolve_stage_gate_rejects_remaining_model_prediction_features():
    config = gate_config_for_profile("standard")
    decision = evaluate_stage_gate(
        "resolve_strategies_and_features",
        {
            "strategy_count": 4,
            "requested_feature_count": 20,
            "remaining_model_prediction_feature_count": 1,
        },
        config,
    )

    assert not decision.passed
    assert any("remaining_model_prediction_feature_count" in failure for failure in decision.failures)


def test_resolve_stage_gate_rejects_remaining_qfail_features():
    config = gate_config_for_profile("standard")
    decision = evaluate_stage_gate(
        "resolve_strategies_and_features",
        {
            "strategy_count": 4,
            "requested_feature_count": 20,
            "remaining_qfail_feature_count": 1,
        },
        config,
    )

    assert not decision.passed
    assert any("remaining_qfail_feature_count" in failure for failure in decision.failures)


def test_bad_regime_exposure_report_quantifies_tail_and_streak_avoidance():
    idx = pd.date_range("2026-01-01", periods=6, freq="24h", tz="UTC")
    returns = pd.DataFrame({"s1": [-1.0, -1.0, 0.5, -2.0, -2.0, -2.0]}, index=idx)
    baseline = pd.DataFrame({"s1": 1.0}, index=idx)
    policy = pd.DataFrame({"s1": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]}, index=idx)

    row = REPORT._bad_regime_exposure_rows(
        run_name="run",
        scope="s1",
        fold=1,
        policy="hard_gate_cutoff_0",
        weights=policy,
        baseline_weights=baseline,
        returns=returns,
    )[0]

    assert row["raw_max_loss_streak_hours"] == 72.0
    assert row["policy_max_active_loss_streak_hours"] == 24.0
    assert row["negative_return_active_share"] < row["negative_return_baseline_active_share"]
    assert row["worst_10pct_active_share"] == 0.0
    assert row["worst_10pct_delta_vs_baseline"] > 0.0
    assert row["prior_loss_streak_ge_24h_count"] > 0
    assert row["prior_loss_streak_ge_24h_active_share"] < row["prior_loss_streak_ge_24h_baseline_active_share"]
